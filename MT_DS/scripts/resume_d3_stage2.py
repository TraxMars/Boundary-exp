"""Resume one stopped D3-stage2 job and adopt the existing six-job monitor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from resume_a1 import digest, trim_metrics
from run_a1_pair import read_json, stamp, write_json
from run_d3_stage2 import COMPARISON, PROJECT, gpu_free, hashes, snapshot


TRAINER = PROJECT / 'code/train_d3_stage2.py'


def matching_training(pid, output):
    try:
        proc = Path('/proc') / str(pid)
        command = (proc / 'cmdline').read_bytes().split(b'\0')
        return (str(TRAINER).encode() in command and str(output).encode() in command
                and (proc / 'stat').read_text().split(') ')[1][0] != 'Z')
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run_dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    with (COMPARISON / '.recovery.lock').open('a') as recovery_lock:
        fcntl.flock(recovery_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read_json(COMPARISON / 'full_launch_manifest.json')
        target = next(run for run in manifest['runs'] if Path(run['output_dir']) == output)
        assert not any(run['state'] == 'queued' for run in manifest['runs']), 'Pending scheduler jobs require separate handoff'
        assert hashes() == manifest['source_hashes'], 'Training sources changed'
        old_status = read_json(output / 'run_status.json')
        assert old_status['state'] != 'completed'
        for proc in Path('/proc').iterdir():
            if proc.name.isdigit():
                assert not matching_training(int(proc.name), output), 'Requested training is already active'
        inventory = gpu_free()
        assert inventory[args.gpu]['free_mib'] >= max(8192, target['required_free_mib']), inventory
        assert not any(run['gpu'] == args.gpu and matching_training(run['pid'], run['output_dir'])
                       for run in manifest['runs']), 'GPU already has another job in this comparison'
        config = read_json(output / 'config.json')
        for rel, expected in read_json(output / 'source_manifest.json').items():
            assert digest(PROJECT / rel) == expected, rel
        for entry in read_json(output / 'data_split_manifest.json')['splits'].values():
            assert digest(Path(entry['path'])) == entry['sha256'], entry['path']
        import torch
        torch.set_num_threads(4)
        checkpoint = output / 'last_checkpoint.pth'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        step = int(saved['iteration'])
        assert 0 < step < config['max_iterations'] == 15000
        for key, value in config.items():
            if key not in ('gpu', 'resume', 'initialization'):
                assert json.dumps(saved['config'][key]) == json.dumps(value), key
        for key in ('student', 'teacher', 'optimizer', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'):
            assert key in saved, key
        for role in ('student', 'teacher'):
            assert all(torch.isfinite(tensor).all() for tensor in saved[role].values()), role
        assert all(torch.isfinite(state['momentum_buffer']).all()
                   for state in saved['optimizer']['state'].values())
        best = read_json(output / 'best_metrics.json')
        assert best['iteration'] <= step and abs(best['dice'] - saved['best_dice']) < 1e-10
        assert all((output / name).is_file() for name in ('best_model.pth', 'teacher_at_best_student.pth'))
        del saved
        args.run_dir.mkdir(parents=True, exist_ok=True)
        assert not (args.run_dir / 'resume_record.json').exists(), 'Recovery directory already used'
        # Replace only the scheduler PID. All trainers were launched in their own
        # sessions, and the scheduler has no handler that terminates its children.
        previous_supervisor = manifest['supervisor_pid']
        try:
            old_args = (Path('/proc') / str(previous_supervisor) / 'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            old_args = []
        if old_args:
            allowed = [PROJECT / 'scripts/run_d3_stage2.py', Path(__file__).resolve()]
            assert any(str(path).encode() in old_args for path in allowed), 'Unexpected supervisor process'
            os.kill(previous_supervisor, signal.SIGTERM)
        with (COMPARISON / '.launcher.lock').open('a') as launch_lock:
            deadline = time.monotonic() + 10
            while True:
                try:
                    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Previous scheduler did not release its lock')
                    time.sleep(.2)
            manifest = read_json(COMPARISON / 'full_launch_manifest.json')
            target = next(run for run in manifest['runs'] if Path(run['output_dir']) == output)
            archive = output / 'resume_history' / args.run_dir.name
            archive.mkdir(parents=True, exist_ok=False)
            metric_files = ['training_metrics.csv', *[path.name for path in sorted(output.glob('*.jsonl'))]]
            files = set(metric_files) | {'run_status.json', 'config.json', 'log.txt', 'last_checkpoint.pth',
                                        'best_model.pth', 'teacher_at_best_student.pth', 'best_metrics.json',
                                        'latest_validation.json', 'latest_boundary_validation.json'}
            for name in files:
                if (output / name).is_file():
                    shutil.copy2(output / name, archive / name)
            for name in ('full_launch_manifest.json', 'supervisor.json', 'summary.json'):
                if (COMPARISON / name).is_file():
                    shutil.copy2(COMPARISON / name, archive / ('comparison_' + name))
            removed = {name: trim_metrics(output / name, step) for name in metric_files}
            for journal, latest in (('validation_metrics.jsonl', 'latest_validation.json'),
                                    ('boundary_validation_metrics.jsonl', 'latest_boundary_validation.json')):
                path = output / journal
                if path.is_file() and path.read_text().strip():
                    write_json(output / latest, json.loads(path.read_text().splitlines()[-1]))
            if (output / 'tensorboard').exists():
                (output / 'tensorboard').rename(archive / 'tensorboard_before_resume')
            command = list(target['command'])
            command[command.index('--gpu') + 1] = str(args.gpu)
            if '--resume' in command:
                command[command.index('--resume') + 1] = str(checkpoint)
            else:
                command += ['--resume', str(checkpoint)]
            record = dict(resumed_at=stamp(), output_dir=str(output), checkpoint=str(checkpoint),
                          checkpoint_iteration=step, checkpoint_sha256=digest(checkpoint),
                          previous_status=old_status, previous_attempt=dict(target),
                          removed_live_rows=removed, archive_dir=str(archive), gpu=args.gpu,
                          gpu_free_before_launch=inventory[args.gpu], command=command,
                          old_supervisor_pid=previous_supervisor, supervisor_pid=os.getpid(),
                          training_source_verified=True,
                          note='Original model/optimizer/RNG restored; sampler and worker prefetch are not checkpointed.')
            write_json(archive / 'resume_record.json', record)
            with (output / 'log.txt').open('a') as log:
                log.write(f'\nRESUME {stamp()}: restoring step {step} on GPU{args.gpu}; archive {archive}.\n')
            log_path = args.run_dir / f'{target["variant"]}_{target["dataset"]}_{target["labelnum"]}l.log'
            environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                               OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            with log_path.open('x') as log:
                child = subprocess.Popen(command, cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            write_json(archive / 'resume_record.json', record)
            write_json(args.run_dir / 'resume_record.json', record)
            target.update(command=command, pid=child.pid, gpu=args.gpu, launch_log=str(log_path),
                          exit_code=None, state='resuming', resume_record=str(archive / 'resume_record.json'))
            write_json(output / 'run_status.json', dict(dataset=config['dataset'], labelnum=config['labelnum'],
                       seed=config['seed'], gpu=str(args.gpu), pid=child.pid, max_iterations=15000,
                       experiment=config['experiment'], state='resuming', iteration=step,
                       updated_at=stamp(), best_dice=best['dice']))
            manifest.update(supervisor_pid=os.getpid(), state='running')
            manifest.setdefault('resume_history', []).append(record)
            write_json(COMPARISON / 'supervisor.json', dict(pid=os.getpid(), phase='resume',
                       started_at=stamp(), log_dir=str(args.run_dir), command=[sys.executable, *sys.argv]))
            print(json.dumps({key: record[key] for key in ('resumed_at', 'checkpoint_iteration', 'gpu', 'pid')}, indent=2), flush=True)
            fcntl.flock(recovery_lock, fcntl.LOCK_UN)
            while True:
                live = False
                for run in manifest['runs']:
                    state = read_json(Path(run['output_dir']) / 'run_status.json')
                    code = child.poll() if run is target else None
                    running = (code is None if run is target else matching_training(run['pid'], run['output_dir']))
                    if running:
                        live = True
                        run.update(state=state.get('state', 'initializing'), exit_code=None)
                    else:
                        complete = state.get('state') == 'completed' and state.get('iteration') == 15000
                        run.update(state='completed' if complete and code in (None, 0) else 'failed',
                                   exit_code=code if code is not None else (0 if complete else run.get('exit_code') or 1))
                manifest['state'] = ('running' if live else 'completed' if all(
                    run['state'] == 'completed' for run in manifest['runs']) else 'failed')
                snapshot(manifest, args.run_dir)
                if not live:
                    break
                time.sleep(30)
            if manifest['state'] != 'completed':
                raise RuntimeError('One or more comparison jobs did not complete; inspect per-run records')


if __name__ == '__main__':
    main()
