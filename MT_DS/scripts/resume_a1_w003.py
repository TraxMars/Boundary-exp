"""Resume one interrupted 0.03 run and adopt monitoring of its running partner."""
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
from run_a1_w003_pair import EXPERIMENT, PROJECT, update


def matching_training(pid, output):
    try:
        proc = Path('/proc') / str(pid)
        args = (proc / 'cmdline').read_bytes().split(b'\0')
        return (str(PROJECT / 'code/train.py').encode() in args
                and str(output).encode() in args
                and (proc / 'stat').read_text().split(') ')[1][0] != 'Z')
    except OSError:
        return False


class AdoptedTraining:
    def __init__(self, run):
        self.run = run
        self.returncode = None

    def poll(self):
        output = Path(self.run['output_dir'])
        if matching_training(self.run['pid'], output):
            self.returncode = None
        else:
            status = read_json(output / 'run_status.json')
            self.returncode = (0 if status.get('state') == 'completed'
                               and status.get('iteration') == 15000
                               and (output / 'test_metrics.json').is_file() else 1)
        return self.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('LA', 'Pancreas'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run_dir', type=Path, required=True)
    args = parser.parse_args()
    # Serialize recovery attempts before handing off the original launcher's lock.
    with (EXPERIMENT / '.resume.lock').open('a') as recovery_lock:
        fcntl.flock(recovery_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read_json(EXPERIMENT / 'full_launch_manifest.json')
        assert manifest['condition'] == 'A1_boundary_supervision_w003_v1'
        active = next(run for run in manifest['runs'] if run['dataset'] == args.dataset)
        output = Path(active['output_dir'])
        old_status = read_json(output / 'run_status.json')
        assert old_status.get('state') != 'completed'
        assert not matching_training(active['pid'], output)
        assert not matching_training(old_status.get('pid'), output)
        config = read_json(output / 'config.json')
        assert config['boundary_weight'] == .03 and config['checkpoint_interval'] == 100
        for rel, expected in read_json(output / 'source_manifest.json').items():
            assert digest(PROJECT / rel) == expected, ('Training source changed', rel)
        split = read_json(output / 'data_split_manifest.json')
        for entry in split['splits'].values():
            assert digest(Path(entry['path'])) == entry['sha256'], entry['path']
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                       '--format=csv,noheader,nounits'], text=True)
        free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in raw.splitlines()}
        assert free.get(args.gpu, 0) >= 14000, ('Insufficient free GPU memory', free)
        import torch
        torch.set_num_threads(4)
        checkpoint = output / 'last_checkpoint.pth'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        step = int(saved['iteration'])
        assert 0 < step < config['max_iterations'] == 15000
        assert json.dumps(saved['config'], sort_keys=True) == json.dumps(config, sort_keys=True)
        best = read_json(output / 'best_metrics.json')
        assert best['iteration'] <= step and abs(best['dice'] - saved['best_dice']) < 1e-10
        for role, multiplier in (('student', 2), ('teacher', 1)):
            assert all(torch.isfinite(tensor).all() for tensor in saved[role].values())
            assert all(int(tensor) == multiplier * step for key, tensor in saved[role].items()
                       if key.endswith('num_batches_tracked'))
        assert all(torch.isfinite(state['momentum_buffer']).all()
                   for state in saved['optimizer']['state'].values())
        assert all((output / name).is_file() for name in ('best_model.pth', 'teacher_at_best_student.pth'))
        del saved
        args.run_dir.mkdir(parents=True, exist_ok=True)
        assert not (args.run_dir / 'resume_record.json').exists(), 'Resume directory already used'
        # The old supervisor has no cleanup handler and its workers use independent
        # sessions. Signal only that verified supervisor PID, never a process group.
        old_supervisor = manifest['supervisor_pid']
        try:
            old_args = (Path('/proc') / str(old_supervisor) / 'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            old_args = []
        if old_args:
            assert str(PROJECT / 'scripts/run_a1_w003_pair.py').encode() in old_args, 'Unexpected live supervisor'
            os.kill(old_supervisor, signal.SIGTERM)
        with (EXPERIMENT / '.launcher.lock').open('a') as launch_lock:
            deadline = time.monotonic() + 10
            while True:
                try:
                    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Original supervisor did not release its lock')
                    time.sleep(.2)
            manifest = read_json(EXPERIMENT / 'full_launch_manifest.json')
            active = next(run for run in manifest['runs'] if run['dataset'] == args.dataset)
            archive = output / 'resume_history' / args.run_dir.name
            archive.mkdir(parents=True, exist_ok=False)
            for name in ('run_status.json', 'config.json', 'training_metrics.csv',
                         'validation_metrics.jsonl', 'boundary_gradient_metrics.jsonl',
                         'latest_validation.json', 'best_metrics.json', 'log.txt',
                         'last_checkpoint.pth', 'best_model.pth', 'teacher_at_best_student.pth'):
                if (output / name).is_file():
                    shutil.copy2(output / name, archive / name)
            for name in ('full_launch_manifest.json', 'supervisor.json', 'summary.json'):
                if (EXPERIMENT / name).is_file():
                    shutil.copy2(EXPERIMENT / name, archive / name)
            removed = {name: trim_metrics(output / name, step) for name in (
                'training_metrics.csv', 'validation_metrics.jsonl', 'boundary_gradient_metrics.jsonl')}
            validation = output / 'validation_metrics.jsonl'
            if validation.is_file() and validation.read_text().strip():
                write_json(output / 'latest_validation.json', json.loads(validation.read_text().splitlines()[-1]))
            if (output / 'tensorboard').exists():
                (output / 'tensorboard').rename(archive / 'tensorboard_before_resume')
            command = list(active['command'])
            command[command.index('--gpu') + 1] = str(args.gpu)
            if '--resume' in command:
                command[command.index('--resume') + 1] = str(checkpoint)
            else:
                command += ['--resume', str(checkpoint)]
            record = dict(dataset=args.dataset, resumed_at=stamp(), checkpoint=str(checkpoint),
                          checkpoint_iteration=step, checkpoint_sha256=digest(checkpoint),
                          previous_logged_iteration=old_status['iteration'], removed_live_rows=removed,
                          gpu=args.gpu, command=command, archive_dir=str(archive),
                          old_supervisor_pid=old_supervisor, supervisor_pid=os.getpid(),
                          previous_attempt=dict(pid=active['pid'], exit_code=active.get('exit_code'),
                                                command=active['command'], launch_log=active['launch_log']),
                          note='Same optimizer/EMA/objectives/schedule; sampler and worker prefetch are not exactly restored.')
            write_json(archive / 'resume_record.json', record)
            with (output / 'log.txt').open('a') as log:
                log.write(f'\nRESUME {stamp()}: restoring step {step}; prior logged step '
                          f'{old_status["iteration"]}; archived at {archive}.\n')
            log_path = args.run_dir / f'{args.dataset}_{active["labelnum"]}l.log'
            environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                               OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            with log_path.open('x') as log:
                child = subprocess.Popen(command, cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            write_json(archive / 'resume_record.json', record)
            write_json(args.run_dir / 'resume_record.json', record)
            active.update(command=command, pid=child.pid, gpu=args.gpu, launch_log=str(log_path),
                          exit_code=None, state='resuming', resume_record=str(archive / 'resume_record.json'))
            manifest.update(supervisor_pid=os.getpid(), state='running')
            manifest.setdefault('resume_history', []).append(record)
            write_json(EXPERIMENT / 'supervisor.json', dict(pid=os.getpid(), phase='resume',
                       dataset=args.dataset, started_at=stamp(), log_dir=str(args.run_dir),
                       command=[sys.executable, *sys.argv]))
            children = [child if run is active else AdoptedTraining(run) for run in manifest['runs']]
            print(json.dumps(record, indent=2), flush=True)
            while True:
                update(manifest, children, args.run_dir)
                if manifest['state'] != 'running':
                    break
                time.sleep(30)
            if manifest['state'] != 'completed':
                raise RuntimeError(f'Pair ended with an incomplete member; inspect {args.run_dir}')


if __name__ == '__main__':
    main()
