"""E4 historical boundary branch or D3 single-head online skip diagnostics."""
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo


PROJECT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["LA", "Pancreas"], required=True)
    parser.add_argument("--data_root", type=Path, default=PROJECT.parents[1] / "Dataset")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--labelnum", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--labeled_bs", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=15000)
    parser.add_argument("--warmup_iterations", type=int, default=2000)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--consistency", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--val_start", type=int, default=800)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--variant", choices=["e4", "d3"], required=True)
    parser.add_argument("--boundary_supervision", action="store_true", default=True)
    parser.add_argument("--boundary_width", type=float, default=2.0)
    parser.add_argument("--boundary_weight", type=float)
    parser.add_argument("--boundary_start", type=int)
    parser.add_argument("--boundary_ramp", type=int)
    parser.add_argument("--gradient_interval", type=int, default=200)
    parser.add_argument("--skip_fraction", type=float, default=.125)
    parser.add_argument("--random_repeats", type=int, default=3)
    parser.add_argument("--intervention_epsilons", type=float, nargs="+", default=[.1, 1.])
    parser.add_argument("--boundary_cache", type=Path, default=PROJECT / "boundary_cache")
    # Small-patch runs are verification tools, never used by the experiment launcher.
    parser.add_argument("--patch_size", nargs=3, type=int)
    args = parser.parse_args()
    defaults = dict(e4=(.03, 0, 1), d3=(.015, 2000, 2000))[args.variant]
    for key, value in zip(("boundary_weight", "boundary_start", "boundary_ramp"), defaults):
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not 0 < args.skip_fraction <= 1 or args.random_repeats < 1:
        parser.error("Invalid channel fraction or random repeat count")
    if any(not 0 < x <= 1 for x in args.intervention_epsilons):
        parser.error("Intervention epsilon must be in (0,1]")
    if args.max_iterations <= 0 or args.warmup_iterations < 0:
        parser.error("Iterations must be positive, warmup must be nonnegative")
    if not 0 < args.labeled_bs < args.batch_size or args.num_workers < 0:
        parser.error("Require 0 < labeled_bs < batch_size and num_workers >= 0")
    if min(args.log_interval, args.checkpoint_interval, args.cpu_threads) <= 0:
        parser.error("Log/checkpoint intervals and cpu_threads must be positive")
    if args.val_interval < 0 or args.val_start < 0:
        parser.error("Validation interval and start must be nonnegative")
    if args.patch_size and any(size < 16 or size % 16 for size in args.patch_size):
        parser.error("Patch dimensions must be positive multiples of 16")
    import math
    if (not math.isfinite(args.boundary_width) or args.boundary_width <= 0
            or not math.isfinite(args.boundary_weight) or args.boundary_weight < 0
            or args.boundary_start < 0 or args.boundary_ramp <= 0 or args.gradient_interval <= 0):
        parser.error("Invalid boundary supervision/probe configuration")
    if not all(math.isfinite(x) for x in (args.base_lr, args.ema_decay, args.consistency)):
        parser.error("Loss and optimizer parameters must be finite")
    if args.base_lr <= 0 or args.consistency < 0:
        parser.error("Invalid learning rate or consistency weight")
    if not 0 <= args.ema_decay < 1:
        parser.error("ema_decay must be in [0, 1)")
    return args


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker_init(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def train(args):
    # CUDA visibility is set before importing torch, including transitive imports.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import copy
    import csv
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from data import SPECS, VolumeDataset, TwoStreamBatchSampler, split_manifest
    from evaluate import evaluate
    from networks.vnet_boundary_probe import VNetBoundaryProbe
    from objectives import supervised_loss, consistency_loss, update_ema
    from boundary import prepare_cache, boundary_weight
    from probe_objectives import morphology_boundary, probe_boundary_loss
    from online_probe import channel_interventions
    from evaluate_probe import evaluate_readout

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on the GPU host rather than the sandbox")
    args.data_root = args.data_root.resolve()
    spec = SPECS[args.dataset]
    root = args.data_root / spec["directory"]
    args.labelnum = args.labelnum if args.labelnum is not None else spec["labels"]
    args.patch_size = tuple(args.patch_size or spec["patch"])
    if not args.resume and args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    manifest = split_manifest(root, args.dataset, args.labelnum)
    cache = None
    if args.variant == "d3":
        cache = prepare_cache(root, args.dataset, manifest["labeled_ids"],
                              args.boundary_width, args.boundary_cache)
    dataset = VolumeDataset(root, args.dataset, args.patch_size, cache, args.labelnum)
    if not args.labeled_bs <= args.labelnum <= len(dataset) - (args.batch_size - args.labeled_bs):
        raise ValueError("Invalid labeled count for this dataset/batch size")
    sampler = TwoStreamBatchSampler(list(range(args.labelnum)),
                                    list(range(args.labelnum, len(dataset))),
                                    args.batch_size, args.labeled_bs)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    student = VNetBoundaryProbe(n_channels=1, n_classes=2, normalization="batchnorm",
                               has_dropout=True, variant=args.variant,
                               boundary_head_seed=args.seed + 1000003).to(args.device)
    teacher = copy.deepcopy(student).requires_grad_(False)
    optimizer = torch.optim.SGD(student.parameters(), lr=args.base_lr,
                                momentum=0.9, weight_decay=1e-4)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(experiment="MT_DS", consistency_loss="four_scale_softmax_mse",
                  supervision="four_scale_mean_ce_dice", lr_schedule="constant",
                  teacher_bn="independent_train_mode_running_stats_parameter_only_ema",
                  perturbation="independent_dropout3d_p0.5", normalization="none",
                  dataset_root=str(root), train_cases=len(dataset),
                  validation_split=spec["validation"], device_name=(
                      torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"))
    config.update(initialization="random_seed" if not args.resume else "same_experiment_resume",
                  validation_protocol="legacy_test_selection_exploratory",
                  surface_distance_unit="voxel", asd_definition="directed_medpy_asd")
    config.update(
        experiment=("E4_legacy_boundary_w003_v1" if args.variant == "e4" else
                    "D3_boundary_skip_w0015_v1"),
        boundary_head_seed=args.seed + 1000003,
        boundary_head_architecture=("conv3_bn_relu_conv1_128_32_1" if args.variant == "e4" else
                                    "conv3_relu_conv1_32_32_1_no_bn"),
        boundary_head_scales=["encoder_e4_eighth" if args.variant == "e4" else "decoder_d3_half"],
        boundary_target=("cropped_GT_3x3x3_local_label_change" if args.variant == "e4" else
                         "full_volume_euclidean_width2_union_area_half"),
        boundary_loss="positive_weighted_bce_plus_soft_dice",
        boundary_head_pos_weight="negative_over_positive_clamped_1_100",
        boundary_head_parameters=sum(p.numel() for n,p in student.named_parameters() if n.startswith("boundary_")),
        boundary_backbone_gradient=True,
        boundary_width_applies_to="d3_only", historical_morphology_radius=1,
        channel_interaction=False, memory_bank=False,
        inference="original_segmentation_head_only_no_boundary_or_donor_forward",
        diagnostic_state="cached_training_encoder_features_eval_decoder_before_SGD_no_state_updates",
        diagnostic_score=("E4_probe_absolute_gradient_times_feature" if args.variant == "e4" else
                          "D3_boundary_signed_negative_E2_skip_gate_derivative"),
        boundary_cache_manifest=str(cache / "manifest.json") if cache else None,
        baseline_dir=str(PROJECT / "Results" / f"seed_{args.seed}" / f"{args.dataset}_{args.labelnum}l"))
    iteration, best_dice = 0, -1.0
    if args.resume:
        saved_config = json.loads((args.output_dir / "config.json").read_text())
        for key in ("dataset", "labelnum", "seed", "batch_size", "labeled_bs", "patch_size",
                    "base_lr", "ema_decay", "consistency", "warmup_iterations", "normalization",
                    "experiment", "boundary_supervision", "boundary_width", "boundary_weight",
                    "boundary_start", "boundary_ramp", "gradient_interval", "boundary_loss",
                    "boundary_head_seed", "boundary_head_architecture", "boundary_head_scales",
                    "boundary_head_pos_weight", "variant", "skip_fraction", "random_repeats",
                    "intervention_epsilons"): 
            if json.dumps(config[key]) != json.dumps(saved_config[key]):
                raise ValueError(f"Resume configuration mismatch: {key}")
        saved_manifest = json.loads((args.output_dir / "data_split_manifest.json").read_text())
        if saved_manifest != manifest:
            raise ValueError("Resume dataset split mismatch")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iteration, best_dice = checkpoint["iteration"], checkpoint["best_dice"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if args.device == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "config.json", config)
        write_json(args.output_dir / "data_split_manifest.json", manifest)
        shutil.copytree(PROJECT / "code", args.output_dir / "code",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(PROJECT / "provenance.json", args.output_dir / "provenance.json")
        import hashlib
        source_hashes = {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted((PROJECT / "code").rglob("*.py"))}
        write_json(args.output_dir / "source_manifest.json", source_hashes)
        if cache is not None:
            shutil.copyfile(cache / "manifest.json", args.output_dir / "boundary_target_manifest.json")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(args.output_dir / "log.txt"),
                                  logging.StreamHandler(sys.stdout)])
    writer = SummaryWriter(str(args.output_dir / "tensorboard"))
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
                        pin_memory=args.device == "cuda", worker_init_fn=worker_init)
    common = dict(dataset=args.dataset, labelnum=args.labelnum, seed=args.seed,
                  gpu=args.gpu, pid=os.getpid(), max_iterations=args.max_iterations,
                  experiment=config["experiment"])
    def status(state, **extra):
        write_json(args.output_dir / "run_status.json", dict(
            common, state=state, iteration=iteration, updated_at=timestamp(),
            best_dice=best_dice if best_dice >= 0 else None, **extra))

    def save_checkpoint():
        temporary = args.output_dir / "last_checkpoint.pth.tmp"
        torch.save(dict(student=student.state_dict(), teacher=teacher.state_dict(),
                        optimizer=optimizer.state_dict(), iteration=iteration, best_dice=best_dice,
                        config=config, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(), cuda_rng=(torch.cuda.get_rng_state_all()
                            if args.device == "cuda" else None)), temporary)
        temporary.replace(args.output_dir / "last_checkpoint.pth")

    status("training")
    logging.info("config: %s", config)
    start = time.monotonic()
    start_iteration = iteration
    try:
        with (args.output_dir / "training_metrics.csv").open("a", newline="", buffering=1) as handle:
            fields = ["iteration", "loss_supervised", "ce", "dice", "loss_consistency",
                      "consistency_weight", "loss_total", "gradient_norm", "lr",
                      "peak_memory_mib", "peak_reserved_mib", "elapsed_seconds"]
            fields += ["loss_boundary", "boundary_weight", "empty_foreground_fraction",
                       "boundary_band_fraction", "boundary_bce", "boundary_dice",
                       "boundary_valid_samples", "boundary_positive_mass", "boundary_pos_weight"]
            metrics_writer = csv.DictWriter(handle, fieldnames=fields)
            if handle.tell() == 0:
                metrics_writer.writeheader()
            while iteration < args.max_iterations:
                student.train()
                teacher.train()
                for batch in loader:
                    images = batch["image"].to(args.device, non_blocking=True)
                    # Only labeled GT enters either objective.
                    labels = batch["label"][:args.labeled_bs].to(args.device, non_blocking=True)
                    supervised_outputs, boundary_outputs, probe_features = student(
                        images[:args.labeled_bs], deepsup=True, return_boundary=True)
                    unlabeled_outputs = student(images[args.labeled_bs:], deepsup=True)
                    loss_sup, ce, dice = supervised_loss(supervised_outputs, labels)
                    with torch.no_grad():
                        teacher_outputs = teacher(images[args.labeled_bs:], deepsup=True)
                    loss_mt = consistency_loss(unlabeled_outputs, teacher_outputs)
                    weight = args.consistency if iteration >= args.warmup_iterations else 0.0
                    loss = loss_sup
                    if weight > 0:
                        loss = loss + weight * loss_mt
                    if args.variant == "e4":
                        target = morphology_boundary(labels)
                    else:
                        bands = batch["boundary_band"][:args.labeled_bs].to(args.device, non_blocking=True)
                        target = (bands[:, None] > 0).float()
                    loss_boundary, boundary_stats = probe_boundary_loss(boundary_outputs, target, args.variant)
                    boundary_coefficient = boundary_weight(
                        iteration, args.boundary_weight, args.boundary_start, args.boundary_ramp)
                    if boundary_coefficient > 0:
                        loss = loss + boundary_coefficient * loss_boundary
                    if not torch.isfinite(loss_boundary):
                        raise FloatingPointError("Nonfinite boundary loss")
                    if not torch.isfinite(torch.stack([loss_sup, loss_mt, loss])).all():
                        raise FloatingPointError("Nonfinite supervised/consistency/total loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    # Observe all gradients without altering their magnitude.
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        student.parameters(), float("inf"), error_if_nonfinite=True)
                    if iteration == 0 or (iteration + 1) % args.gradient_interval == 0:
                        probe = channel_interventions(
                            student, probe_features, labels, target, step=iteration + 1,
                            seed=args.seed, fraction=args.skip_fraction,
                            random_repeats=args.random_repeats, epsilons=args.intervention_epsilons)
                        probe.update(boundary_weight=boundary_coefficient,
                                     head_learning_active=boundary_coefficient > 0)
                        with (args.output_dir / "channel_interventions.jsonl").open("a") as probe_log:
                            probe_log.write(json.dumps(probe, allow_nan=False) + "\n")
                        writer.add_scalar("probe/score_norm", probe["score_norm"], iteration + 1)
                    del probe_features
                    optimizer.step()
                    update_ema(student, teacher, args.ema_decay, iteration)
                    iteration += 1
                    if iteration == 1 or iteration % args.log_interval == 0 or iteration == args.max_iterations:
                        elapsed = time.monotonic() - start
                        peak = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0
                        row = dict(iteration=iteration, loss_supervised=loss_sup.item(), ce=ce.item(),
                                   dice=dice.item(), loss_consistency=loss_mt.item(), consistency_weight=weight,
                                   loss_total=loss.item(), gradient_norm=gradient_norm.item(),
                                   lr=optimizer.param_groups[0]["lr"], peak_memory_mib=peak,
                                   peak_reserved_mib=(torch.cuda.max_memory_reserved() / 2**20 if args.device == "cuda" else 0),
                                   elapsed_seconds=elapsed)
                        if args.boundary_supervision:
                            row.update(boundary_stats, loss_boundary=loss_boundary.item(),
                                       boundary_weight=boundary_coefficient,
                                       empty_foreground_fraction=float((labels.flatten(1).sum(1) == 0).float().mean()),
                                       boundary_band_fraction=float(target.mean()))
                        metrics_writer.writerow(row)
                        for key, value in row.items():
                            if key != "iteration":
                                writer.add_scalar(f"train/{key}", value, iteration)
                        eta = elapsed / max(iteration - start_iteration, 1) * (args.max_iterations - iteration)
                        status("training", **{key: value for key, value in row.items() if key != "iteration"},
                               eta_hours=eta / 3600)
                        logging.info("iteration %d: sup=%.6f mt=%.6f weight=%.3f grad=%.4f memory=%.0fMiB eta=%.2fh",
                                     iteration, row["loss_supervised"], row["loss_consistency"], weight,
                                     row["gradient_norm"], peak, eta / 3600)
                        if args.boundary_supervision:
                            logging.info("boundary %s: loss=%.6f coefficient=%.6f bce=%.6f dice=%.6f band_fraction=%.4f",
                                         args.variant, row["loss_boundary"], boundary_coefficient,
                                         row["boundary_bce"], row["boundary_dice"], row["boundary_band_fraction"])
                    if args.val_interval and iteration >= args.val_start and iteration % args.val_interval == 0:
                        status("validating")
                        result = evaluate(student, root, args.dataset, args.patch_size, args.device)
                        result["iteration"] = iteration
                        boundary_evaluation = evaluate_readout(student, root, args.dataset, args.patch_size, args.device)
                        boundary_evaluation["iteration"] = iteration
                        write_json(args.output_dir / "latest_boundary_validation.json", boundary_evaluation)
                        with (args.output_dir / "boundary_validation_metrics.jsonl").open("a") as probe_log:
                            probe_log.write(json.dumps(boundary_evaluation, allow_nan=False) + "\n")
                        write_json(args.output_dir / "latest_validation.json", result)
                        with (args.output_dir / "validation_metrics.jsonl").open("a") as validation_log:
                            validation_log.write(json.dumps(result, allow_nan=False) + "\n")
                        writer.add_scalar("validation/dice", result["dice"], iteration)
                        logging.info("validation iteration %d: Dice=%.6f", iteration, result["dice"])
                        if result["dice"] > best_dice:
                            best_dice = result["dice"]
                            torch.save(student.state_dict(), args.output_dir / "best_model.pth")
                            torch.save(teacher.state_dict(), args.output_dir / "teacher_at_best_student.pth")
                            write_json(args.output_dir / "best_metrics.json", result)
                        status("training")
                    if iteration == 1 or iteration % args.checkpoint_interval == 0 or iteration == args.max_iterations:
                        save_checkpoint()
                    if iteration >= args.max_iterations:
                        break
        if (args.output_dir / "best_model.pth").is_file():
            student.load_state_dict(torch.load(args.output_dir / "best_model.pth",
                                               map_location=args.device, weights_only=True))
            status("testing_best_checkpoint")
            result = evaluate(student, root, args.dataset, args.patch_size, args.device, split="test")
            write_json(args.output_dir / "test_metrics.json", result)
        status("completed")
    except BaseException as exc:
        status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
               error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    train(arguments())
