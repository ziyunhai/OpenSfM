# pyre-strict
"""SfM pipeline execution for benchmarking."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from benchmark.config import BenchmarkConfig
from benchmark.workspace import (
    cleanup_local_dir,
    stage_dataset_local,
    sync_dataset_to_nas,
)

logger: logging.Logger = logging.getLogger(__name__)

PIPELINE_STEPS: List[str] = [
    "extract_metadata",
    "detect_features",
    "match_features",
    "create_tracks",
    "reconstruct",
    "compute_statistics",
    "export_report",
]

# Dense-reconstruction stages (opt-in via --dense).  ``undistort`` is the
# prerequisite that produces the undistorted dataset the dense stages consume.
# They are inserted right before ``compute_statistics`` so the latter picks up
# their per-stage timing reports into stats.json (and thus the HTML report).
DENSE_STEPS: List[str] = [
    "undistort",
    "dense_clustering",
    "dense_equalize",
    "compute_depthmaps",
    "fuse_depthmaps",
    "dense_merging",
]


def build_pipeline_steps(dense: bool = False) -> List[str]:
    """Return the ordered pipeline steps for a run.

    When ``dense`` is True the dense stages (and their ``undistort``
    prerequisite) are spliced in just before ``compute_statistics`` so their
    timings flow into stats.json like every other step.
    """
    if not dense:
        return list(PIPELINE_STEPS)
    idx = PIPELINE_STEPS.index("compute_statistics")
    return PIPELINE_STEPS[:idx] + DENSE_STEPS + PIPELINE_STEPS[idx:]


# Superset of every step that can appear in any run (used for argparse choices).
ALL_PIPELINE_STEPS: List[str] = build_pipeline_steps(dense=True)

# For each step, the path relative to dataset_path that indicates completion.
# Directories are considered complete when non-empty.
STEP_OUTPUT_FILES: Dict[str, str] = {
    "extract_metadata": "exif",
    "detect_features": "features",
    "match_features": "reports/matches.json",
    "create_tracks": "tracks.csv",
    "reconstruct": "reconstruction.json",
    "undistort": "undistorted/reconstruction.json",
    "dense_equalize": "reports/dense_equalize.json",
    "dense_clustering": "reports/dense_clustering.json",
    "compute_depthmaps": "reports/dense_depthmaps.json",
    "fuse_depthmaps": "reports/dense_fusion.json",
    "dense_merging": "reports/dense_merging.json",
    "compute_statistics": "stats/stats.json",
    "export_report": "stats/report.pdf",
}

# All files/directories a step produces that a subsequent step may need.
# Used to bootstrap a new run from an existing one.  The dense stages write
# their artefacts inside ``undistorted/`` (carried over wholesale by the
# undistort symlink), so only their report files are listed here.
STEP_OUTPUTS: Dict[str, List[str]] = {
    "extract_metadata": ["exif", "camera_models.json"],
    "detect_features": ["features"],
    "match_features": ["matches", "reports/matches.json"],
    "create_tracks": ["tracks.csv", "reports/tracks.json"],
    "reconstruct": ["reconstruction.json", "reports/reconstruction.json"],
    "undistort": ["undistorted"],
    "dense_equalize": ["reports/dense_equalize.json"],
    "dense_clustering": ["reports/dense_clustering.json"],
    "compute_depthmaps": ["reports/dense_depthmaps.json"],
    "fuse_depthmaps": ["reports/dense_fusion.json"],
    "dense_merging": ["reports/dense_merging.json"],
    "compute_statistics": ["stats"],
    "export_report": [],
}

# Extra CLI arguments appended to specific steps' invocation.  dense_merging is
# run georeferenced so the LAS/LAZ and DSM/ortho products land in the output
# coordinate system.
STEP_EXTRA_ARGS: Dict[str, List[str]] = {
    "dense_merging": ["--georeferenced"],
}


def _set_oom_score_adj(score: int) -> None:
    """Write an OOM score adjustment to /proc/self/oom_score_adj.

    Called as preexec_fn in pipeline subprocesses so the kernel preferentially
    kills them under memory pressure rather than the benchmark orchestrator.
    """
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(str(score))
    except OSError:
        pass


def _make_oom_preexec(score: int = 500) -> Callable[[], None]:
    def _preexec() -> None:
        _set_oom_score_adj(score)
    return _preexec


def protect_self_from_oom(score: int = -500) -> None:
    """Lower the current process's OOM score so it survives memory pressure."""
    _set_oom_score_adj(score)


def _refresh_disk_cache(dataset_path: str) -> None:
    """
    Force NFS attribute-cache refresh on key dataset files.
    """
    for name in ("image_list.txt", "config.yaml"):
        p = os.path.join(dataset_path, name)
        try:
            os.stat(p)
            # Touch-read a few bytes to force the NFS client to revalidate
            with open(p, "rb") as f:
                f.read(1)
        except OSError:
            pass
    # Also stat the directory itself
    try:
        os.listdir(dataset_path)
    except OSError:
        pass


def bootstrap_dataset(
    source_dataset_dir: str,
    target_dataset_dir: str,
    from_step: str,
    dense: bool = False,
) -> None:
    """Populate target_dataset_dir with outputs of all steps before from_step.

    Directories are symlinked; individual files are copied.
    The target dataset directory must already exist (image_list.txt etc. in place).
    """
    steps = build_pipeline_steps(dense)
    from_idx = steps.index(from_step)
    steps_to_copy = steps[:from_idx]

    old_umask = os.umask(0o000)
    try:
        for step in steps_to_copy:
            for rel_path in STEP_OUTPUTS.get(step, []):
                src = os.path.join(source_dataset_dir, rel_path)
                dst = os.path.join(target_dataset_dir, rel_path)

                if not os.path.exists(src):
                    continue

                # Ensure parent directory exists
                parent = os.path.dirname(dst)
                if parent:
                    os.makedirs(parent, exist_ok=True, mode=0o777)

                if os.path.lexists(dst):
                    continue  # already bootstrapped

                if os.path.isdir(src):
                    os.symlink(src, dst)
                    logger.info("  bootstrap symlink: %s -> %s", dst, src)
                else:
                    import shutil
                    shutil.copy2(src, dst)
                    logger.info("  bootstrap copy:    %s", rel_path)
    finally:
        os.umask(old_umask)


def is_step_complete(step: str, dataset_path: str) -> bool:
    """Return True if the step's output already exists in the dataset directory."""
    rel = STEP_OUTPUT_FILES.get(step)
    if not rel:
        return False
    full_path = os.path.join(dataset_path, rel)
    if os.path.isdir(full_path):
        return bool(os.listdir(full_path))
    return os.path.isfile(full_path)


def save_run_meta(run_meta: Dict[str, Any], run_dir: str) -> None:
    meta_path = os.path.join(run_dir, "run_meta.json")
    old_umask = os.umask(0o000)
    try:
        with open(meta_path, "w") as f:
            json.dump(run_meta, f, indent=2)
        os.chmod(meta_path, 0o666)
    finally:
        os.umask(old_umask)


def run_pipeline(
    opensfm_bin: str,
    dataset_path: str,
    conda_env: str,
    resume: bool = False,
    from_step: Optional[str] = None,
    dense: bool = False,
) -> Dict[str, Any]:
    """Run the SfM pipeline on a single dataset.

    Args:
        resume:    If True and from_step is None, skip steps whose outputs
                   already exist (crash recovery).
        from_step: If set, steps *before* this step are skipped unconditionally
                   (assumed bootstrapped or already complete); this step and all
                   subsequent steps are always run.
        dense:     If True, also run the dense-reconstruction stages.

    Returns a dict with per-step timings and success/failure status.
    """
    steps = build_pipeline_steps(dense)
    from_idx = steps.index(from_step) if from_step else 0

    result: Dict[str, Any] = {
        "success": True,
        "steps": {},
        "failed_step": None,
    }

    for step in steps:
        step_idx = steps.index(step)
        ds_name = os.path.basename(dataset_path)

        # Steps before from_step: skip unconditionally
        if from_step and step_idx < from_idx:
            logger.info(
                "  [%s] %s SKIPPED (before --from-step)", ds_name, step)
            result["steps"][step] = {
                "skipped": True, "reason": "before_from_step"}
            continue

        # Steps at/after from_step (or all steps when no from_step):
        # if resume and no from_step, also skip already-complete steps
        if resume and from_step is None and is_step_complete(step, dataset_path):
            logger.info("  [%s] %s SKIPPED (already complete)", ds_name, step)
            result["steps"][step] = {
                "skipped": True, "reason": "already_complete"}
            continue

        logger.info("  [%s] %s ...", ds_name, step)
        t0 = time.monotonic()

        # Force disk attribute-cache refresh
        _refresh_disk_cache(dataset_path)

        extra_args = STEP_EXTRA_ARGS.get(step, [])
        cmd = f"{opensfm_bin} {step} {dataset_path}"
        if extra_args:
            cmd += " " + " ".join(extra_args)

        try:
            subprocess.run(
                [
                    "conda", "run", "--name", conda_env,
                    "bash", "-c",
                    f"umask 0000 && {cmd}",
                ],
                capture_output=True,
                text=True,
                check=True,
                preexec_fn=_make_oom_preexec(500),
            )
            elapsed = time.monotonic() - t0
            result["steps"][step] = {
                "wall_time": round(elapsed, 2),
                "success": True,
            }
            logger.info("  [%s] %s done (%.1fs)", ds_name, step, elapsed)
        except subprocess.CalledProcessError as e:
            elapsed = time.monotonic() - t0
            result["steps"][step] = {
                "wall_time": round(elapsed, 2),
                "success": False,
                "stderr": e.stderr[-2000:] if e.stderr else "",
            }
            result["success"] = False
            result["failed_step"] = step
            logger.error(
                "  [%s] %s FAILED (%.1fs)\n%s",
                ds_name,
                step,
                elapsed,
                e.stderr[-500:] if e.stderr else "",
            )
            break

    return result


def run_all_datasets(
    opensfm_bin: str,
    run_dir: str,
    config: BenchmarkConfig,
    commit_hash: str,
    conda_env: str,
    resume: bool = False,
    from_step: Optional[str] = None,
    existing_meta: Optional[Dict[str, Any]] = None,
    bootstrap_run_dir: Optional[str] = None,
    dense: bool = False,
    local_staging: bool = False,
    scratch_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the pipeline on all datasets, saving run_meta.json after each one.

    Args:
        bootstrap_run_dir: When from_step is set on a fresh run, copy/symlink
                           outputs of steps before from_step from this directory.
        dense:             If True, also run the dense-reconstruction stages.
        local_staging:     If True, process each dataset on a local scratch disk
                           and mirror the result tree back to ``run_dir`` once,
                           so NAS I/O does not dominate the timings.
        scratch_dir:       Base directory for local staging (default: $TMPDIR).

    Returns the final run metadata dict.
    """
    staging_root: Optional[str] = None
    if local_staging:
        base = scratch_dir or tempfile.gettempdir()
        staging_root = os.path.join(
            base, "opensfm-bench-" + os.path.basename(os.path.normpath(run_dir))
        )
        os.makedirs(staging_root, exist_ok=True)
        logger.info("Local staging enabled — scratch root: %s", staging_root)
    run_meta: Dict[str, Any] = existing_meta or {
        "commit": commit_hash,
        "date": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
        "dense": dense,
        "config": {
            "root": config.root,
            "datasets": config.datasets,
            "output_dir": config.output_dir,
        },
        "datasets": {},
    }
    # Ensure status is reset when re-running
    run_meta["status"] = "in_progress"
    run_meta["dense"] = dense
    save_run_meta(run_meta, run_dir)

    total_t0 = time.monotonic()

    for dataset_name in config.datasets:
        nas_dataset_path = os.path.join(run_dir, dataset_name)
        logger.info("Running pipeline on %s", dataset_name)

        # Process on local disk when staging; otherwise straight on the NAS.
        dataset_path = nas_dataset_path
        local_dir: Optional[str] = None
        if staging_root is not None:
            local_dir = os.path.join(staging_root, dataset_name)
            logger.info("  Staging %s on local disk: %s",
                        dataset_name, local_dir)
            stage_dataset_local(nas_dataset_path, local_dir)
            dataset_path = local_dir

        try:
            if from_step and bootstrap_run_dir:
                src_ds = os.path.join(bootstrap_run_dir, dataset_name)
                if os.path.isdir(src_ds):
                    logger.info("  Bootstrapping %s from %s",
                                dataset_name, src_ds)
                    bootstrap_dataset(
                        src_ds, dataset_path, from_step, dense=dense)
            pipeline_result = run_pipeline(
                opensfm_bin, dataset_path, conda_env,
                resume=resume,
                from_step=from_step,
                dense=dense,
            )
        finally:
            # Always mirror results back (even on failure) so partial outputs
            # and reports are preserved on the NAS, then drop the local copy.
            if local_dir is not None:
                logger.info("  Moving %s results to %s",
                            dataset_name, nas_dataset_path)
                sync_dataset_to_nas(local_dir, nas_dataset_path)
                cleanup_local_dir(local_dir)

        run_meta["datasets"][dataset_name] = pipeline_result
        # Persist after each dataset so a crash is recoverable
        save_run_meta(run_meta, run_dir)

    run_meta["total_wall_time"] = round(time.monotonic() - total_t0, 2)
    run_meta["status"] = "complete"
    save_run_meta(run_meta, run_dir)
    logger.info("Run metadata written to %s/run_meta.json", run_dir)

    if staging_root is not None:
        shutil.rmtree(staging_root, ignore_errors=True)

    return run_meta
