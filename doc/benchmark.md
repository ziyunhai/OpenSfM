# Benchmarking

The `benchmark/` module runs the OpenSfM pipeline on a fixed set of datasets **at a specific git commit**, collects per-step timings and quality statistics, and produces an HTML report that **compares one commit against another**. It's the tool to answer "did my change make things faster / more accurate / break anything?".

## How it works

Each benchmark run is fully isolated from your working tree:

1. **Resolve the commit** you ask for (hash, branch, tag, or `HEAD`).
2. **Check it out in a throwaway git worktree** under `.benchmark-worktrees/<hash>/` — your current checkout is never touched.
3. **Build it in a dedicated conda env** (`opensfm-bench-<hash>`) created from the commit's own lock file (`conda-<os>-<arch>.lock`, falling back to `conda.yml`), so the binary matches that commit exactly.
4. **Set up each dataset** as a lightweight directory (an `image_list.txt` of absolute paths to the source images, ancillary files copied, and the chosen [workflow config](using.md#workflow-presets-configs) installed with `processes` set to the machine's CPU count).
5. **Run the pipeline** step by step, timing each one.
6. **Generate `comparison.html`**, diffing this run's metrics against a reference run.
7. **Tear down** the worktree and conda env.

Results land in `benchmark_runs/<short-hash>_<timestamp>/`. Both `.benchmark-worktrees/` and `benchmark_runs/` are git-ignored.

> The orchestrator also hardens itself against the Linux OOM killer: it lowers its own OOM score and raises the pipeline subprocesses', so under memory pressure the kernel kills a *pipeline step* (recoverable with `--resume`) rather than the whole benchmark.

## Prerequisites

- A **git checkout** of the repo (worktrees require it — a source tarball won't work) and `conda` on `PATH`.
- A **datasets root**: a directory with one subfolder per dataset, each containing an `images/` folder. Optional per-dataset extras are picked up automatically: `gcp_list.txt`, `ground_control_points.json`, `camera_models_overrides.json`, `exif_overrides.json`, and a `masks/` folder.

## The config file

A small JSON file describes what to run. See [`benchmark/benchmark_example.json`](../benchmark/benchmark_example.json):

```json
{
    "root": "./data",
    "datasets": {
        "berlin": "terrestrial",
        "lund": "terrestrial"
    },
    "output_dir": "./benchmark_runs"
}
```

| Field | Meaning |
| ----- | ------- |
| `root` | Directory containing the dataset subfolders. Must exist. |
| `datasets` | Map of `dataset_name → workflow config`. The config name resolves to [`configs/<name>.yaml`](using.md#workflow-presets-configs) (`aerial`, `terrestrial`, `object`). |
| `output_dir` | Where run directories are written (default `./benchmark_runs`). |

The config is validated up front: the root, every dataset folder, every `images/` subfolder, and every referenced `configs/<name>.yaml` must exist.

## Running a benchmark

The module is invoked with `python -m benchmark.run`. The simplest run benchmarks a commit:

```bash
python -m benchmark.run --config benchmark/benchmark_example.json --commit HEAD
```

Add `--dense` to also run the dense stages so their timings appear in the report:

```bash
python -m benchmark.run --config benchmark/benchmark_example.json --commit HEAD --dense
```

Useful flags:

| Flag | Purpose |
| ---- | ------- |
| `--commit <ref>` | Commit/branch/tag to benchmark (starts a new run). Mutually exclusive with `--resume`. |
| `--resume <run_dir>` | Resume or re-run an existing run directory. |
| `--dense` | Also run `undistort` + the dense stages before `compute_statistics`. |
| `--from-step <step>` | Start at this step (see [FAQ](#faq--troubleshooting)). |
| `--reference <run_dir \| commit-prefix>` | What to compare against (default: most recent other run). |
| `--bootstrap <run_dir>` | With `--commit --from-step`, reuse earlier-step outputs from this run. |
| `--local-staging` / `--scratch-dir <dir>` | Process on a local disk and mirror back (for NAS run dirs). |
| `--report-only` | Regenerate the HTML only (requires `--resume`). |
| `--output-dir <dir>` | Override `output_dir` (ignored with `--resume`). |
| `-v` / `--verbose` | Debug logging. |

## Comparing results between commits

The typical A/B workflow is two runs — a baseline and your change:

```bash
# 1. Baseline (e.g. master)
python -m benchmark.run --config cfg.json --commit master

# 2. Your feature branch, compared explicitly against the baseline
python -m benchmark.run --config cfg.json --commit my-feature --reference master
```

`--reference` accepts either a **run directory** or a **commit-hash prefix** (it searches `output_dir` for a matching run). If you omit it, the most recent other run is used automatically — so simply running two commits back-to-back already produces a comparison.

Open `benchmark_runs/<run>/comparison.html`. Each dataset shows a `reference` row and a `current` row across several metric groups, with cells coloured **green (better)** / **red (worse)**:

- **Reconstruction Summary** — components, reconstructed shots/points, observations
- **Reprojection Errors** — normalized / pixel / angular
- **Track Statistics** — average track length (and length > 2)
- **Feature Statistics** — detected & reconstructed features (mean/median)
- **GPS Errors** / **GCP Errors** — average error
- **Processing Times** — per-step wall times (plus the dense stages when either run used `--dense`), and total time

A dataset that failed shows a **FAILED** banner with the step it died on; the dataset header links to that dataset's `stats/report.pdf`. All numbers are read from each dataset's `stats/stats.json` (produced by `compute_statistics`); see [quality report](quality_report.md) for what they mean.

## Output layout

```
benchmark_runs/
└── a1b2c3d4_20260628_124501/        # <short-hash>_<UTC-timestamp>
    ├── run_meta.json                # commit, status, per-dataset per-step timings + success/stderr
    ├── comparison.html              # the A/B report
    ├── berlin/                      # a lightweight dataset dir (image_list.txt, config.yaml,
    │   ├── ...                      #   exif/, features/, reconstruction.json, reports/, stats/ …)
    │   └── stats/{stats.json,report.pdf}
    └── lund/
        └── ...
```

`run_meta.json` is written **after every dataset** (and updated as steps complete), which is what makes a run crash-safe and resumable.

## Pipeline steps

SfM (always): `extract_metadata` → `detect_features` → `match_features` → `create_tracks` → `reconstruct` → `compute_statistics` → `export_report`.

Dense (with `--dense`, spliced in before `compute_statistics`): `undistort` → `dense_clustering` → `dense_equalize` → `compute_depthmaps` → `fuse_depthmaps` → `dense_merging` (run with `--georeferenced`). See [dense reconstruction](dense.md).

## FAQ / troubleshooting

### The benchmark crashed or was interrupted — how do I resume?

Point `--resume` at the run directory. It reads the commit from `run_meta.json` (or infers it from the folder name), then skips every step whose output already exists and continues where it stopped:

```bash
python -m benchmark.run --config cfg.json --resume benchmark_runs/a1b2c3d4_20260628_124501
```

### How do I re-run from a specific step?

Use `--from-step` with `--resume`. Steps *before* it are skipped unconditionally; that step and everything after it are re-run **even if their outputs already exist** (use this after changing, say, reconstruction code):

```bash
python -m benchmark.run --config cfg.json --resume <run_dir> --from-step reconstruct
```

Valid steps: `extract_metadata`, `detect_features`, `match_features`, `create_tracks`, `reconstruct`, `undistort`, `dense_clustering`, `dense_equalize`, `compute_depthmaps`, `fuse_depthmaps`, `dense_merging`, `compute_statistics`, `export_report`.

### How do I regenerate only the metrics HTML (no rebuild, no pipeline)?

`--report-only` (requires `--resume`) skips the worktree, build and pipeline entirely and just rebuilds `comparison.html` from existing results. Handy after the pipeline finished but you want a different comparison:

```bash
python -m benchmark.run --config cfg.json --resume <run_dir> --report-only --reference <other_run_or_prefix>
```

### How do I run only SfM, or SfM + dense?

SfM-only is the default — just omit `--dense`. Add `--dense` for the dense stages too. Selecting a dense `--from-step` (e.g. `compute_depthmaps`) implies `--dense` automatically. On `--resume`, the dense setting is inherited from the resumed run's `run_meta.json` unless you pass `--dense` again.

### How do I start a fresh run but reuse expensive early steps (features/matches)?

Combine `--commit` with `--from-step`: outputs of the steps before `--from-step` are symlinked/copied from an existing run. Without `--bootstrap`, the most recent **complete** run for the same commit is auto-detected:

```bash
# reuse everything up to reconstruct from a prior run of this commit
python -m benchmark.run --config cfg.json --commit my-feature --from-step reconstruct
# ...or name the source explicitly
python -m benchmark.run --config cfg.json --commit my-feature --from-step reconstruct --bootstrap <run_dir>
```

### One dataset failed but the others were fine.

Each dataset is independent: a failure stops *that* dataset at its failing step (recorded in `run_meta.json` as `failed_step` plus the tail of stderr) and the report shows a FAILED banner, but the remaining datasets still run. Re-run with `--resume` to retry just the unfinished steps, or `--resume --from-step <step>` to force a redo.

### My timings look I/O-bound (run directory on a NAS).

Use `--local-staging` (optionally with `--scratch-dir <fast-local-dir>`): each dataset is processed on local scratch and mirrored back to the run directory once, so network I/O doesn't pollute the timed steps.

### The build fails in the worktree.

The conda env is created from the *benchmarked commit's* `conda-<os>-<arch>.lock` (or `conda.yml`). Make sure that commit actually contains a valid lock file / `conda.yml`. Stale worktrees in `.benchmark-worktrees/` are force-removed and recreated each run; you can safely delete that directory between runs.

## See also

- [Workflow presets (`configs/`)](using.md#workflow-presets-configs) — the per-dataset configs the benchmark installs.
- [Quality report](quality_report.md) — the metrics surfaced in the comparison.
- [Pipeline commands](using.md) — the steps being benchmarked.
