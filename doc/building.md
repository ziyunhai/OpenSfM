# Building OpenSfM

## Prerequisites

[Conda](https://docs.conda.io/en/latest/miniconda.html) (or Miniconda) is the only prerequisite on all platforms. The conda environment installs the full toolchain — the C++ compiler (MSVC on Windows, clang/gcc elsewhere), CMake, Ninja, and all library dependencies.

> The build also needs the bundled `pybind11` git submodule. The `--recursive` clone below fetches it; in an existing clone run `git submodule update --init --recursive`.

## Installation

### Linux

```bash
git clone --recursive https://github.com/OpenSfM/OpenSfM
cd OpenSfM
conda create --name opensfm --file conda-linux-64.lock --yes
conda activate opensfm
pip install -e .
```

### macOS (Apple Silicon)

```bash
git clone --recursive https://github.com/OpenSfM/OpenSfM
cd OpenSfM
conda create --name opensfm --file conda-osx-arm64.lock --yes
conda activate opensfm
pip install -e .
```

> Only Apple Silicon (`arm64`) is supported. Intel Macs (`osx-64`) are not supported — there is no lock file for them.

### Windows

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first, then run from the *Anaconda Prompt*:

```bat
git clone --recursive https://github.com/OpenSfM/OpenSfM
cd OpenSfM
conda create --name opensfm --file conda-win-64.lock --yes
conda activate opensfm
pip install -e .
```

> The MSVC C++ compiler ships with the environment (`conda-win-64.lock` pins `vs2022_win-64`), so a separate Visual Studio / Build Tools install is **not** required.

## Running

Activate the environment, then run the desired pipeline step:

**Linux / macOS:**

```bash
conda activate opensfm
./bin/opensfm COMMAND path/to/dataset
```

**Windows:**

```bat
conda activate opensfm
bin\opensfm.bat COMMAND path\to\dataset
```

See [using.md](using.md) for the full list of available commands.

## Viewer

A web-based 3D viewer is included under `viewer/`. See the [quickstart](quickstart.md#viewer) and [viewer/README.md](../viewer/README.md) for how to serve a dataset.

## Building Docker Images

Example Dockerfiles are provided for Ubuntu 20.04 and 24.04:

```bash
docker build -t opensfm.ubuntu24 -f Dockerfile.ubuntu24 .
```
