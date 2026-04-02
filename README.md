# DurationModulatedDynamics

Code to reproduce figures from the manuscript *"Duration-modulated neural population dynamics in humans during BMI control"*.

## Prerequisites

- Python >= 3.10
- Git (to install the `ssm` dependency)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ynffsy/DurationModulatedDynamics.git
cd DurationModulatedDynamics

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install Cython first (required to build ssm)
pip install cython numpy

# 4. Install the package and all dependencies
pip install -e .
```

> **Note on `ssm`:** The [Linderman Lab SSM package](https://github.com/lindermanlab/ssm) is installed
> automatically from its GitHub repository. It requires Cython and a C compiler. On macOS, the Xcode
> Command Line Tools provide one (`xcode-select --install`). On Linux, `gcc` is typically pre-installed.

## Data setup

All scripts read data from a single base directory that you specify. This directory must contain three sub-folders:

```
<BASE_DIR>/
  data/cg/processed/     # processed neural and behavioral data (.csv, .npz)
  results/dynamics_paper/ # intermediate model fits and analysis outputs
  visualizations/dynamics_paper/  # generated figures
```

Tell the code where this lives by **one** of two methods:

**Option A** -- environment variable (recommended for one-off use):

```bash
export DMD_BASE_DIR=/path/to/your/base/directory
```

**Option B** -- `.env` file (recommended for repeated use):

Create a file called `.env` in the repository root:

```
DMD_BASE_DIR=/path/to/your/base/directory
```

This file is already in `.gitignore` and will not be committed.

## Reproducing figures

With the environment configured and data in place:

```bash
python -m scripts.figure_generation_MASTER
```

This script orchestrates all figure generation and statistical analyses. Individual
figure blocks inside the script are toggled by commenting/uncommenting the relevant
sections.

## Running individual experiment scripts

SLDS model fitting scripts can be run as modules from the repo root:

```bash
python -m experiments.same_speed_SLDS
python -m experiments.joint_SLDS
python -m experiments.cross_speed_SLDS_test
```

Model parameters (number of states, iterations, etc.) are configured in
`scripts/config.py`.

## Repository structure

```
DurationModulatedDynamics/
  experiments/      # SLDS model fitting pipelines
  metrics/          # Decoding, inference, entropy, and DSUP metrics
  scripts/          # Master scripts, config, figure generation, statistics
  utils/            # Data loading, processing, decoding, and visualization helpers
  visualizations/   # Paper figure plotting and visual configuration
```

## Configuration

Edit `scripts/config.py` to:

- Select which sessions to analyze (`session_data_names`)
- Set model sweep parameters (`ns_states`, `ns_discrete_states`, `ns_iters`)
- Choose trial filter conditions (`trial_filters`)
- Toggle training paradigms (`train_test_options`)
