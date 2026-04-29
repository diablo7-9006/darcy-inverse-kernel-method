# Robustness Study for a Kernel-Based Darcy Flow Inverse Problem

This repository contains the code and generated results for an ECEN 744 Scientific Machine Learning final project. The project studies a simplified kernel-based approach for an inverse Darcy flow problem, where the goal is to recover both the solution field \(u(x)\) and the log-permeability field \(a(x)\) from limited noisy pointwise observations of \(u\).

The implementation is not a full reproduction of the Gaussian-process method in the reference paper. Instead, it keeps the main idea of using the governing PDE to guide the reconstruction and uses a simpler numerical setup that is easier to test and interpret.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── darcy_inverse_problem.py
├── results/
│   ├── figures/
│   │   ├── darcy_01_collocation_points.png
│   │   ├── darcy_02_truth_a.png
│   │   ├── darcy_03_truth_u.png
│   │   ├── darcy_04_loss_history.png
│   │   ├── darcy_05_recovered_a.png
│   │   ├── darcy_06_recovered_u.png
│   │   ├── darcy_exp_01_observation_count.png
│   │   ├── darcy_exp_02_noise_level.png
│   │   ├── darcy_exp_03_collocation_count.png
│   │   ├── darcy_exp_04_kernel_length_scale.png
│   │   └── darcy_exp_05_regularization.png
│   └── data/
│       ├── darcy_experiment_raw_results.csv
│       └── darcy_experiment_summary_tables.csv
├── scripts/
│   └── run_single_trial.py
└── report/
    └── README.md
```

The `report/` folder is left as a placeholder so the final submitted report can be added later.

## Method Summary

The code first generates synthetic data for a two-dimensional Darcy flow problem on the unit square. The true permeability is prescribed through a smooth function, and the forward Darcy problem is solved numerically to produce the reference solution field. No external dataset is required.

The unknown fields \(u(x)\) and \(a(x)\) are approximated using Gaussian radial basis functions. The solver then fits the coefficient vectors by combining four pieces of information:

1. the Darcy PDE residual at interior collocation points,
2. the zero boundary condition for \(u\),
3. noisy pointwise observations of \(u\), and
4. coefficient regularization for numerical stability.

The nonlinear least-squares problem is solved using a Gauss--Newton iteration with damping and a backtracking line search.

## Experiments

The script runs one baseline reconstruction and five robustness studies:

1. changing the number of observation points,
2. changing the observation noise level,
3. changing the number of interior collocation points,
4. changing the kernel length scale,
5. changing the regularization parameter.

Each robustness setting is repeated over three independent random trials. The plotted errors are relative \(L^2\) errors for the recovered \(a(x)\) and \(u(x)\).

## Main Files

- `src/darcy_inverse_problem.py`: full project code, including data generation, forward solve, inverse solve, plotting, and parameter sweeps.
- `results/figures/`: generated baseline and robustness plots.
- `results/data/darcy_experiment_raw_results.csv`: raw trial-level errors.
- `results/data/darcy_experiment_summary_tables.csv`: mean and standard-deviation summary tables.
- `scripts/run_single_trial.py`: helper script for running one robustness setting manually.

## How to Run

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Run the full project script from the repository root:

```bash
python src/darcy_inverse_problem.py
```

The script writes figures and CSV files to the current working directory. The generated copies used for the report are already included under `results/figures/` and `results/data/`.

If JAX uses too many CPU threads on your machine, run with these environment variables:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python src/darcy_inverse_problem.py
```

## Notes on Reproducibility

The code uses fixed random seeds for the baseline and repeated trials, so the included results should be reproducible up to small differences caused by package versions and hardware. The computations are synthetic and self-contained.

## Authors

- Harshavardhan Ramachandran
- Xu Ruikun
