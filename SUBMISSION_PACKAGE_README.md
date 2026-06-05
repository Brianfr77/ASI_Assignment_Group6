# Submission Package Notes

This folder is the cleaned GitHub-ready submission package for the Advanced Statistical Inference assignment on **Auto-Encoding Variational Bayes**.

It is intended to contain all resources needed by the instructor or a teammate to inspect the implementation, reproduce the experiments, and verify the reported results.

## Contents

- `README.md`: main project README and run commands.
- `configs/`: fixed YAML configurations for AEVB, Wake-Sleep, marginal likelihood, and smoke tests.
- `scripts/`: CLI entry points for running the experiments and collecting figures/tables.
- `src/aevb_v3/`: implementation of data loading, VAE, ELBO, Wake-Sleep, HMC, MCEM, plotting, and result collection.
- `notebooks/AEVB_MNIST_Reproduction_v3_colab.ipynb`: clean Colab notebook.
- `notebooks/AEVB_MNIST_Reproduction_v3_colab_executed.ipynb`: Colab notebook copy with saved output evidence.
- `results/metrics/`: CSV training curves and marginal-likelihood estimates.
- `results/tables/`: report-ready summary tables.
- `results/figures/`: report-ready figures.
- `results/manifests/`: selected learning rate and experiment manifest.
- `results/checkpoints/`: trained model checkpoints, intended for Git LFS.

Raw MNIST data is intentionally not included. The code downloads MNIST through `torchvision` and regenerates the static Bernoulli dataset using seed `2026`.

## Git LFS

The checkpoint files under `results/checkpoints/*.pt` are included for complete reproducibility and should be committed through Git LFS. The repository already includes `.gitattributes` rules for `*.pt` and `*.pth`.

After cloning the GitHub repository, run:

```bash
git lfs pull
```

to download the checkpoint contents.

## Reproducing The Experiments

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the quick smoke test:

```bash
python scripts/run_smoke_tests.py --config configs/quick_debug.yaml
```

Run the full suite:

```bash
python scripts/run_v3_suite.py --backup-dir /content/drive/MyDrive/AEVB_MNIST_Reproduction_v3_backups
```

Manual equivalent:

```bash
python scripts/run_pilot.py --config configs/aevb_binarized_practical.yaml
python scripts/run_aevb_sweep.py --config configs/aevb_binarized_practical.yaml --resume
python scripts/run_wake_sleep.py --config configs/wake_sleep_binarized_practical.yaml --resume
python scripts/run_marginal_likelihood.py --config configs/marginal_likelihood_z3_h100_practical.yaml --resume
python scripts/make_figures.py --config configs/aevb_binarized_practical.yaml
python scripts/collect_results.py --results-dir results
```

## Most Important Result Files

Use these files when checking the report claims:

```text
results/tables/final_test_summary_by_method_z.csv
results/tables/report_aevb_latent_sweep_clean.csv
results/tables/report_aevb_vs_wake_sleep_seed0_clean.csv
results/metrics/marginal_likelihood_summary.csv
results/figures/figure2_style_lower_bound_comparison_clean.png
results/figures/figure3_style_marginal_likelihood_comparison_clean.png
results/figures/aevb_binarized_z2_latent_manifold.png
```

Do not use these older automatically generated figure files for the final report:

```text
results/figures/figure2_style_lower_bound_comparison.png
results/figures/figure3_style_marginal_likelihood_comparison.png
```

The clean versions fix presentation issues without changing the underlying experiment results.

## Files Intentionally Excluded

The cleaned package excludes:

- zip archives;
- `__pycache__/` and `*.pyc`;
- `*.egg-info/`;
- duplicate `results/src`, `results/scripts`, and `results/configs` directories from the Colab zip;
- raw MNIST data.
