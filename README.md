# AEVB MNIST Reproduction v3

This repository contains a paper-aligned practical reproduction of **Auto-Encoding Variational Bayes** on MNIST.

The implementation keeps the main experimental structure of Kingma and Welling's paper while using a focused run plan suitable for Colab A100:

- statically binarized MNIST with a Bernoulli decoder;
- one-hidden-layer MLP encoder/decoder with `tanh`;
- Adagrad with a pilot-selected global learning rate from `{0.01, 0.02, 0.1}`;
- AEVB lower-bound sweeps for `z_dim=[3,5,10,20,200]`;
- a `z_dim=2` AEVB run for the latent manifold figure;
- Wake-Sleep lower-bound baselines using the same architecture;
- a low-dimensional `z=3`, `hidden=100` marginal likelihood experiment with HMC and MCEM.

## Setup

In Colab:

```bash
pip install -r requirements.txt
pip install -e .
```

Run commands from the repository root.

## Notebooks

- `notebooks/AEVB_MNIST_Reproduction_v3_colab.ipynb`: clean Colab notebook for rerunning the full suite.
- `notebooks/AEVB_MNIST_Reproduction_v3_colab_executed.ipynb`: sanitized Colab notebook with saved output evidence from the completed run.

## Recommended Run

The suite runner executes the smoke test, pilot, AEVB sweep, Wake-Sleep, marginal likelihood, figures, result tables, and a backup after every stage:

```bash
python scripts/run_v3_suite.py --backup-dir /content/drive/MyDrive/AEVB_MNIST_Reproduction_v3_backups
```

Equivalent manual commands:

```bash
python scripts/run_smoke_tests.py --config configs/quick_debug.yaml
python scripts/run_pilot.py --config configs/aevb_binarized_practical.yaml
python scripts/run_aevb_sweep.py --config configs/aevb_binarized_practical.yaml --resume
python scripts/run_wake_sleep.py --config configs/wake_sleep_binarized_practical.yaml --resume
python scripts/run_marginal_likelihood.py --config configs/marginal_likelihood_z3_h100_practical.yaml --resume
python scripts/make_figures.py --config configs/aevb_binarized_practical.yaml
python scripts/collect_results.py --results-dir results
```

Create an emergency backup during a long Colab run with:

```bash
python scripts/backup_results.py --results-dir results --output-dir /content/drive/MyDrive/AEVB_MNIST_Reproduction_v3_backups
```

## Outputs

- `results/metrics/*.csv`: training curves, pilot selection, marginal likelihood estimates.
- `results/tables/*.csv`: report-ready summaries.
- `results/figures/*.png`: lower-bound curves, marginal likelihood curves, samples, and latent manifolds.
- `results/checkpoints/*.pt`: model checkpoints intended for Git LFS or GitHub Releases.
- `results/manifests/*.json`: package versions, file inventory, validation summary, and selected learning rate.

Raw MNIST is not committed. The code downloads MNIST through torchvision and regenerates the static Bernoulli version with seed `2026`.

## Reliability

- Metrics are flushed to CSV after every evaluation.
- AEVB and Wake-Sleep save intermediate checkpoints every 50 epochs.
- Marginal-likelihood runs save checkpoints at epochs `[1, 5, 10, 20, 30, 50]`.
- Checkpoints include model state, optimizer state, epoch, samples seen, configuration, and package versions.
- Rerunning with `--resume` continues from the latest intermediate checkpoint when the final checkpoint is not present.

## Main Implementation Choices

- The decoder returns logits and the Bernoulli reconstruction term is computed with `binary_cross_entropy_with_logits`.
- The MAP parameter prior `p(theta)=N(0,I)` is implemented as an explicit `0.5 * sum(theta^2) / N_train` penalty on the per-datapoint objective.
- Wake-Sleep uses the same encoder and decoder classes as AEVB. The lower bound is evaluated by the same evaluator for both methods.
- The paper does not fully specify the density estimator used inside the marginal likelihood estimator. This code uses a full-covariance Gaussian fitted separately for each datapoint's HMC posterior samples.

## AI Use Disclosure Draft

AI tools were used to help plan the reproduction, structure the codebase, draft implementation code, and debug PyTorch/Colab issues. All experiments, code behavior, generated results, and report claims must be inspected and understood by the author before submission.
