# AEVB MNIST Reproduction v3: Experiment Handoff Report

This document is a full English handoff report for the teammate who will write the final course paper. It is intentionally more detailed than the final five-page NeurIPS-format submission. The goal is to preserve the complete experimental chain: why the project was set up this way, what was implemented, how the experiments were run, what the results mean, what should be reported, and what should not be overclaimed.

Primary paper: Diederik P. Kingma and Max Welling, "Auto-Encoding Variational Bayes", arXiv:1312.6114, https://arxiv.org/abs/1312.6114.

Primary local project: `C:\Users\wangz\Desktop\ASI\Assignment\AEVB_MNIST_Reproduction_v3`

Primary local result download: `C:\Users\wangz\Desktop\ASI\Assignment\AEVB_MNIST_Reproduction_v3\AEVB_MNIST_Reproduction_v3_results`

Important convention for paths in this document:

- In the current local download, result artifacts are under `AEVB_MNIST_Reproduction_v3_results/results/...`.
- For the final GitHub repository, the same artifacts should be placed under the repository-root `results/...` directory.
- When this report says `results/...`, it refers to the final repository layout. In the current local machine, prepend `AEVB_MNIST_Reproduction_v3_results/`.

## 1. Title and Executive Summary

### Working title

Reproducing Auto-Encoding Variational Bayes on Binarized MNIST

### One-paragraph summary

This project reproduces the core MNIST experiments from Kingma and Welling's Auto-Encoding Variational Bayes paper using a practical, paper-aligned PyTorch implementation. The final experimental line is v3. It uses statically binarized MNIST, a Bernoulli decoder, a diagonal Gaussian encoder, one-hidden-layer MLP networks with `tanh` activations, Adagrad optimization, and an explicit MAP-style L2 parameter prior. The main experiments are: an AEVB latent-dimensionality sweep, a Wake-Sleep baseline evaluated with the same ELBO evaluator, a two-dimensional latent manifold visualization, random sample grids, and a reduced-scale marginal likelihood comparison using AEVB, Wake-Sleep, and MCEM-HMC. The main quantitative result is that AEVB improves strongly as latent dimensionality increases and outperforms Wake-Sleep for every matched latent dimension in the reproduction. The marginal likelihood experiment is useful but must be presented as reduced-scale and not as an exact reproduction of the paper's full Figure 3.

### Final recommendation

Use only the v3 results as the final experimental evidence. Earlier v1 and v2 experiments should not be included in the main report. They were useful for debugging and planning, but including them would make the final project narrative more complex and less paper-aligned.

### Report-ready claims

The following claims are safe to use in the final paper:

1. The v3 implementation reproduces the main qualitative behavior of AEVB on MNIST: larger latent spaces improve the variational lower bound, with diminishing returns at high latent dimensionality.
2. In the reproduced lower-bound comparison, AEVB gives better test negative ELBO than Wake-Sleep for every matched latent dimension.
3. The best AEVB lower-bound result in the main sweep is obtained at `z_dim=200`, with mean test negative ELBO `99.23`.
4. The largest AEVB gain over Wake-Sleep in the matched seed-0 comparison is at `z_dim=200`, where Wake-Sleep is worse by about `8.76` NELBO.
5. The reduced-scale marginal likelihood experiment shows MCEM-HMC with the best final estimated marginal log-likelihood under the implemented estimator, while AEVB and Wake-Sleep become close by the end of the 50-epoch small-data run.

### Claims to avoid

Do not claim that this is a bit-for-bit or fully exact reproduction of all original paper experiments. In particular:

- The v3 marginal likelihood experiment uses `N_train=1000` only, not the full original comparison over both `N_train=1000` and `N_train=50000`.
- The HMC-based marginal likelihood estimator uses an explicit full-covariance Gaussian density estimator because the original paper does not fully specify that implementation detail.
- The main training budget is practical (`300` epochs), not necessarily the same as the original paper's long-run training scale.
- The qualitative sample quality should be described as recognizable but blurry/noisy, not as state-of-the-art generation.

## 2. Assignment Objective and Reproduction Scope

### Course objective

The course assignment asks students to reproduce part of a research paper in probabilistic machine learning, submit code and results, and write a short report explaining the method, experiments, and findings. The selected paper is:

Kingma, D. P. and Welling, M. "Auto-Encoding Variational Bayes." arXiv:1312.6114.

The paper is a natural fit for the assignment because it connects variational inference, latent-variable models, stochastic optimization, and neural amortized inference. It is also experimentally reproducible on MNIST without requiring very large datasets.

### Final reproduction scope

The final reported experiment should use the v3 project only:

`C:\Users\wangz\Desktop\ASI\Assignment\AEVB_MNIST_Reproduction_v3`

The final result set is:

`C:\Users\wangz\Desktop\ASI\Assignment\AEVB_MNIST_Reproduction_v3\AEVB_MNIST_Reproduction_v3_results`

The v3 scope is:

- Binarized MNIST reproduction.
- AEVB/VAE lower-bound sweep over latent dimensions.
- Wake-Sleep baseline using the same architecture and evaluator.
- Two-dimensional latent manifold visualization.
- Random sample grids.
- Reduced-scale marginal likelihood comparison using AEVB, Wake-Sleep, and MCEM-HMC.

### Out of scope for the final paper

The following should not be part of the final reported experiment:

- v1 grayscale MNIST experiments.
- v2 long-run intermediate experiments.
- Modern diagnostics such as active units, IWAE estimates, posterior-prior mismatch plots, or interpolation diagnostics.
- Claims about exact reproduction of the original paper's full marginal likelihood estimator.

The final paper can mention, at most, that preliminary grayscale experiments were used to debug the implementation, but all reported results use the v3 paper-aligned binarized MNIST setup.

## 3. Paper Background

### Problem addressed by the paper

The paper studies learning and inference in directed latent-variable models. In these models, each observation `x` is assumed to be generated from an unobserved latent variable `z`. The generative process is:

```text
z ~ p(z)
x ~ p_theta(x | z)
```

The marginal likelihood is:

```text
p_theta(x) = integral p_theta(x | z) p(z) dz
```

For neural decoders, this integral is generally intractable. The true posterior

```text
p_theta(z | x) = p_theta(x | z)p(z) / p_theta(x)
```

is also intractable. The paper's key contribution is to learn an approximate posterior using a neural recognition model and optimize a stochastic variational lower bound through the reparameterization trick.

### Main idea of AEVB

Auto-Encoding Variational Bayes combines:

- A generative model, or decoder, `p_theta(x | z)`.
- A recognition model, or encoder, `q_phi(z | x)`.
- A variational lower bound on `log p_theta(x)`.
- A differentiable sampling transformation that allows stochastic gradient optimization.

This framework is now commonly called the Variational Autoencoder (VAE).

### Why MNIST is appropriate

MNIST is a standard binary or grayscale image dataset. In the original paper, MNIST is used to evaluate whether AEVB can train deep latent-variable models with continuous latent variables and a neural decoder. In this reproduction, MNIST is used with static Bernoulli binarization, which is aligned with the Bernoulli decoder and makes the likelihood interpretation clean.

## 4. Theoretical Foundations

### Latent-variable model

For a single datapoint `x`, the model assumes:

```math
p_\theta(x,z) = p_\theta(x | z)p(z)
```

The prior is a standard Gaussian:

```math
p(z) = N(0, I)
```

For MNIST, `x` is a 784-dimensional binary vector after binarization. The decoder outputs Bernoulli logits:

```math
f_\theta(z) \in R^{784}
```

and the Bernoulli probabilities are:

```math
\pi_\theta(z) = sigmoid(f_\theta(z)).
```

The likelihood is:

```math
p_\theta(x | z) = \prod_{d=1}^{784} Bernoulli(x_d; \pi_{\theta,d}(z)).
```

Equivalently, the log likelihood is:

```math
\log p_\theta(x | z)
= \sum_{d=1}^{784}
  x_d \log \pi_{\theta,d}(z)
  + (1 - x_d)\log(1 - \pi_{\theta,d}(z)).
```

In code, this is implemented with `binary_cross_entropy_with_logits`, which is numerically more stable than applying `sigmoid` and then taking logs manually.

### Approximate posterior

The encoder approximates the intractable posterior `p_theta(z | x)` with a diagonal Gaussian:

```math
q_\phi(z | x)
= N(z; \mu_\phi(x), diag(\sigma_\phi^2(x))).
```

The encoder network maps `x` to two vectors:

```math
\mu_\phi(x), \log \sigma_\phi^2(x) \in R^J
```

where `J` is the latent dimension.

### Evidence lower bound

The marginal log likelihood can be decomposed as:

```math
\log p_\theta(x)
= L(\theta,\phi;x)
+ D_{KL}(q_\phi(z | x) || p_\theta(z | x)).
```

Since the KL divergence is non-negative, the first term is a lower bound:

```math
L(\theta,\phi;x)
= E_{q_\phi(z | x)}[\log p_\theta(x | z)]
- D_{KL}(q_\phi(z | x) || p(z)).
```

The training objective is to maximize this ELBO. In implementation, the optimizer minimizes the negative ELBO:

```math
-L(\theta,\phi;x)
= BCE(x, decoder(z))
+ D_{KL}(q_\phi(z | x) || p(z)).
```

The reported metrics use:

```text
negative_elbo = bce + kl
lower_bound = -negative_elbo
```

Important implementation detail: the MAP L2 parameter prior is included in the optimization objective, but the reported `negative_elbo`, `bce`, and `kl` metrics are the data likelihood and latent KL terms only. This matches the code in `src/aevb_v3/losses.py`.

### Reparameterization trick

Directly sampling

```math
z ~ q_\phi(z | x)
```

would make gradients with respect to `phi` difficult. The reparameterization trick writes the sample as:

```math
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon,
\quad \epsilon ~ N(0, I).
```

Now the randomness is isolated in `epsilon`, and the mapping from `phi` to `z` is differentiable. This is the core technical device that makes AEVB efficient with stochastic gradient methods.

### Closed-form Gaussian KL

For a diagonal Gaussian approximate posterior and a standard Gaussian prior:

```math
q_\phi(z | x) = N(\mu, diag(\sigma^2)),
\quad p(z)=N(0,I),
```

the KL divergence has the closed form:

```math
D_{KL}(q_\phi(z | x) || p(z))
= -0.5 \sum_j (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2).
```

The implementation uses `logvar = log(sigma^2)` and computes:

```python
kl = -0.5 * sum(1 + logvar - mu.pow(2) - exp(logvar))
```

This term is always non-negative up to numerical precision.

### MAP parameter prior

The original paper uses a small weight decay/MAP-style regularization. In v3, this is implemented explicitly as a Gaussian prior over trainable parameters:

```math
p(\theta) = N(0, I).
```

The corresponding penalty is:

```math
\frac{1}{2N}\sum_i \theta_i^2.
```

In code, this is:

```python
0.5 * scale * sum(param.pow(2).sum() for param in parameters)
```

with `scale = 1 / N_train`.

This choice is clearer than relying on optimizer-level `weight_decay`, because it makes the parameter prior explicit and documents exactly how it enters the objective.

### Wake-Sleep baseline

Wake-Sleep is a classical method for training models with latent variables and a recognition model. It alternates between two phases:

1. Wake phase:
   - Use the encoder `q_phi(z | x)` to infer `z` for real data `x`.
   - Update the decoder/generative parameters to improve `p_theta(x | z)`.

2. Sleep phase:
   - Sample `z ~ p(z)`.
   - Generate synthetic `x ~ p_theta(x | z)`.
   - Update the encoder to improve `q_phi(z | x)` on synthetic pairs.

In v3, Wake-Sleep uses the same encoder and decoder architecture as AEVB. It is evaluated using the same ELBO evaluator, which allows a direct comparison of final lower bounds.

### MCEM-HMC and marginal likelihood estimator

The reduced-scale marginal likelihood experiment uses a low-dimensional latent space (`z_dim=3`) and hidden dimension `100`. This is because HMC over latent variables is practical only for low-dimensional latent spaces.

The MCEM idea is:

1. For each minibatch, sample latent variables from an approximate posterior using HMC.
2. Update decoder/generative parameters using the sampled latent variables.
3. Repeat for a fixed number of epochs.

The marginal likelihood estimate follows the paper's broad low-dimensional HMC idea, but the exact density-estimation detail is made explicit in this reproduction. For each datapoint, HMC posterior samples are drawn, a full-covariance Gaussian density estimator is fit to those posterior samples, and the marginal likelihood is estimated using the relationship:

```math
p_\theta(x) = \frac{p_\theta(x,z)}{p_\theta(z | x)}.
```

Since the true posterior density is unavailable, the fitted Gaussian posterior density is used as the denominator estimate. This implementation choice must be disclosed in the formal report.

## 5. Implementation Overview

### Repository structure

The v3 repository is organized as a GitHub-ready project:

```text
AEVB_MNIST_Reproduction_v3/
  configs/
  notebooks/
  scripts/
  src/aevb_v3/
  tests/
  results/
  README.md
  requirements.txt
  pyproject.toml
```

The downloaded result package is:

```text
AEVB_MNIST_Reproduction_v3_results/
  configs/
  scripts/
  src/
  results/
    checkpoints/
    figures/
    manifests/
    metrics/
    tables/
```

For final GitHub submission, results should be placed in the root `results/` directory. Checkpoints are large and should be handled with Git LFS or a GitHub Release.

### Core modules

The implementation is modular rather than notebook-only:

- `src/aevb_v3/data.py`: MNIST loading, static binarization, shape/range checks, DataLoader creation.
- `src/aevb_v3/models.py`: VAE encoder/decoder model and distribution helper functions.
- `src/aevb_v3/losses.py`: ELBO components, BCE, KL, MAP L2 penalty.
- `src/aevb_v3/training.py`: AEVB training, evaluation, learning-rate pilot.
- `src/aevb_v3/wake_sleep.py`: Wake-Sleep training.
- `src/aevb_v3/hmc.py`: HMC sampling utilities.
- `src/aevb_v3/mcem.py`: MCEM training.
- `src/aevb_v3/marginal_likelihood.py`: reduced HMC-based marginal likelihood estimation.
- `src/aevb_v3/plots.py`: lower-bound plots, marginal likelihood plots, sample grids, latent manifold.
- `src/aevb_v3/collect.py`: result collection, validation, summary tables, manifest writing.

### Main CLI scripts

The suite runner executes the whole pipeline:

```bash
python scripts/run_v3_suite.py --backup-dir /content/drive/MyDrive/AEVB_MNIST_Reproduction_v3_backups
```

Manual commands are:

```bash
python scripts/run_smoke_tests.py --config configs/quick_debug.yaml
python scripts/run_pilot.py --config configs/aevb_binarized_practical.yaml
python scripts/run_aevb_sweep.py --config configs/aevb_binarized_practical.yaml --resume
python scripts/run_wake_sleep.py --config configs/wake_sleep_binarized_practical.yaml --resume
python scripts/run_marginal_likelihood.py --config configs/marginal_likelihood_z3_h100_practical.yaml --resume
python scripts/make_figures.py --config configs/aevb_binarized_practical.yaml
python scripts/collect_results.py --results-dir results
```

### Colab execution

The v3 experiment was run on Colab using an A100 GPU. The runtime environment reported:

- GPU: NVIDIA A100-SXM4-80GB.
- PyTorch: `2.11.0+cu128`.
- CUDA available.

The run generated metrics, checkpoints, figures, tables, and manifest files. The final result zip was downloaded locally.

### Reliability design

The v3 code was designed to survive Colab disconnections:

- CSV metrics are flushed after every evaluation.
- Intermediate checkpoints are saved every `50` epochs for AEVB and Wake-Sleep.
- Marginal likelihood checkpoints are saved at epochs `[1, 5, 10, 20, 30, 50]`.
- Resume mode is enabled by default.
- A backup script can zip `results/` to Google Drive.

This matters because the experiments are long enough that relying only on final in-memory state would be risky.

## 6. Experimental Design

### Dataset

Dataset: MNIST.

Dataset sizes:

- Training set: `60000`.
- Test set: `10000`.
- Image shape: `1 x 28 x 28`.
- Flattened input dimension: `784`.

The code verifies shape and pixel range when loading data.

### Preprocessing

The main v3 experiments use static Bernoulli binarization:

```text
binarization = static_bernoulli
binarization_seed = 2026
```

Raw grayscale MNIST images are scaled to `[0, 1]`. Each pixel is then sampled once as:

```math
x_d^{binary} ~ Bernoulli(x_d^{gray}).
```

The binary train and test tensors are cached. The train and test random generators use deterministic seeds, with a split offset for the test set. This makes the dataset reproducible.

This differs from the earlier v1 grayscale experiment, which is not used in the final report.

### Main AEVB architecture

Main AEVB model:

```text
Encoder:
  784 -> 500 -> (mu, logvar)

Decoder:
  z_dim -> 500 -> 784 logits
```

Architecture settings:

- Hidden dimension: `500`.
- Activation: `tanh`.
- Encoder distribution: diagonal Gaussian.
- Decoder distribution: Bernoulli with logits.
- Parameter initialization: `N(0, 0.01)`.

The latent dimensions in the main sweep are:

```text
z_dim = [3, 5, 10, 20, 200]
```

The qualitative manifold run uses:

```text
z_dim = 2
```

### Optimization settings

Main optimization settings:

```text
optimizer = Adagrad
batch_size = 100
epochs = 300
eval_every = 5
checkpoint_every = 50
eval_samples = 1
map_l2_scale = 1 / N_train
```

Learning-rate pilot:

```text
lr_candidates = [0.01, 0.02, 0.1]
pilot_z_dim = 10
pilot_seed = 0
pilot_epochs = 5
```

Pilot results from `results/metrics/learning_rate_pilot_summary.csv`:

| Learning rate | Pilot train negative ELBO |
|---:|---:|
| 0.01 | 141.03 |
| 0.02 | 149.92 |
| 0.10 | 11485.43 |

Selected learning rate:

```text
lr = 0.01
```

The `0.10` setting diverged or became numerically poor in the pilot, so it should not be used in the final run.

### Main AEVB sweep

Main AEVB configuration:

```text
z_dim = [3, 5, 10, 20, 200]
seeds = [0, 1, 2]
epochs = 300
hidden_dim = 500
batch_size = 100
lr = 0.01
```

The run also includes:

```text
z_dim = 2
seed = 0
```

for the latent manifold visualization. The `z=2` run should not be treated as part of the main latent-dimensionality sweep because it was included mainly for qualitative visualization.

### Wake-Sleep baseline

Wake-Sleep configuration:

```text
z_dim = [3, 5, 10, 20, 200]
seed = [0]
epochs = 300
hidden_dim = 500
batch_size = 100
lr = 0.01
```

Wake-Sleep uses the same architecture and the same test ELBO evaluator as AEVB. The main limitation is that only seed `0` is used for Wake-Sleep, whereas AEVB uses three seeds.

### Marginal likelihood and MCEM-HMC experiment

Marginal likelihood configuration:

```text
z_dim = 3
hidden_dim = 100
N_train = 1000
seed = 0
epochs = 50
batch_size = 100
lr = 0.01
```

Methods:

- AEVB.
- Wake-Sleep.
- MCEM-HMC.

HMC and evaluation settings:

```text
MCEM posterior_steps = 5
MCEM leapfrog_steps = 5
MCEM weight_updates_per_sample = 3
Marginal likelihood max_points = 200
Marginal likelihood posterior_samples = 30
Marginal likelihood burn_in = 30
Marginal likelihood leapfrog_steps = 4
Target acceptance = 0.90
Initial step size = 0.05
```

Checkpoint epochs:

```text
[1, 5, 10, 20, 30, 50]
```

This experiment should be described as reduced-scale and paper-aligned, not as the full original Figure 3 reproduction.

## 7. Results

### Artifact summary

The local downloaded result package contains:

- `31` CSV files.
- `150` checkpoint files.
- `10` PNG figures after postprocessing.
- `4` summary/report tables.
- Experiment manifests and selected learning-rate JSON.

Important report-ready artifacts:

- `results/tables/final_test_summary_by_method_z.csv`
- `results/tables/report_aevb_latent_sweep_clean.csv`
- `results/tables/report_aevb_vs_wake_sleep_seed0_clean.csv`
- `results/metrics/marginal_likelihood_summary.csv`
- `results/figures/figure2_style_lower_bound_comparison_clean.png`
- `results/figures/figure3_style_marginal_likelihood_comparison_clean.png`
- `results/figures/aevb_binarized_z2_latent_manifold.png`
- `results/figures/aevb_binarized_random_samples_z2_seed0.png`
- `results/figures/aevb_binarized_random_samples_z5_seed0.png`
- `results/figures/aevb_binarized_random_samples_z10_seed0.png`
- `results/figures/aevb_binarized_random_samples_z20_seed0.png`
- `results/figures/aevb_binarized_random_samples_z200_seed0.png`

Important note: use the `clean` Figure 2 and Figure 3 files. The original generated Figure 2 included pilot and hidden-100 marginal-likelihood rows through the generic collector, which compressed the y-axis. The original Figure 3 had an empty second subplot because v3 only ran `N_train=1000`. The clean figures fix these presentation issues without changing the underlying training results.

### Completeness check

All planned v3 runs are present:

- Main AEVB full runs: `5` latent dimensions times `3` seeds = `15` runs.
- Qualitative AEVB run: `z=2`, seed `0` = `1` run.
- Wake-Sleep full runs: `5` latent dimensions, seed `0` = `5` runs.
- Marginal likelihood runs: AEVB, Wake-Sleep, MCEM-HMC for `z=3`, hidden `100`, `N_train=1000`.

The main training metrics cover epochs `1` through `300`, evaluated at epoch `1`, every `5` epochs, and epoch `300`, giving `61` evaluation points per full run.

There are no missing planned CSVs or checkpoints. The main training metrics contain no invalid numerical values. Summary standard deviations are blank for single-seed groups, which is expected.

### Main AEVB latent-dimensionality sweep

The main AEVB sweep results come from `results/tables/report_aevb_latent_sweep_clean.csv`.

| Method | z_dim | Hidden dim | Test NELBO mean | Test NELBO std | Test BCE mean | Test KL mean | NELBO improvement vs previous z |
|---|---:|---:|---:|---:|---:|---:|---:|
| AEVB | 3 | 500 | 141.30 | 0.40 | 133.55 | 7.75 | N/A |
| AEVB | 5 | 500 | 124.41 | 0.37 | 113.07 | 11.33 | 16.90 |
| AEVB | 10 | 500 | 107.57 | 0.17 | 89.21 | 18.36 | 16.83 |
| AEVB | 20 | 500 | 100.61 | 0.71 | 73.98 | 26.63 | 6.96 |
| AEVB | 200 | 500 | 99.23 | 0.04 | 70.71 | 28.53 | 1.37 |

Interpretation:

- Increasing the latent dimension improves the test negative ELBO.
- The improvement is large from `z=3` to `z=20`.
- The improvement from `z=20` to `z=200` is only about `1.37`, showing diminishing returns.
- BCE decreases as latent dimension increases, indicating better reconstructions.
- KL increases as latent dimension increases, indicating that the model uses more latent information.
- The best test negative ELBO in the main AEVB sweep is `99.23` at `z=200`.

The final paper should describe this as a successful reproduction of the paper's main qualitative trend: larger latent spaces improve the lower bound, but gains saturate.

### AEVB vs Wake-Sleep lower-bound comparison

The matched seed-0 comparison comes from `results/tables/report_aevb_vs_wake_sleep_seed0_clean.csv`.

The table reports lower bounds. Since `lower_bound = -negative_elbo`, a larger lower bound is better. The final column below is the Wake-Sleep minus AEVB negative ELBO gap, so a positive value means AEVB is better.

| z_dim | AEVB lower bound | Wake-Sleep lower bound | AEVB lower-bound advantage | Wake-Sleep minus AEVB NELBO |
|---:|---:|---:|---:|---:|
| 3 | -141.76 | -146.83 | 5.07 | 5.07 |
| 5 | -124.50 | -127.97 | 3.46 | 3.46 |
| 10 | -107.41 | -112.97 | 5.56 | 5.56 |
| 20 | -100.76 | -106.35 | 5.59 | 5.59 |
| 200 | -99.26 | -108.02 | 8.76 | 8.76 |

Interpretation:

- AEVB outperforms Wake-Sleep at every matched latent dimension.
- The difference is not just a small numerical fluctuation; it is between about `3.46` and `8.76` NELBO points.
- The largest gap occurs at `z=200`, where Wake-Sleep does not benefit from the larger latent space as effectively as AEVB.
- This supports the paper's broad conclusion that AEVB is a more effective training method than Wake-Sleep for this class of neural latent-variable models.

### Marginal likelihood comparison

The marginal likelihood comparison comes from `results/metrics/marginal_likelihood_summary.csv`. Only test split values are summarized here.

| Method | Samples seen | Checkpoint epoch | Test estimated marginal log-likelihood | Test std | HMC acceptance |
|---|---:|---:|---:|---:|---:|
| AEVB | 1000 | 1 | -245.38 | 43.48 | 0.998 |
| AEVB | 5000 | 5 | -207.25 | 39.09 | 0.996 |
| AEVB | 10000 | 10 | -202.31 | 41.12 | 0.993 |
| AEVB | 20000 | 20 | -198.04 | 43.07 | 0.992 |
| AEVB | 30000 | 30 | -196.20 | 43.69 | 0.989 |
| AEVB | 50000 | 50 | -194.68 | 44.56 | 0.988 |
| Wake-Sleep | 1000 | 1 | -412.62 | 16.14 | 0.999 |
| Wake-Sleep | 5000 | 5 | -222.78 | 46.36 | 0.999 |
| Wake-Sleep | 10000 | 10 | -201.32 | 47.84 | 0.995 |
| Wake-Sleep | 20000 | 20 | -197.15 | 46.41 | 0.993 |
| Wake-Sleep | 30000 | 30 | -195.42 | 46.26 | 0.990 |
| Wake-Sleep | 50000 | 50 | -193.69 | 45.27 | 0.989 |
| MCEM | 1000 | 1 | -211.68 | 41.39 | 0.997 |
| MCEM | 5000 | 5 | -202.86 | 39.16 | 0.995 |
| MCEM | 10000 | 10 | -201.69 | 39.67 | 0.997 |
| MCEM | 20000 | 20 | -195.28 | 39.98 | 0.997 |
| MCEM | 30000 | 30 | -187.40 | 38.96 | 0.997 |
| MCEM | 50000 | 50 | -180.23 | 37.80 | 0.996 |

Interpretation:

- AEVB improves rapidly from epoch `1` to epoch `5`, then continues improving more gradually.
- Wake-Sleep starts very poorly at epoch `1`, improves rapidly by epoch `10`, and then becomes close to AEVB by epoch `50`.
- MCEM-HMC obtains the best final estimated marginal log-likelihood in this reduced setup.
- The result should not be used to argue that MCEM is generally better in scalable settings. MCEM-HMC is expensive and was only practical here because the experiment used `z_dim=3`, hidden dimension `100`, and `N_train=1000`.
- HMC acceptance is very high, usually above `0.99`. This suggests conservative HMC steps in the estimator. The estimates are usable for a reduced-scale comparison, but the final paper should explicitly mention that this is a practical estimator, not a fully exact reproduction of the original paper's marginal likelihood procedure.

### Qualitative results

The qualitative artifacts are:

- `results/figures/aevb_binarized_z2_latent_manifold.png`
- `results/figures/aevb_binarized_random_samples_z2_seed0.png`
- `results/figures/aevb_binarized_random_samples_z5_seed0.png`
- `results/figures/aevb_binarized_random_samples_z10_seed0.png`
- `results/figures/aevb_binarized_random_samples_z20_seed0.png`
- `results/figures/aevb_binarized_random_samples_z200_seed0.png`

The `z=2` latent manifold is interpretable but limited. It shows smooth transitions in a two-dimensional latent grid, but the learned manifold is dominated by a narrow set of digit-like transitions rather than covering all digit classes evenly. This is expected: two latent dimensions are too restrictive for full MNIST variability.

The random samples are recognizable but blurry/noisy. This is consistent with:

- A simple one-hidden-layer MLP decoder.
- Bernoulli likelihood over pixels.
- No convolutional inductive bias.
- No modern likelihood or sample-quality improvements.

The final paper should present qualitative samples as supporting evidence only. The central quantitative evidence should be the lower-bound and marginal-likelihood tables/figures.

## 8. Detailed Analysis

### Why the AEVB latent sweep behaves as expected

The latent dimension controls the capacity of the model to encode information about each image. With a very small latent space, the decoder must reconstruct diverse digit images from a compressed representation. This produces high reconstruction error, reflected in a high BCE term.

As `z_dim` increases:

- The decoder receives a richer latent representation.
- Reconstruction improves.
- BCE decreases.
- KL increases because the approximate posterior can carry more information away from the prior.

The observed values show this clearly:

| z_dim | BCE | KL | NELBO |
|---:|---:|---:|---:|
| 3 | 133.55 | 7.75 | 141.30 |
| 5 | 113.07 | 11.33 | 124.41 |
| 10 | 89.21 | 18.36 | 107.57 |
| 20 | 73.98 | 26.63 | 100.61 |
| 200 | 70.71 | 28.53 | 99.23 |

The important point is the tradeoff:

- From `z=3` to `z=20`, the BCE reduction is large enough to dominate the increase in KL.
- From `z=20` to `z=200`, BCE still improves slightly, but KL also increases, so total NELBO improves only marginally.

This is the standard VAE rate-distortion behavior. The BCE term is a distortion-like term; the KL term is a rate-like term. Better reconstructions require more latent information, but after a point the marginal gain becomes small.

### Why AEVB beats Wake-Sleep in the lower-bound comparison

AEVB directly optimizes a stochastic estimate of the ELBO. The encoder and decoder are trained jointly with an objective aligned with the evaluation metric.

Wake-Sleep optimizes two different objectives:

- The wake phase trains the decoder using latent samples from the encoder.
- The sleep phase trains the encoder using synthetic data from the model.

The sleep phase may not train the encoder on the true data distribution, especially early in training when the generative model is poor. This creates a mismatch between the training objective and the final test ELBO evaluator.

In the reproduction, AEVB has a consistent advantage over Wake-Sleep:

| z_dim | AEVB test NELBO seed 0 | Wake-Sleep test NELBO seed 0 | Difference |
|---:|---:|---:|---:|
| 3 | 141.76 | 146.83 | 5.07 |
| 5 | 124.50 | 127.97 | 3.46 |
| 10 | 107.41 | 112.97 | 5.56 |
| 20 | 100.76 | 106.35 | 5.59 |
| 200 | 99.26 | 108.02 | 8.76 |

The `z=200` result is especially informative. AEVB uses the larger latent space effectively, while Wake-Sleep does not obtain the same benefit. This supports the interpretation that AEVB is better aligned with variational lower-bound optimization.

### Convergence behavior

The final 50 epochs still improved test NELBO for both AEVB and Wake-Sleep. Earlier checks showed improvements from epoch `250` to epoch `300` of roughly `0.5` to `0.8` NELBO depending on the run. Therefore, the models had not fully plateaued. However, the remaining improvement rate was small compared with the gains achieved earlier in training.

This is important for the final discussion:

- The results are good enough to establish the main trends.
- Longer training might slightly improve all methods.
- Longer training is unlikely to reverse the main AEVB vs Wake-Sleep comparison, because the observed gaps are several NELBO points.

### Generalization gap

The final train-test gaps for AEVB were small. For example:

- `z=3`: test NELBO is about `1.4` to `1.7` worse than train.
- `z=10`: test NELBO is about `0.2` to `0.3` worse than train.
- `z=20`: test NELBO is about `0.1` to `0.4` worse than train.
- `z=200`: test NELBO is about `0.3` to `0.4` worse than train.

This suggests no severe overfitting in the main AEVB runs despite the large `z=200` latent space. The MAP L2 prior and the variational KL term both regularize the model.

### Marginal likelihood interpretation

The marginal likelihood experiment has a different purpose from the main lower-bound sweep. It asks whether the trained generative model assigns high marginal likelihood to data under an HMC-based estimator.

The reduced experiment shows:

- AEVB has a strong start compared with Wake-Sleep.
- Wake-Sleep catches up substantially by epoch `50`.
- MCEM-HMC reaches the best final estimate.

This is plausible because MCEM-HMC directly uses posterior sampling during training and is evaluated with a related posterior-sampling estimator. However, MCEM is much less scalable than AEVB. It was feasible only in the reduced `z=3`, hidden-100, `N_train=1000` setting.

The final paper should present this as:

```text
In a reduced low-dimensional marginal-likelihood experiment, MCEM-HMC obtained the highest final estimated marginal log-likelihood, while AEVB and Wake-Sleep became close by the end of training. This experiment is not a full reproduction of the original Figure 3, but it reproduces the paper's low-dimensional HMC comparison idea.
```

### Figure recommendations

Use:

- `figure2_style_lower_bound_comparison_clean.png`
- `figure3_style_marginal_likelihood_comparison_clean.png`

Do not use:

- `figure2_style_lower_bound_comparison.png`
- `figure3_style_marginal_likelihood_comparison.png`

Reason:

- The original Figure 2 accidentally includes pilot and hidden-100 marginal-likelihood metric rows, which compresses the y-axis.
- The original Figure 3 has an empty second panel because v3 intentionally ran only `N_train=1000`.

The clean figures use the same result data but present it correctly.

## 9. Comparison With The Original Paper

### Aligned aspects

The reproduction aligns with the original paper in the following important ways:

| Aspect | Original paper direction | v3 reproduction |
|---|---|---|
| Dataset | MNIST used for experiments | MNIST |
| Likelihood | Bernoulli-style image likelihood | Bernoulli decoder with logits |
| Prior | Standard Gaussian latent prior | `N(0, I)` |
| Encoder | Recognition model / approximate posterior | Diagonal Gaussian encoder |
| Optimization | Stochastic gradient optimization with Adagrad | Adagrad |
| LR selection | Global Adagrad step size selected from candidates | Selected from `[0.01, 0.02, 0.1]` |
| MNIST hidden units | 500 hidden units | 500 hidden units |
| Latent dimensions | Multiple latent dimensions including MNIST sweep | `3, 5, 10, 20, 200` |
| Wake-Sleep comparison | Compared against Wake-Sleep | Wake-Sleep baseline implemented |
| Marginal likelihood | Low-dimensional HMC-based estimate | Reduced HMC-based estimate |

### Differences and deviations

The reproduction differs from the original paper in the following ways:

| Aspect | v3 choice | Why it matters |
|---|---|---|
| Binarization | Static Bernoulli binarization with seed `2026` | Improves reproducibility; may differ from original preprocessing details |
| Training budget | `300` epochs for main runs | Practical Colab budget; not necessarily the paper's full long-run scale |
| Seeds | AEVB uses `3` seeds; Wake-Sleep uses `1` seed | AEVB has mean/std; Wake-Sleep comparison is seed-0 only |
| Marginal likelihood train size | `N_train=1000` only | Reduced version; does not reproduce full `N_train=50000` setting |
| Marginal likelihood estimator | Full-covariance Gaussian fitted to HMC posterior samples | Explicit implementation choice because the paper's density estimator detail is not fully specified |
| Architecture type | MLP only | Paper also includes other datasets/settings; this project focuses on MNIST |
| Earlier grayscale run | Not reported | Kept only as preliminary debugging |

### How to phrase this in the final report

Recommended wording:

```text
We reproduce the MNIST part of Kingma and Welling's AEVB experiments with a paper-aligned practical setup. Our main experiments use statically binarized MNIST, a Bernoulli decoder, diagonal Gaussian recognition model, one hidden layer with 500 tanh units, Adagrad optimization, and the latent dimensions used in the paper's MNIST sweep. We also include a Wake-Sleep baseline and a reduced low-dimensional HMC/MCEM marginal-likelihood comparison. Some aspects are intentionally practical rather than exact: the binarization is fixed by seed for reproducibility, Wake-Sleep is run with one seed, and the marginal-likelihood experiment uses N_train=1000 only with an explicitly specified full-covariance Gaussian density estimator for posterior samples.
```

## 10. Limitations and Threats to Validity

### Not a complete reproduction of every paper result

The project reproduces the MNIST-centered part of the AEVB paper, not the entire paper. It does not reproduce every dataset, every original training duration, or every original marginal-likelihood setting.

### Reduced marginal likelihood experiment

The marginal likelihood experiment is useful but limited:

- Only `N_train=1000` was used.
- Only `z_dim=3`, hidden `100` was used.
- HMC evaluation used `max_points=200`.
- Posterior samples were limited to `30`.
- The density estimator was explicitly chosen by us.

Therefore, this result should be treated as a reduced-scale comparison.

### HMC acceptance too high

The HMC acceptance rates in the marginal likelihood estimator are very high, often above `0.99`. This suggests step sizes were conservative. Conservative HMC is stable but may mix slowly. The estimates are still useful for relative trends, but the report should disclose this.

### Wake-Sleep has fewer seeds

AEVB uses seeds `[0, 1, 2]`, while Wake-Sleep uses seed `0` only. The comparison is still informative because the gaps are substantial, but the formal paper should not present Wake-Sleep standard deviations unless additional seeds are run.

### Simple model architecture

The model uses MLPs rather than convolutional networks. This matches the old-paper style better than modern architectures, but it limits sample quality. Blurry or noisy samples are expected.

### Static binarization

Static Bernoulli binarization improves reproducibility but may differ from dynamic binarization or other preprocessing choices used in other VAE implementations. This affects absolute likelihood values.

### Remaining training improvement

The main models were still slightly improving near epoch `300`. Longer training might improve results by a small amount. The main conclusions are unlikely to change, but this should be mentioned as a practical compute-budget limitation.

## 11. What To Use In The Final 5-Page Report

### Recommended main structure

The final report should be much shorter than this handoff. Suggested five-page structure:

1. Introduction and paper summary.
2. Method: latent-variable model, ELBO, reparameterization, Bernoulli decoder.
3. Reproduction setup: MNIST, static binarization, architecture, optimizer, latent sweep, Wake-Sleep, reduced marginal likelihood.
4. Results: AEVB latent sweep, AEVB vs Wake-Sleep, qualitative samples/manifold, reduced marginal likelihood.
5. Discussion: what matched the paper, deviations, limitations, what was challenging.

### Figures to include

Use these figures:

1. `results/figures/figure2_style_lower_bound_comparison_clean.png`
2. `results/figures/figure3_style_marginal_likelihood_comparison_clean.png`
3. `results/figures/aevb_binarized_z2_latent_manifold.png`

If there is space, include one random sample grid, preferably:

```text
results/figures/aevb_binarized_random_samples_z10_seed0.png
```

or combine multiple sample grids in the appendix.

### Tables to include

Use the AEVB latent sweep table:

| z_dim | Test NELBO mean | Test NELBO std | Test BCE mean | Test KL mean |
|---:|---:|---:|---:|---:|
| 3 | 141.30 | 0.40 | 133.55 | 7.75 |
| 5 | 124.41 | 0.37 | 113.07 | 11.33 |
| 10 | 107.57 | 0.17 | 89.21 | 18.36 |
| 20 | 100.61 | 0.71 | 73.98 | 26.63 |
| 200 | 99.23 | 0.04 | 70.71 | 28.53 |

Use the AEVB vs Wake-Sleep comparison:

| z_dim | AEVB test NELBO seed 0 | Wake-Sleep test NELBO seed 0 | Wake-Sleep minus AEVB |
|---:|---:|---:|---:|
| 3 | 141.76 | 146.83 | 5.07 |
| 5 | 124.50 | 127.97 | 3.46 |
| 10 | 107.41 | 112.97 | 5.56 |
| 20 | 100.76 | 106.35 | 5.59 |
| 200 | 99.26 | 108.02 | 8.76 |

### Suggested final paper language for results

Possible wording:

```text
The AEVB sweep shows a monotonic improvement in test negative ELBO as the latent dimensionality increases. The largest gains occur between z=3 and z=20, while the improvement from z=20 to z=200 is only 1.37 nats per datapoint, suggesting diminishing returns. The BCE term decreases with larger latent spaces, while the KL term increases, indicating that the model trades greater latent information usage for better reconstructions.
```

Possible wording for Wake-Sleep:

```text
Using the same architecture and evaluator, AEVB outperforms Wake-Sleep for every matched latent dimension. The gap ranges from 3.46 to 8.76 negative-ELBO points in the seed-0 comparison. This supports the original paper's qualitative conclusion that directly optimizing the reparameterized variational lower bound is more effective than the Wake-Sleep training objective in this setting.
```

Possible wording for marginal likelihood:

```text
We also ran a reduced low-dimensional marginal-likelihood comparison with z=3, hidden dimension 100, and N_train=1000. MCEM-HMC obtained the best final estimated marginal log-likelihood, while AEVB and Wake-Sleep became close after 50 epochs. Since this estimator uses only 200 evaluation points and an explicitly chosen full-covariance Gaussian density estimator for HMC posterior samples, we report it as a reduced-scale reproduction of the paper's HMC comparison idea rather than an exact reproduction of Figure 3.
```

### What not to include

Do not include:

- v1 grayscale results.
- v2 intermediate results.
- The unclean original Figure 2.
- The unclean original Figure 3.
- Claims that the random samples are high quality.
- Claims that the marginal likelihood experiment exactly matches the original paper.

## 12. AI Use Disclosure Draft

The course requires disclosure of AI tool use. A suitable draft is:

```text
AI tools were used during this project to help plan the reproduction strategy, organize the codebase, draft implementation scaffolding, debug Colab/PyTorch issues, and summarize experimental results. The final experiments were executed by the authors, and all reported numerical results were taken from the generated CSV files and inspected manually. The authors remain responsible for the correctness of the code, the experimental setup, and the interpretation of the results.
```

If the final report has a separate AI-use section outside the five-page limit, include the above paragraph there. If the report must also mention group contribution, add a separate short paragraph explaining who wrote the code, who ran experiments, and who wrote the final paper.

## 13. Appendix: Commands, File Inventory, and Result Paths

### Main commands

The full suite command:

```bash
python scripts/run_v3_suite.py --backup-dir /content/drive/MyDrive/AEVB_MNIST_Reproduction_v3_backups
```

Manual command sequence:

```bash
python scripts/run_smoke_tests.py --config configs/quick_debug.yaml
python scripts/run_pilot.py --config configs/aevb_binarized_practical.yaml
python scripts/run_aevb_sweep.py --config configs/aevb_binarized_practical.yaml --resume
python scripts/run_wake_sleep.py --config configs/wake_sleep_binarized_practical.yaml --resume
python scripts/run_marginal_likelihood.py --config configs/marginal_likelihood_z3_h100_practical.yaml --resume
python scripts/make_figures.py --config configs/aevb_binarized_practical.yaml
python scripts/collect_results.py --results-dir results
```

Emergency backup command:

```bash
python scripts/backup_results.py --results-dir results --output-dir /content/drive/MyDrive/AEVB_MNIST_Reproduction_v3_backups
```

### Main configs

AEVB sweep:

```text
configs/aevb_binarized_practical.yaml
```

Wake-Sleep:

```text
configs/wake_sleep_binarized_practical.yaml
```

Marginal likelihood:

```text
configs/marginal_likelihood_z3_h100_practical.yaml
```

Quick debug:

```text
configs/quick_debug.yaml
```

### Main tables

Use:

```text
results/tables/report_aevb_latent_sweep_clean.csv
results/tables/report_aevb_vs_wake_sleep_seed0_clean.csv
results/tables/final_test_summary_by_method_z.csv
results/tables/final_summary_by_seed.csv
```

### Main metrics

Use:

```text
results/metrics/learning_rate_pilot_summary.csv
results/metrics/marginal_likelihood_summary.csv
```

Main training metric files follow names such as:

```text
results/metrics/aevb_static_bernoulli_z10_h500_seed0_lr0p01_e300.csv
results/metrics/wake_sleep_static_bernoulli_z10_h500_seed0_lr0p01_e300.csv
```

### Main figures

Use in report:

```text
results/figures/figure2_style_lower_bound_comparison_clean.png
results/figures/figure3_style_marginal_likelihood_comparison_clean.png
results/figures/aevb_binarized_z2_latent_manifold.png
results/figures/aevb_binarized_random_samples_z10_seed0.png
```

Optional supplementary sample figures:

```text
results/figures/aevb_binarized_random_samples_z2_seed0.png
results/figures/aevb_binarized_random_samples_z5_seed0.png
results/figures/aevb_binarized_random_samples_z20_seed0.png
results/figures/aevb_binarized_random_samples_z200_seed0.png
```

Do not use in report:

```text
results/figures/figure2_style_lower_bound_comparison.png
results/figures/figure3_style_marginal_likelihood_comparison.png
```

### GitHub submission notes

Recommended GitHub root:

```text
C:\Users\wangz\Desktop\ASI\Assignment\AEVB_MNIST_Reproduction_v3
```

Before submission:

- Merge `AEVB_MNIST_Reproduction_v3_results/results/` into the root `results/` directory.
- Keep `configs/`, `scripts/`, `src/`, `notebooks/`, `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`, and `.gitattributes`.
- Do not commit `__pycache__/`, `.egg-info/`, or zip files.
- Use Git LFS or a GitHub Release for `.pt` checkpoints.
- Commit CSV, PNG, JSON, YAML, Markdown, and source code directly.

### Final one-line summary for teammate

The final project should be written as a v3-only, paper-aligned practical reproduction of AEVB on statically binarized MNIST, centered on the AEVB latent sweep, the Wake-Sleep ELBO comparison, and the reduced low-dimensional marginal likelihood experiment, with all numeric claims taken from the local v3 CSV results.

