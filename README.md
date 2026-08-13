# CBM (Computational Brain/Behavior Modeling)

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

"Computational Brain/Behavior Modeling (CBM)" Library for model fitting and model selection.

> [!NOTE]
> **This branch adds cross-task joint modelling.**
> `feature-joint-modeling` introduces an opt-in `HBIConfig.covariance_blocks`, which lets
> the HBI group level carry a non-diagonal covariance over chosen parameter pairs — so you
> can estimate how parameters covary across subjects, within or across tasks.
>
> - **To use it:** [HOWTO_JOINT_MODELING.md](HOWTO_JOINT_MODELING.md) — install, a worked
>   example, and the settings that matter.
> - **To review it:** [README_JOINT_MODELING.md](README_JOINT_MODELING.md) — a file-by-file
>   diff against upstream `ccb8aa8`, with the open questions in section 7.
> - **To check it:** `python -m cbm.test_joint_elbo` — confirms the joint branch reproduces
>   the standard branch exactly when no pairs are linked.
>
> This is a fork, not part of the official CBM toolbox. Please send joint-modelling
> questions to Ben J. Wagner (ben.wagner@tuebingen.mpg.de) rather than upstream.

## Overview

CBM provides a complete pipeline for fitting computational models to behavioral data and comparing competing models at the group level. The toolkit implements three core methods:

1. **Individual-level fitting** uses Laplace approximation to estimate maximum a posteriori (MAP) parameters for each subject, providing both parameter estimates and model evidence.

2. **Bayesian Model Selection (BMS)** implements random effects analysis to determine the frequency of different models in the population, accounting for between-subject variability.

3. **Hierarchical Bayesian Inference (HBI)** performs joint fitting and model comparison, pooling information across subjects and models to improve parameter estimation while simultaneously inferring model frequencies.

The library is designed for cognitive modeling applications where multiple computational models are fit to data from multiple subjects, and researchers need to determine which model best explains the data at the group level.

## Features

- **Individual Fitting**: Maximum a posteriori (MAP) estimation with Laplace approximation
- **Bayesian Model Selection**: Random effects model selection with exceedance probabilities
- **Hierarchical Bayesian Inference**: Group-level inference across subjects and models

## Installation

```bash
git clone https://github.com/payampiray/cbm_python.git
cd cbm_python
pip install -e .
```

## Examples

- `examples/exampla_individual_fit.py`: Minimal individual fit
- `examples/example_model_selection.py`: BMS with linear models
- `examples/example.py`: HBI with linear models
- `examples/example_RL.py`: Individual fit and HBI with RL vs RL2 (dual learning rates)

Outputs are written to `examples/output/`.

## Development

- Python >= 3.9
- Dependencies: NumPy, SciPy, Pickle
- Logs and pickle outputs are stored under `examples/output/`


## Related packages

- [cbm_power](https://github.com/payampiray/cbm_power): CBM library for power analysis and sample-size optimization for computational studies.

- A MATLAB implementation of CBM: [github.com/payampiray/cbm](https://github.com/payampiray/cbm).

## References
If you use this package, please cite the following paper:
- Piray et al., "Hierarchical Bayesian inference for concurrent model fitting and comparison for group studies", *PLoS Computational Biology*, 2019.
