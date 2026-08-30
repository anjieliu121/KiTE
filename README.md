# KiTE

KiTE (Kernel-based AI Trustworthiness Examiner) is a JAX library for checking whether a binary classifier is locally calibrated, and for adjusting its probabilities when it is not.

A model can look well calibrated on average and still be wrong for particular groups (age, race, income). KiTE measures that gap with kernel local calibration error (KLCE) and can train a small correction on top of an existing predictor.

The statistic and test come from Vashistha and Farahi (AISTATS 2025). Related earlier work on U-trustworthy models is in their AAAI 2024 paper.

## What KLCE measures

Local calibration error (LCE) asks whether predicted probabilities match outcomes in neighborhoods of feature space, not only on the whole sample.

KLCE is the kernel form of that error. With kernels $k$ on predicted probabilities and $l$ on features, a model $f$ is I-trustworthy if and only if $\mathrm{KLCE}^2 = 0$. The unbiased estimator is a U-statistic on residuals $y - f$, with the diagonal dropped.

KiTE uses KLCE in two ways:

1. A permutation test (`KLCE_test`) for the null that the model is locally calibrated.
2. A penalty while training a recalibration network (`recalibrated_model`).

## Install

```bash
pip install git+https://github.com/anjieliu121/KiTE.git
```

From a clone:

```bash
pip install .
```

Dependencies: `jax` and `optax` (see `pyproject.toml` for lower bounds). Python 3.9 or newer.

## Example

```python
import jax.random as random
from KiTE import KLCE_test, recalibrated_model

# X: calibration features, y: labels, p: base predicted probabilities
key = random.PRNGKey(0)
stat, p_value = KLCE_test(
    X, y, p,
    prob_kernel_width=0.1,
    iterations=200,
    key=key,
)

recal = recalibrated_model(num_steps=200, seed=0)
recal.fit(p, X, y)
p_hat = recal.predict_proba(p, X)
```

## Citation

Vashistha, R. & Farahi, A. (2025). I-trustworthy Models. A framework for trustworthiness evaluation of probabilistic classifiers. Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, PMLR 258:4726–4734. [https://arxiv.org/abs/2501.15617](https://arxiv.org/abs/2501.15617)

```bibtex
@inproceedings{vashistha2025itrustworthy,
  title     = {I-trustworthy Models. A framework for trustworthiness evaluation of probabilistic classifiers},
  author    = {Vashistha, Ritwik and Farahi, Arya},
  booktitle = {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  pages     = {4726--4734},
  year      = {2025},
  volume    = {258},
  publisher = {PMLR},
  url       = {https://arxiv.org/abs/2501.15617}
}
```

