# Preference Learning — Three Models, One Dataset

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0+-008000?style=flat-square)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.51+-blueviolet?style=flat-square)](https://shap.readthedocs.io)
[![uv](https://img.shields.io/badge/uv-de5d43?style=flat-square)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](../LICENSE)

A side-by-side study of three preference-learning approaches on the same
dataset: a monotone-constrained gradient-boosted tree, a fully interpretable
neural MCDA model (ANN-UTADIS), and a conventional deep MLP. All three are
trained on identical splits and seeds; the report contrasts accuracy,
class-wise behaviour, and interpretability — the deep MLP shines on the
metric column, the ANN-UTADIS gives directly readable marginal utility
functions and per-criterion class thresholds, and the monotone XGBoost is
the practical middle ground.

## 📊 Dataset

`data/lectures evaluation.csv` — 1,000 alternatives, four ordinal criteria
$c_1, \dots, c_4$ on the grid $\{0, 0.25, 0.5, 0.75, 1.0\}$, and a quality
class in $\{0, 1, 2, 3, 4\}$. The five raw classes are imbalanced (class 4
holds only 27 samples), so they are merged into three:

| Raw class | Merged class | Label |
|---|---|---|
| 0, 1 | 0 | low |
| 2 | 1 | medium |
| 3, 4 | 2 | high |

The processed dataset is stratified 80/20 with `random_state = 1234`.

## 🧠 Methodology

### 1. XGBoost with monotone constraints

A 100-tree, depth-3 booster fit with `monotone_constraints = (1, 1, 1, 1)`,
enforcing that an increase in any criterion can only increase the predicted
score. SHAP is used for both per-alternative explanations (waterfall plots
on the chosen example alternatives) and global feature importance.

### 2. ANN-UTADIS

A small monotonic neural network implementing the additive value-function
formulation

$$U(a) = \sum_{j=1}^{m} u_j(g_j(a)),\quad \text{class}(a) = k \iff t_{k-1} \le U(a) < t_k$$

with $L = 50$ hidden components per criterion. The marginal utility
functions $u_j$ are read directly off the trained spread → leaky-hard-sigmoid →
combine block in [`layers/`](layers/); the class thresholds $t_k$ are the
learned parameters of the [`OrdinalThresholdLayer`](layers/threshold_layer.py).

Anchoring $U(0) = 0$ and $U(1) = 1$ is done by wrapping `Uta` in a
[`NormLayer`](layers/norm_layer.py); the ordinal layer is parameterised via
cumulative softplus so the thresholds stay strictly ordered throughout
training. The leaky-hard-sigmoid's slope is annealed linearly from $10^{-2}$
to $3 \times 10^{-3}$, sharpening the marginal utility plateaus as training
converges.

### 3. Deep MLP

A plain $4 \to 64 \to 32 \to 16 \to 3$ ReLU network with 0.2 dropout,
trained with Adam at $5 \times 10^{-3}$ for up to 600 epochs. The interpretability
analysis falls back to model-agnostic tools — permutation feature importance,
partial dependence plots, SHAP on the logits.

## 📁 Project Structure

```
Project 2 - Preference Learning/
├── data/
│   └── lectures evaluation.csv           # raw dataset (no header)
├── layers/                               # ANN-UTADIS PyTorch building blocks
│   ├── __init__.py
│   ├── uta.py                            # comprehensive utility wrapper
│   ├── monotonic_layer.py                # spread → activation → combine
│   ├── criterion_layer_spread.py
│   ├── criterion_layer_combine.py
│   ├── leaky_hard_sigmoid.py
│   ├── norm_layer.py                     # anchors U(0)=0, U(1)=1
│   └── threshold_layer.py                # binary + ordinal K-1 thresholds
├── src/                                  # importable training / explanation helpers
│   ├── __init__.py
│   ├── config.py                         # XgbConfig / AnnUtadisConfig / MlpConfig
│   ├── training.py                       # generic train loop + early stopping
│   └── explain.py                        # per-criterion contributions, min-change
├── notebooks/
│   └── report.ipynb                      # consolidated comparison report
├── reports/
│   └── report.html                       # static HTML export
├── pyproject.toml
└── uv.lock
```

## 🚀 Setup

```bash
uv sync                                                       # install dependencies
uv run jupyter lab                                            # open notebooks/report.ipynb
uv run jupyter nbconvert --to html --output-dir reports notebooks/report.ipynb
```

The notebook runs top-to-bottom from a clean kernel; random seeds are pinned
so the reported metrics are reproducible.

## 🏛️ Key Architecture Decisions

- **Monotonicity by construction, not by penalty.** The ANN-UTADIS marginal
  utility is monotonic in every criterion because the spread weights, the
  activation, and the combine weights are each individually monotonic — no
  monotonicity loss term or sign-constrained optimiser is needed.
- **Cumulative-softplus thresholds instead of constrained optimisation.**
  Ordering $t_1 < t_2 < \dots < t_{K-1}$ is enforced by parameterising each
  gap as `softplus(raw)`. The thresholds are strictly ordered for any
  real-valued `raw` vector, so the optimiser can use any unconstrained
  gradient method.
- **Temperature-controlled cumulative sigmoids.** Hard bucketise on the
  thresholds gives non-differentiable predictions, which break SHAP and
  AUC; the small `temperature` in [`OrdinalThresholdLayer`](layers/threshold_layer.py)
  yields a sharp but differentiable distribution that `argmax`'s to the
  same class as the hard rule.
- **`src/` package is the library twin of the notebook.** Training loops,
  SHAP setups, and the minimum-change analyses are duplicated in the
  notebook on purpose — the report stays self-contained — but `src/` is the
  reusable version of the same code, importable from other scripts.

## 📚 References

- *Rank consistent ordinal regression for neural networks with application to age estimation* — Cao, Mirjalili, Raschka (2020). Pattern Recognition Letters. [arXiv:1901.07884](https://arxiv.org/abs/1901.07884)
- *A Unified Approach to Interpreting Model Predictions* — Lundberg, Lee (2017). NeurIPS. [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)

## 📝 License

MIT — see [LICENSE](../LICENSE).
