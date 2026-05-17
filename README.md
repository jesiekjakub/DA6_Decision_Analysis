# Decision Analysis Portfolio

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PuLP](https://img.shields.io/badge/PuLP-3.3+-2C2D72?style=flat-square)](https://github.com/coin-or/pulp)
[![pandas](https://img.shields.io/badge/pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![uv](https://img.shields.io/badge/uv-de5d43?style=flat-square)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

Three end-to-end multi-criteria decision analysis projects sharing the same
26-country / 8-criterion European quality-of-life dataset. Each project
implements a different family of MCDA methods from scratch and writes the
full pipeline up in a report notebook.

| Project | Methods | Stack |
|---|---|---|
| [Project 1 — UTA & AHP](Project%201%20-%20UTA,%20AHP/) | UTA inconsistency MILP, UTA discrimination LP, AHP eigenvector method | Python, PuLP, pandas, NumPy, Matplotlib |
| [Project 2 — Preference Learning](Project%202%20-%20Preference%20Learning/) | Monotone XGBoost, ANN-UTADIS (interpretable neural MCDA), Deep MLP | PyTorch, XGBoost, scikit-learn, SHAP |
| [Project 3 — PROMETHEE & ELECTRE Tri-B](Project-3-Promethee-Electre/) | PROMETHEE I/II ranking, ELECTRE Tri-B sorting | Python, NumPy, pandas, NetworkX, Graphviz |

Each subproject is self-contained — its own `pyproject.toml`, `uv.lock`,
README, and report notebook — and is intended to be opened in isolation
with `uv sync` from inside that subdirectory. Pick a project below to read
the methodology and results in depth.

## 🚀 Quick Start

```bash
cd "Project 1 - UTA, AHP"           # or one of the others
uv sync
uv run jupyter lab
```

Each project ships with one or more report notebooks under `notebooks/` or
`reports/` that walk through the full analysis end-to-end. The committed
HTML exports (where present) are static snapshots of the latest run.

## 🧠 What ties the three projects together

- **One dataset, three lenses.** The UTA, AHP, PROMETHEE I/II and ELECTRE
  Tri-B analyses all operate on the same wide-format dataset of 26
  European countries on 8 quality-of-life criteria — built once in
  Project 1 and reused as-is by Project 3. The same decision-maker
  preferences (`q`, `p`, `w` thresholds, pairwise comparisons) thread
  through every method.
- **One DM, multiple recommendation types.** Project 1 produces a *value
  function*, Project 2 a *learned classifier*, Project 3 a *partial
  ranking* and a *sorting*. The contrast highlights what each method
  optimises for and what kind of recommendation it can deliver.
- **From scratch, with explicit math.** Every method is implemented from
  the LP / MILP / linear-algebra primitives upward; no MCDA library
  shortcuts. The piecewise-linear utility functions, the eigenvector
  consistency ratios, the cumulative-softplus ordinal thresholds, and
  the PROMETHEE / ELECTRE concordance pipelines are all readable in the
  source.

## 📁 Repository Structure

```
.
├── Project 1 - UTA, AHP/                  # Linear-programming MCDA + AHP
├── Project 2 - Preference Learning/       # Preference-learning model comparison
├── Project-3-Promethee-Electre/           # PROMETHEE + ELECTRE Tri-B outranking
├── .gitignore
├── LICENSE
└── README.md
```

## 📝 License

MIT — see [LICENSE](LICENSE).
