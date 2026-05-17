# PROMETHEE & ELECTRE Tri-B — Outranking Analysis of European Countries

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![pandas](https://img.shields.io/badge/pandas-3.0+-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![NetworkX](https://img.shields.io/badge/NetworkX-2C3E50?style=flat-square)](https://networkx.org)
[![Graphviz](https://img.shields.io/badge/Graphviz-PyGraphviz-2C3E50?style=flat-square)](https://pygraphviz.github.io)
[![uv](https://img.shields.io/badge/uv-de5d43?style=flat-square)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](../LICENSE)

Two complementary outranking analyses on the same 26-country / 8-criterion
dataset used in the UTA/AHP project: **PROMETHEE I/II** for ranking, and
**ELECTRE Tri-B** for sorting into ordered categories. Both methods are
implemented from scratch — the same `difference_function`, the same
preference-threshold semantics, the same dataset — so the two analyses
fully agree on the model's input and only differ in the aggregation rules
and the type of recommendation produced.

## 📊 Dataset

`dataset-uta/` mirrors the dataset prepared for Project 1: 26 European OECD
members on eight quality-of-life criteria (employment, long-term
unemployment, earnings, life expectancy, life satisfaction, work-life
balance, air quality, distance from Poznań). Preference information adds
indifference $q$, preference $p$ and — for ELECTRE — veto $v$ thresholds
per criterion, plus criterion weights $w$. ELECTRE Tri-B uses two boundary
profiles, partitioning the alternatives into three ordered categories.

A small toy dataset (`dataset.csv` / `preference.csv` at the root of each
subdirectory) is also included for sanity-checking individual functions
during development.

## 🧠 Methodology

### PROMETHEE I and II

For every alternative pair $(a, b)$ and criterion $j$:

1. Signed difference $d_j(a, b)$ accounting for gain / cost direction.
2. V-shape marginal preference $P_j(d) = \max(0, \min(1, (d - q_j) / (p_j - q_j)))$.
3. Comprehensive preference index $\pi(a, b) = \sum_j w_j P_j(d_j(a, b)) / \sum_j w_j$.

From the comprehensive matrix the **positive**, **negative** and **net**
flows are extracted:

$$\varphi^+(a) = \sum_b \pi(a, b), \quad \varphi^-(a) = \sum_b \pi(b, a), \quad \varphi(a) = \varphi^+(a) - \varphi^-(a)$$

PROMETHEE I builds a *partial* ranking by intersecting the two flows: $a \succeq_I b$
iff $\varphi^+(a) \ge \varphi^+(b)$ and $\varphi^-(a) \le \varphi^-(b)$.
PROMETHEE II builds a *complete* ranking by total-ordering on $\varphi$.

### ELECTRE Tri-B

The same preference information feeds the concordance/discordance pipeline:

1. **Marginal concordance** $c_j(a, b) = \max(0, \min(1, (d_j + p_j) / (p_j - q_j)))$.
2. **Comprehensive concordance** $C(a, b) = \sum_j w_j c_j(a, b) / \sum_j w_j$.
3. **Marginal discordance** $d_j(a, b)$ using the veto threshold $v_j$ — 1 below $-v_j$, 0 above $-p_j$, linear in between.
4. **Credibility** $\sigma(a, b) = C(a, b) \prod_{d_j > C} (1 - d_j) / (1 - C)$.
5. **Outranking** $a S b \iff \sigma(a, b) \ge \lambda$ for a cutoff $\lambda$.

Two assignment rules — *pessimistic* (top-down scan: assign to the highest
category whose lower boundary the alternative outranks) and *optimistic*
(bottom-up scan: assign to the first category whose upper boundary outranks
the alternative) — produce a pair of category labels per country.

## 📁 Project Structure

```
Project-3-Promethee-Electre/
├── mcda_common/                          # Reusable library version of the algorithms
│   ├── __init__.py
│   ├── types.py                          # CriterionType enum
│   ├── io.py                             # CSV loaders
│   ├── promethee.py                      # PROMETHEE I & II pipeline
│   ├── electre.py                        # ELECTRE Tri-B pipeline
│   └── viz.py                            # Ranking graphs, marginal preference plots
├── promethee-exercises/
│   ├── 01_promethee.ipynb                # Full PROMETHEE walkthrough
│   ├── utils.py                          # Thin re-export of mcda_common
│   ├── dataset.csv                       # Toy dataset
│   ├── preference.csv                    # Toy preference info
│   └── dataset-uta/                      # 26-country UTA dataset
├── Electre-exercises/
│   ├── 06_electre_tri.ipynb              # Full ELECTRE Tri-B walkthrough
│   ├── utils.py                          # Thin re-export of mcda_common
│   ├── dataset.csv                       # Toy dataset
│   ├── boundary_profiles.csv             # Toy boundary profiles
│   ├── preference.csv                    # Toy preference info with veto thresholds
│   └── dataset-uta/                      # 26-country UTA dataset with veto thresholds
├── pyproject.toml
└── uv.lock
```

The two notebooks define the analyses inline; `mcda_common/` is the
importable, type-hinted, docstring-covered library version of the same
algorithms. Each `utils.py` is a small re-export that prepends the package
root to `sys.path` and pulls in only the symbols the corresponding notebook
needs, so the existing `import utils` / `from utils import display_ranking`
calls keep working without notebook edits.

## 🚀 Setup

```bash
uv sync                # install dependencies
uv run jupyter lab     # open the two notebooks
```

Or, from a Python REPL:

```python
from pathlib import Path
from mcda_common.io import load_dataset, load_preference_information
from mcda_common.promethee import (
    calculate_marginal_preference_matrix,
    calculate_comprehensive_preference_index,
    calculate_positive_flow,
    calculate_negative_flow,
    calculate_net_flow,
)

dataset = load_dataset(Path("promethee-exercises/dataset-uta"))
prefs = load_preference_information(Path("promethee-exercises/dataset-uta"))
m = calculate_marginal_preference_matrix(dataset, prefs)
pi = calculate_comprehensive_preference_index(m, prefs)
phi_plus = calculate_positive_flow(pi, dataset.index)
phi_minus = calculate_negative_flow(pi, dataset.index)
phi = calculate_net_flow(phi_plus, phi_minus).sort_values(ascending=False)
```

## 🏛️ Key Architecture Decisions

- **Two notebooks, one library.** PROMETHEE and ELECTRE Tri-B share their
  preference-threshold semantics and gain/cost direction logic; the
  duplicate `difference_function` in the original `utils.py` files now
  lives in `mcda_common.promethee` and is reused by the ELECTRE module via
  a single import.
- **Thin notebook shims, no notebook edits.** The two `utils.py` files
  are kept as compatibility layers so the notebooks' `import utils` lines
  remain valid. The shims push the project root onto `sys.path` so
  `mcda_common` resolves regardless of which subdirectory the notebook is
  launched from.
- **Backward-compatible aliases for the original names.** Functions whose
  original names contained typos (`calculate_pessimistic_assigment`,
  `test_marginal_preference_function`) are preserved as aliases for the
  corrected names so the notebooks keep working.
- **The two utility CSV files in each subfolder remain.** The toy datasets
  are tiny and used for unit-test-style sanity checks during the
  notebook's setup section; keeping them next to the notebook is more
  ergonomic than moving them under a shared `tests/` directory.

## 📚 References

- *A Preference Ranking Organisation Method (The PROMETHEE Method for Multiple Criteria Decision-Making)* — Brans, Vincke (1985). Management Science 31(6). [doi:10.1287/mnsc.31.6.647](https://doi.org/10.1287/mnsc.31.6.647)
- *The Outranking Approach and the Foundations of ELECTRE Methods* — Roy (1991). Theory and Decision 31. [doi:10.1007/BF00134132](https://doi.org/10.1007/BF00134132)

## 📝 License

MIT — see [LICENSE](../LICENSE).
