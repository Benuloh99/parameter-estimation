# Parameter estimation

Notebooks for CHME 5137 (2026): fitting models to data, after Lectures 12 and 13.

| Notebook | What it is |
|---|---|
| `batch-reactor-regression.ipynb` | The regression homework: A ⇌ B in a batch reactor, fitted with `curve_fit` |

Each notebook has a paired `.py` file holding the same cells as plain text (see
**Jupytext** below).

## Getting started

1. **Fork** this repository to your own GitHub account (button at top right).
2. **Clone your fork**, not this one:
   ```bash
   git clone git@github.com:YOUR-USERNAME/parameter-estimation.git   # SSH, e.g. on Explorer
   # or
   git clone https://github.com/YOUR-USERNAME/parameter-estimation.git
   ```
   On Explorer, clone it into `/courses/CHME5137.202710/students/$USER/`.
3. Open `batch-reactor-regression.ipynb` and work through it, using your own kernel
   (`chme5137` on Explorer). Commit and push as you go:
   ```bash
   git add batch-reactor-regression.ipynb batch-reactor-regression.py
   git commit -m "Task 1: fit kf and kr at 298 K"
   git push
   ```
   This is your own fork, so committing on `main` is fine. Don't open a pull request
   back to `CHME5137/parameter-estimation`: that would publish your answers to everyone.

## Jupytext

`jupytext.toml` pairs each notebook with a `.py` script in the *percent* format, as in
the `differential-equations` and `kinetic-monte-carlo` repositories. Saving the notebook
updates the `.py` as well. If the two get out of step,
`jupytext --sync batch-reactor-regression.ipynb` brings them back in line.

---

*Written with Claude Code (Claude Opus 5.5) and checked by Prof. West.*
