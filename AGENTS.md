# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Active numerics: `second_draft/pqes_new_numerics/` (see its `README.md`). Its CSVs in `second_draft/data/pqes_new_numerics/` are published results. A change must leave them byte-identical under the defaults, with new results going into new CSVs.
- **Reproducing the committed CSVs:** use a venv with Python 3.13.1, `numpy==2.5.4`, `pandas==3.0.6`, `scipy==1.18.1`, `matplotlib==3.11.2` and `pytest`. Other NumPy versions change the sampled columns.
- **Verifying byte-identity:**
  - run `run_simulations.py --profile full --seed 31 --rmse 0.03 --data-dir <scratch>` (about 10 min) and `cmp` each CSV against the committed one;
  - `manifest.json` always differs in `task_runtimes_s`;
  - `--task readout_rule` writes only its own CSVs and `manifest_readout_rule.json`.
- `second_draft/manuscript/` and `third_draft/` are never committed to this repository.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
