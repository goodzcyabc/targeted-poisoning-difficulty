# Third-party code

This repository builds on two public research codebases. Their files are kept at the top level so that the
original import structure and evaluation protocol are unchanged; our own additions live in `metrics/`,
`scripts/` and `results/`.

- **Poisoning benchmark** — the attack-evaluation harness (`learning_module.py`, `poison_test.py`,
  `benchmark_test.py`, `test_model.py`, `train_model.py`, `tinyimagenet_module.py`, `models/`,
  `poison_crafting/`, `poison_setups/`). Distributed under the license in `LICENSE.txt`.
- **Gradient matching ("Witches' Brew")** — used to craft the GM poisons evaluated here.

Full citations are given in the paper; they are omitted from this README only to preserve anonymity during
review.
