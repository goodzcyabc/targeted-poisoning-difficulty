# Licensing and third-party code

## Our code

Everything outside `third_party/` was written for this paper and is released under the MIT license in
`LICENSE`: `metrics/`, `scripts/`, `results/`, and the documentation at the repository root.

## Third-party code

`third_party/poisoning-benchmark/` is the public data-poisoning benchmark this work builds on. Its files are
redistributed verbatim under their own MIT license, retained unmodified at
`third_party/poisoning-benchmark/LICENSE`. That license covers only that subtree.

It contains `learning_module.py`, `poison_test.py`, `test_model.py`, `train_model.py`,
`tinyimagenet_module.py`, `models/`, `poison_crafting/` and `poison_setups/`.

Two credits those files carry, passed through here:

- `tinyimagenet_module.py` is, by its own docstring, heavily based on code by Meng Lee (mnicnc404),
  https://github.com/leemengtaiwan/tiny-imagenet.
- Several files under `models/` are ports of widely-used CIFAR reference implementations and carry their own
  attribution headers, which have been left intact.

## What produced the results in this release

The poisons evaluated in the paper were crafted and evaluated with the public **gradient matching**
("Witches' Brew") framework. **That code is not redistributed here** — it was a tool we ran, not code we
ship. Every table under `results/attacks/` is in its output schema, which is why those tables cannot be
regenerated from this repository alone; the framework is cited in the paper.

The benchmark harness under `third_party/` is included because our target selection and our reported
`benchmark_idx` are defined against it: the `benchmark_idx` column in every attack table indexes into the
fixed setup lists in `third_party/poisoning-benchmark/poison_setups/*.pickle`. Those pickles are needed to
map our released tables back to concrete targets. No file in `metrics/` or `scripts/` imports anything under
`third_party/`.

Full citations for both upstream projects are given in the paper.
