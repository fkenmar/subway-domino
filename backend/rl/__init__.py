"""Subway Domino reinforcement-learning pipeline (ARCHITECTURE §4.5-§4.10).

This is the only package allowed to import torch, gymnasium, stable-baselines3,
sb3-contrib or tensorboard (CLAUDE.md, import firewall). Everything it trains against
(physics, observations, masks, cost) comes from :mod:`sim`, the same code path the server
uses.

Modules:

* :mod:`rl.env` - ``SubwayDominoEnv``: the AI sim plus its counterfactual Normal twin.
* :mod:`rl.vec` - picklable env factories and vectorized-env construction.
* :mod:`rl.train` / :mod:`rl.callbacks` - MaskablePPO training with a curriculum, atomic
  checkpoints, validation and resumable runs (local or Colab).
* :mod:`rl.evaluate` - Normal vs AI on the fixed evaluation seeds.
* :mod:`rl.export_onnx` - actor-only ONNX export with a parity check and model card.
* :mod:`rl.bench_env` - environment throughput benchmark.
* :mod:`rl.make_colab_notebook` / :mod:`rl.make_repo_zip` - the Colab workflow.
"""
