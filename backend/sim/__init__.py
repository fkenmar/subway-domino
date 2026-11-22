"""Subway Domino simulation package (ARCHITECTURE §3-§4): pure NumPy, shared by training and serving.

Importing ``sim`` never imports torch, gymnasium, stable-baselines3, sb3-contrib or
tensorboard; ``onnxruntime`` is imported lazily by :mod:`sim.policy` only.
"""

from __future__ import annotations

from sim import constants
from sim.engine import ActionEvent, DecisionBatch, NYCSubwaySim
from sim.graph import Graph, load_graph
from sim.obs import FEATURE_NAMES
from sim.policy import OnnxPolicy, load_policy
from sim.scenarios import Scenario, ScheduledDisruption, eval_scenario, preset_scenario, sample_training_scenario

__all__ = [
    "NYCSubwaySim",
    "Graph",
    "load_graph",
    "Scenario",
    "ScheduledDisruption",
    "DecisionBatch",
    "ActionEvent",
    "OnnxPolicy",
    "load_policy",
    "constants",
    "FEATURE_NAMES",
    "sample_training_scenario",
    "eval_scenario",
    "preset_scenario",
]
