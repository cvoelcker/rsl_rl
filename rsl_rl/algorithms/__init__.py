# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different learning algorithms."""

from .distillation import Distillation
from .ppo import PPO
from .reppo import REPPO
from .reppo_hybrid import REPPO as REPPOHybrid

__all__ = ["PPO", "REPPO", "Distillation", "REPPOHybrid"]
