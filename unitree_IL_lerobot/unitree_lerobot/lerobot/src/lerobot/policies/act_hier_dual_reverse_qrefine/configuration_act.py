#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act_hier_dual_reverse.configuration_act import ACTHierDualReverseConfig


@PreTrainedConfig.register_subclass("act_hier_dual_reverse_qrefine")
@dataclass
class ACTHierDualReverseQRefineConfig(ACTHierDualReverseConfig):
    """Dual reverse ACT policy with query refinement."""
