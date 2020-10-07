# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import json
import gym
import ray

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from procgen_ray_launcher import ProcgenSageMakerRayLauncher

from ray_experiment_builder import RayExperimentBuilder

from utils.loader import load_algorithms, load_preprocessors
try:
    from procgen.envs.procgen_env_wrapper import ProcgenEnvWrapper
    from procgen.envs.framestack import FrameStack
except ModuleNotFoundError:
    from envs.procgen_env_wrapper import ProcgenEnvWrapper
    from envs.framestack import FrameStack

class MyLauncher(ProcgenSageMakerRayLauncher):
    def register_env_creator(self):
        pass

    def _get_ray_config(self):
        return {
            # Leave "ray_num_cpus" and "ray_num_gpus" blank for multi-instance training
            # "ray_num_cpus": 16,
            # "ray_num_gpus": 0,
            "eager": False,
             "v": True, # requried for CW to catch the progress
        }

    def _get_rllib_config(self):
        return {
            "queue_trials": True,
            "config_file": "experiments/impala.yaml"
        }

    def register_algorithms_and_preprocessors(self):
        try:
            from custom.algorithms import CUSTOM_ALGORITHMS
            from custom.preprocessors import CUSTOM_PREPROCESSORS
            from custom.models.my_vision_network import MyVisionNetwork
            from custom.models.impala_cnn_tf import ImpalaCNN
        except ModuleNotFoundError:
            from algorithms import CUSTOM_ALGORITHMS
            from preprocessors import CUSTOM_PREPROCESSORS
            from models.my_vision_network import MyVisionNetwork
            from models.impala_cnn_tf import ImpalaCNN

        load_algorithms(CUSTOM_ALGORITHMS)

        load_preprocessors(CUSTOM_PREPROCESSORS)
        ModelCatalog.register_custom_model("my_vision_network", MyVisionNetwork)
        ModelCatalog.register_custom_model("impala_cnn_tf", ImpalaCNN)

    def get_experiment_config(self):
        params = dict(self._get_ray_config())
        params.update(self._get_rllib_config())
        reb = RayExperimentBuilder(**params)
        return reb.get_experiment_definition()

if __name__ == "__main__":
    MyLauncher().train_main()
