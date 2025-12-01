# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Right-Arm-Bot-Direct-v0",
    entry_point=f"{__name__}.right_arm_bot_env:RightArmBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.right_arm_bot_env_cfg:RightArmBotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{__name__}:agents/rl_games_sac_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
