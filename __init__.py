# right_arm_bot/tasks/__init__.py

import gymnasium as gym
##
# Register Gym environments.
##

# 1. Define the path string manually since we can't import the module directly
agents_path = f"{__name__}.agents"

gym.register(
    id="Template-Right-Arm-Bot-Direct-v0",
    entry_point=f"{__name__}.right_arm_bot_env:RightArmBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.right_arm_bot_env_cfg:RightArmBotEnvCfg",
        
        # 2. Use 'agents_path' here instead of 'agents.__name__'
        "rsl_rl_cfg_entry_point": f"{agents_path}.rsl_rl_ppo_cfg:PPORunnerCfg",
        
        "skrl_sac_cfg_entry_point": f"{agents_path}:skrl_sac_cfg.yaml",
        
        "skrl_amp_cfg_entry_point": f"{agents_path}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents_path}:skrl_ppo_cfg.yaml",
        
        # This one can stay as is if it refers to the root package, 
        # but consistency is good:
        "rl_games_cfg_entry_point": f"{agents_path}:rl_games_sac_cfg.yaml", 
    },
)