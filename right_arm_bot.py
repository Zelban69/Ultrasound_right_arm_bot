from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

# 1. Renamed to RIGHT_ARM_BOT_CFG to match your environment's import
RIGHT_ARM_BOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=r"C:\IsaacLab\MyRobot\robot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ), 

    # 3. The 'init_state' block is now INSIDE the ArticulationCfg call
    init_state=ArticulationCfg.InitialStateCfg(
        # 4. REMOVED the extra, invalid parenthesis from here
        
        # Initial position (X, Y, Z) of the robot's
        # root (base_link) in the world frame.
        pos=(0.0, 0.0, 0.1),  # Example: 0.5m above the ground
        
        # Default joint positions for all 10 joints.
        # This must be a dictionary where keys match the
        # joint names in the USD file.
        joint_pos={
            "Joint_00_02": 0.0,
            "Joint_02_03": 0.0,
            "Joint_03_04": 0.0,
            "Joint_04_05": 0.0,
            "Joint_05_06": 0.0,
            "Joint_06_07": 0.0,
            "Joint_07_08": 1.571,
            "Joint_08_09": 0.0,
            "Joint_09_10": -0.698,
            "Joint_10_11": 0.0,
        },

        # Default joint velocities for all 10 joints.
        # Using a regex is common here.
        joint_vel={".*": 0.0},
    ),  # <-- 5. This is now a comma

    # 6. The 'actuators' block is now INSIDE the ArticulationCfg call
    actuators={
        "all_joints": ImplicitActuatorCfg(
            # Use a regex to match all 10 joints.
            joint_names_expr=["Joint_.*"], 
            
            # These are the *solver-level* caps.
            # Max torque [N-m]
            effort_limit_sim=5.0,  # Use effort_limit_sim [1]
            # Max velocity [rad/s]
            velocity_limit_sim=1.0, 
            
            # These are the PD gains.
            # P-gain: Needs tuning. Higher values = stiffer.
            stiffness=40.0,
            # D-gain: Needs tuning. Higher values = more damping.
            damping=1.0,
        ),
    },
)  # <-- 7. The ArticulationCfg call properly ENDS here.