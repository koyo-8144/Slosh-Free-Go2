import argparse
import os
import pickle
import shutil

from legged_rapid_env import LeggedRapidEnv
from rsl_rl.runners import OnPolicyRunner

# import sys
# sys.path.append("/home/psxkf4/Genesis/rsl_rl_ts")
# from rsl_rl_ts.runners import OnPolicyRunner

import genesis as gs
from datetime import datetime
import re
import wandb

def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": 50,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 2000,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # "robot_urdf": "urdf/go2/urdf/go2.urdf",
        "robot_mjcf": "xml/go2/go2.xml",
        # joint/link names
        'links_to_keep': ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',],
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,
            "RR_hip_joint": -0.1,

            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,

            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ],
        'PD_stiffness': {'hip':   20.0,
                         'thigh': 20.0,
                          'calf': 20.0},
        'PD_damping': {'hip':    0.5,
                        'thigh': 0.5,
                        'calf':  0.5},
        'force_limit': {'hip':    45.0,
                        'thigh':  45.0,
                        'calf':   45.0},
        # termination
        'termination_contact_link_names': ['base_link'],
        'penalized_contact_link_names': ['base_link', 'thigh', 'calf'],
        # 'termination_contact_link_names': ['base_link', 'hip', 'thigh', 'calf'],
        # 'penalized_contact_link_names': ['base_link', 'hip', 'thigh', 'calf'],
        'feet_link_names': ['foot'],
        'base_link_name': ['base_link'], 
        "hip_joint_names": [
            "FL_hip_joint",
            "FR_hip_joint",
            "RL_hip_joint",
            "RR_hip_joint",            
        ],
        "termination_if_roll_greater_than": 170,  # degree. 
        "termination_if_pitch_greater_than": 170,
        "termination_if_height_lower_than": 0,
        "termination_duration": 0.002, #seconds
        # base pose
        "base_init_pos": [0.0, 0.0, 0.5],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        'send_timeouts': True,
        "clip_actions": 100.0,
        'control_freq': 50,
        'decimation': 4,
        # random push
        'push_interval_s': -1,
        'max_push_vel_xy': 1.0,
        # domain randomization
        'randomize_friction': True,
        'friction_range': [0.1, 1.5], #[0.1m 1.5]
        'randomize_base_mass': True,
        'added_mass_range': [-1., 3.],
        'randomize_com_displacement': True,
        'com_displacement_range': [-0.01, 0.01],
        'randomize_motor_strength': False,
        'motor_strength_range': [0.9, 1.1],
        'randomize_motor_offset': False,
        'motor_offset_range': [-0.02, 0.02],
        'randomize_kp_scale': False,
        'kp_scale_range': [0.8, 1.2],
        'randomize_kd_scale': False,
        'kd_scale_range': [0.8, 1.2],
    }
    obs_cfg = {
        "num_obs": 42,
        "num_privileged_obs": 48, # num_obs + base_lin_vel
        # "num_obs": 54,
        # "num_privileged_obs": 57,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "ax": 1.0,
            "az": 1.0,
            "pitch_ang": 1.0
        },
        "clip_observations":100,
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.28,
        "step_period": 0.8,
        "step_offset": 0.5,
        "front_feet_relative_height_from_base": 0.1,
        "front_feet_relative_height_from_world": 0.2,
        "rear_feet_relative_height_from_base": 0.15,
        "soft_dof_pos_limit": 0.9,
        "soft_torque_limit": 1.0,
        "max_contact_force": 100,
        "only_positive_rewards": True,
        # "base_height_start_step": 100,
        "min_pitch_num": 5.0,
        "action_rate_weight_decay_rate": 0.5,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            # "base_height": -50.0,
            "action_rate": -0.01,
            "similar_to_default": -0.08,

            "contact": 0.1,
            "constrained_slosh_free": -50.0
        },
    }
    command_cfg = {
        "num_commands": 3,
        # "num_commands": 4,
        "curriculum_seed": 100,
        "lin_vel_x_range": [-1.0, 1.0], # This is list
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.5, 0.5],
        "limit_vel_x": [-10.0, 10.0],
        "limit_vel_y": [-0.6, 0.6],
        "limit_vel_yaw": [-10.0, 10.0],
        # "lin_vel_x_range_start": [-1.0, 1.0],
        # "lin_vel_x_range_goal": [-2.0, 2.0],
        "achieve_rate": 0.8,
        # "increase_rate": 0.1,

        # "lin_vel_y_range": [-0.5, 0.5],
        # "ang_vel_range": [-0.5, 0.5],
    }
    noise_cfg = {
        "add_noise": True,
        "noise_level": 1.0,
        "noise_scales":{
            "dof_pos": 0.01,
            "dof_vel": 1.5,
            "lin_vel": 0.1,
            "ang_vel": 0.2,
            "gravity": 0.05,
        }

    }
    terrain_cfg = {
        "terrain_type": "plane",
        "subterrain_size": 12.0,
        "horizontal_scale": 0.25,
        "vertical_scale": 0.005,
        "cols": 5,  #should be more than 5
        "rows": 5,   #should be more than 5
        "selected_terrains":{
            "flat_terrain" : {"probability": .5},
            "random_uniform_terrain" : {"probability": 0.5},
            "pyramid_sloped_terrain" : {"probability": 0.1},
            "discrete_obstacles_terrain" : {"probability": 0.5},
            "pyramid_stairs_terrain" : {"probability": 0.0},
            "wave_terrain": {"probability": 0.5},

        }
    }

    return env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_slosh_free_rapid")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=100000)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if this flag is set")
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--view", action="store_true", help="If you would like to see how robot is trained")
    parser.add_argument("--offline", action="store_true", help="If you do not want to upload online via wandb")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir_ = f"/home/psxkf4/Genesis/logs/{args.exp_name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir_, timestamp)
    env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    
    env = LeggedRapidEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        noise_cfg=noise_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,        
        terrain_cfg=terrain_cfg,        
        show_viewer=args.view,
    )

    if args.resume:
        # Get all subdirectories in the base log directory
        subdirs = [d for d in os.listdir(log_dir_) if os.path.isdir(os.path.join(log_dir_, d))]

        # Sort subdirectories by their names (assuming they are timestamped in lexicographical order)
        most_recent_subdir = sorted(subdirs)[-1] if subdirs else None
        most_recent_path = os.path.join(log_dir_, most_recent_subdir)

        if args.ckpt == 0:
            # List all files in the most recent subdirectory
            files = os.listdir(most_recent_path)

            # Regex to match filenames like 'model_100.pt' and extract the number
            model_files = [(f, int(re.search(r'model_(\d+)\.pt', f).group(1)))
                        for f in files if re.search(r'model_(\d+)\.pt', f)]
            model_file = max(model_files, key=lambda x: x[1])[0]
        else:
            model_file = f"model_{args.ckpt}.pt"
        # resume_path = os.path.join(most_recent_path,  model_file)
        resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free/20250213_184130/model_10000.pt"

    os.makedirs(log_dir, exist_ok=True)        
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    if args.resume:
        runner.load(resume_path)


    wandb.init(project='custom_genesis', name=args.exp_name, dir=log_dir, mode='offline' if args.offline else 'online')

    pickle.dump(
        [env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, train_cfg, terrain_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go1_train.py
"""
