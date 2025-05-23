import argparse
import os
import pickle

import torch
from legged_env import LeggedEnv
from legged_sf_env import LeggedSfEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
import copy
import re

# def get_train_cfg(exp_name, max_iterations):

#     train_cfg_dict = {
#         "algorithm": {
#             "clip_param": 0.2,
#             "desired_kl": 0.01,
#             "entropy_coef": 0.01,
#             "gamma": 0.99,
#             "lam": 0.95,
#             "learning_rate": 0.001,
#             "max_grad_norm": 1.0,
#             "num_learning_epochs": 5,
#             "num_mini_batches": 4,
#             "schedule": "adaptive",
#             "use_clipped_value_loss": True,
#             "value_loss_coef": 1.0,
#         },
#         "init_member_classes": {},
#         "policy": {
#             "activation": "elu",
#             "actor_hidden_dims": [512, 256, 128],
#             "critic_hidden_dims": [512, 256, 128],
#             "init_noise_std": 1.0,
#         },
#         "runner": {
#             "algorithm_class_name": "PPO",
#             "checkpoint": -1,
#             "experiment_name": exp_name,
#             "load_run": -1,
#             "log_interval": 1,
#             "max_iterations": max_iterations,
#             "num_steps_per_env": 24,
#             "policy_class_name": "ActorCritic",
#             "record_interval": 50,
#             "resume": False,
#             "resume_path": None,
#             "run_name": "",
#             "runner_class_name": "runner_class_name",
#             "save_interval": 10,
#             "policy_class_name": "ActorCriticSLR",
#         },
#         "runner_class_name": "OnPolicyRunner",
#         "seed": 1,
#     }

#     return train_cfg_dict


# def get_cfgs():
#     env_cfg = {
#         "num_actions": 12,
#         "n_proprio": 45,
#         "history_len": 10,
#         "robot_mjcf": "xml/go2/go2.xml",
#         # joint/link names
#         'links_to_keep': ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',],
#         "default_joint_angles": {  # [rad]
#             "FL_hip_joint": 0.1,
#             "FR_hip_joint": -0.1,
#             "RL_hip_joint": 0.1,
#             "RR_hip_joint": -0.1,

#             "FL_thigh_joint": 0.8,
#             "FR_thigh_joint": 0.8,
#             "RL_thigh_joint": 1.0,
#             "RR_thigh_joint": 1.0,

#             "FL_calf_joint": -1.5,
#             "FR_calf_joint": -1.5,
#             "RL_calf_joint": -1.5,
#             "RR_calf_joint": -1.5,
#         },
#         "dof_names": [
#             "FL_hip_joint",
#             "FL_thigh_joint",
#             "FL_calf_joint",
#             "FR_hip_joint",
#             "FR_thigh_joint",
#             "FR_calf_joint",
#             "RL_hip_joint",
#             "RL_thigh_joint",
#             "RL_calf_joint",
#             "RR_hip_joint",
#             "RR_thigh_joint",
#             "RR_calf_joint",
#         ],
#         'PD_stiffness': {'hip':   40.0,
#                          'thigh': 40.0,
#                           'calf': 40.0},
#         'PD_damping': {'hip':    1.0,
#                         'thigh': 1.0,
#                         'calf':  1.0},
#         'force_limit': {'hip':    23.7,
#                         'thigh':  23.7,
#                         'calf':   23.7},
#         # termination
#         # 'termination_contact_link_names': ['base'],
#         'termination_contact_link_names': [],
#         # 'penalized_contact_link_names': ['base', 'thigh', 'calf'],
#         'penalized_contact_link_names': ['thigh', 'calf'],
#         'feet_link_names': ['foot'],
#         'base_link_name': ['base'], 
#         "hip_joint_names": [
#             "FL_hip_joint",
#             "FR_hip_joint",
#             "RL_hip_joint",
#             "RR_hip_joint",            
#         ],
#         "termination_if_roll_greater_than": 170,  # degree. 
#         "termination_if_pitch_greater_than": 170,
#         "termination_if_height_lower_than": -20,
#         "termination_duration": 0.002, #seconds
#         # base pose
#         "base_init_pos": [0.0, 0.0, 0.42],
#         "base_init_quat": [1.0, 0.0, 0.0, 0.0],
#         "episode_length_s": 20.0,
#         # "resampling_time_s": 4.0,
#         "resampling_time_s": 10.0,
#         "action_scale": 0.25,
#         "simulate_action_latency": True,
#         'send_timeouts': True,
#         "clip_actions": 100.0,
#         'control_freq': 50,
#         'decimation': 4,
#         # random push
#         # 'push_interval_s': 5,
#         'push_interval_s': 15,
#         'max_push_vel_xy': 1.0,
#         # domain randomization
#         'randomize_friction': True,
#         'friction_range': [0.2, 1.25],
#         'randomize_base_mass': True,
#         'added_mass_range': [-1., 2.],
#         'randomize_com_displacement': True,
#         'com_displacement_range': [-0.05, 0.05],
#         'randomize_motor_strength': True,
#         'motor_strength_range': [0.9, 1.1],
#         'randomize_motor_offset': False,
#         'motor_offset_range': [-0.02, 0.02],
#         'randomize_kp_scale': True,
#         'kp_scale_range': [0.9, 1.1],
#         'randomize_kd_scale': True,
#         'kd_scale_range': [0.9, 1.1],
#     }
#     obs_cfg = {
#         "num_obs": 53,
#         "num_privileged_obs": 56,
#         "obs_scales": {
#             "lin_vel": 2.0,
#             "ang_vel": 0.25,
#             "dof_pos": 1.0,
#             "dof_vel": 0.05,
#         },
#         "clip_observations":100,
#         # "n_proprio": 48,
#         # "history_len": 10,
#     }

#     reward_cfg = {
#         "tracking_sigma": 0.25,
#         "base_height_target": 0.28,
#         "step_period": 0.8,
#         "step_offset": 0.5,
#         "front_feet_relative_height_from_base": 0.1,
#         "front_feet_relative_height_from_world": 0.2,
#         "rear_feet_relative_height_from_base": 0.15,
#         "soft_dof_pos_limit": 0.9,
#         "soft_torque_limit": 1.0,
#         "only_positive_rewards": True,
#         "max_contact_force": 200,
#         "reward_scales": {
#             "tracking_lin_vel": 1.5,
#             "tracking_ang_vel": 0.75,
#             "lin_vel_z": -.001, #-5.0
#             # "orientation": -0, #-30.0
#             "ang_vel_xy": -0.1,
#             "collision": -5.0,
#             "action_rate": -0.01,
#             "contact_no_vel": -0.002,
#             "dof_acc": -2.5e-7,
#             "hip_pos": -1.0, #-1.0
#             "contact": 0.01,
#             "dof_pos_limits": -3.0,
#             'torques': -0.00002,
#             "termination": -30.0,
#             "feet_contact_forces": -0.1
#             # "front_feet_swing_height": -10.0, #-10.0
#             # "rear_feet_swing_height": -0.1, #-10.0
#         },
#     }
#     command_cfg = {
#         "num_commands": 3,
#         # "lin_vel_x_range": [.5, 1.0],
#         # "lin_vel_y_range": [0.0, 0.0],
#         # "ang_vel_range": [0.0, 0.0],
#         # "lin_vel_x_range": [-1.0, 1.0],
#         # "lin_vel_y_range": [-0.5, 0.5],
#         # "ang_vel_range": [-1.0, 1.0],
#         "lin_vel_x_range": [-0.5, 0.5],
#         "lin_vel_y_range": [-0.5, 0.5],
#         "ang_vel_range": [-1.0, 1.0],
#     }
#     noise_cfg = {
#         "add_noise": True,
#         "noise_level": 1.0,
#         "noise_scales":{
#             "dof_pos": 0.01,
#             "dof_vel": 1.5,
#             "lin_vel": 0.1,
#             "ang_vel": 0.2,
#             "gravity": 0.05,
#         }

#     }
#     terrain_cfg = {
#         "terrain_type": "plane",
#         "subterrain_size": 12.0,
#         "horizontal_scale": 0.25,
#         "vertical_scale": 0.005,
#         "cols": 5,  #should be more than 5
#         "rows": 5,   #should be more than 5
#         "selected_terrains":{
#             "flat_terrain" : {"probability": .5},
#             "random_uniform_terrain" : {"probability": 0.5},
#             "pyramid_sloped_terrain" : {"probability": 0.1},
#             "discrete_obstacles_terrain" : {"probability": 0.5},
#             "pyramid_stairs_terrain" : {"probability": 0.0},
#             "wave_terrain": {"probability": 0.5},

#         }
#     }

#     return env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_slosh_free_v2")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"/home/psxkf4/Genesis/logs/{args.exp_name}"
    # Get all subdirectories in the base log directory
    subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

    # Sort subdirectories by their names (assuming they are timestamped in lexicographical order)
    most_recent_subdir = sorted(subdirs)[-1] if subdirs else None
    log_dir = os.path.join(log_dir, most_recent_subdir)
    env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, train_cfg, terrain_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    # env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg = get_cfgs()
    # train_cfg = get_train_cfg("slr", "100000")
    # reward_cfg["reward_scales"] = {}

    env = LeggedSfEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        noise_cfg=noise_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        terrain_cfg=terrain_cfg,
        show_viewer=True,
    )

    

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    # List all files in the most recent subdirectory
    files = os.listdir(log_dir)

    # Regex to match filenames like 'model_100.pt' and extract the number
    model_files = [(f, int(re.search(r'model_(\d+)\.pt', f).group(1)))
                for f in files if re.search(r'model_(\d+)\.pt', f)]
    model_file = max(model_files, key=lambda x: x[1])[0]

    resume_path = os.path.join(log_dir,  model_file)
    # resume_path = "/home/koyo/ts_Genesis/examples/locomotion/unitree/logs/slr/model_40000_sdk.pt"
    runner.load(resume_path)
    # resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    # runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    # export policy as a jit module (used to run it from C++)
    EXPORT_POLICY = False
    if EXPORT_POLICY:
        path = os.path.join(log_dir, 'exported', 'policies')
        # export_policy_as_jit(runner.alg.actor_critic, path)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(runner.alg.actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)
        print('Exported policy as jit script to: ', path)
        # Convert the policy to a version-less format
        versionless_path = os.path.join(log_dir, 'exported', 'policies', "policy_safe.pt")
        loaded_model = torch.jit.load(path)
        loaded_model.eval()
        loaded_model.save(versionless_path)
        print("Model successfully converted to version-less format: ", versionless_path)
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go1_eval.py -e go1-walking -v --ckpt 100
"""
