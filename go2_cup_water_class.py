import argparse
import torch
import numpy as np
import genesis as gs
from genesis.utils.geom import inv_quat


class GenesisSimulation:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")

        self.env_cfg, self.obs_cfg, self.noise_cfg, self.reward_cfg, self.command_cfg, self.terrain_cfg = self.get_cfgs()
        self.train_cfg = self.get_train_cfg(args.exp_name, args.max_iterations)

        self.theta = 0
        self.theta_increment = 0.01
        self.radius = 5.0
        self.camera_height = 2.5
        self.target = np.array([0, 0, 0.5])
        self._recording = False
        self._recorded_frames = []
        self.show_vis = args.view

        gs.init()
        self.setup_scene()
        self.setup_entities()

    def get_train_cfg(self, exp_name, max_iterations):
        return {
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
                "save_interval": 2500,
            },
            "runner_class_name": "OnPolicyRunner",
            "seed": 1,
        }

    def get_cfgs(self):
        env_cfg = {
            "num_actions": 12,
            "robot_mjcf": "xml/go2/go2.xml",
            'links_to_keep': ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'],
            "default_joint_angles": {
                "FL_hip_joint": 0.1, "FR_hip_joint": -0.1,
                "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
                "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
                "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
                "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
                "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
            },
            "dof_names": [
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            ],
            'PD_stiffness': {'hip': 20.0, 'thigh': 20.0, 'calf': 20.0},
            'PD_damping': {'hip': 0.5, 'thigh': 0.5, 'calf': 0.5},
            'force_limit': {'hip': 45.0, 'thigh': 45.0, 'calf': 45.0},
            'termination_contact_link_names': ['base_link'],
            'penalized_contact_link_names': ['base_link', 'hip', 'thigh', 'calf'],
            'feet_link_names': ['foot'],
            'base_link_name': ['base_link'],
            "hip_joint_names": [
                "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            ],
            "termination_if_roll_greater_than": 170,
            "termination_if_pitch_greater_than": 170,
            "termination_if_height_lower_than": 0,
            "termination_duration": 0.002,
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
            'push_interval_s': -1,
            'max_push_vel_xy': 1.0,
            'randomize_friction': True,
            'friction_range': [0.1, 1.5],
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
            "num_obs": 54,
            "num_privileged_obs": 57,
            "obs_scales": {
                "lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0,
                "dof_vel": 0.05, "ax": 1.0, "az": 1.0, "pitch_ang": 1.0
            },
            "clip_observations": 100,
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
            "reward_scales": {
                "tracking_lin_vel": 1.0,
                "tracking_ang_vel": 0.2,
                "lin_vel_z": -1.0,
                "base_height": -50.0,
                "action_rate": -0.005,
                "similar_to_default": -0.1,
                "tracking_pitch_ang": 1.0,
            },
        }
        command_cfg = {
            "num_commands": 4,
            "lin_vel_x_range": [-1.0, 1.0],
            "lin_vel_y_range": [-0.5, 0.5],
            "ang_vel_range": [-0.5, 0.5],
            "pitch_ang_range": [-30.0, 30.0],
        }
        noise_cfg = {
            "add_noise": True,
            "noise_level": 1.0,
            "noise_scales": {
                "dof_pos": 0.01, "dof_vel": 1.5, "lin_vel": 0.1,
                "ang_vel": 0.2, "gravity": 0.05,
            }
        }
        terrain_cfg = {
            "terrain_type": "plane",
            "subterrain_size": 12.0,
            "horizontal_scale": 0.25,
            "vertical_scale": 0.005,
            "cols": 5,
            "rows": 5,
            "selected_terrains": {
                "flat_terrain": {"probability": 0.5},
                "random_uniform_terrain": {"probability": 0.5},
                "pyramid_sloped_terrain": {"probability": 0.1},
                "discrete_obstacles_terrain": {"probability": 0.5},
                "pyramid_stairs_terrain": {"probability": 0.0},
                "wave_terrain": {"probability": 0.5},
            }
        }
        return env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg

    def setup_scene(self):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
            sph_options=gs.options.SPHOptions(
                lower_bound=(-0.025, -0.025, 0.4),
                upper_bound=(0.025, 0.025, 0.6),
                particle_size=0.01,
            ),
            vis_options=gs.options.VisOptions(visualize_sph_boundary=True),
            show_viewer=True,
        )
        self.cam_0 = self.scene.add_camera(
            pos=(5.0, 0.0, 2.5),
            lookat=(0, 0, 0.5),
            fov=30,
            GUI=False,
        )

    def setup_entities(self):
        self.scene.add_entity(morph=gs.morphs.Plane())

        base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=self.env_cfg["robot_mjcf"],
                pos=base_init_pos.cpu().numpy(),
                quat=base_init_quat.cpu().numpy(),
            )
        )

        self.liquid = self.scene.add_entity(
            material=gs.materials.SPH.Liquid(),
            morph=gs.morphs.Cylinder(
                pos=(0.0, 0.0, base_init_pos[2].item() + 0.025),
                height=0.04,
                radius=0.025,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.8, 1.0),
                vis_mode="particle",
            ),
        )

    def update_camera_pose(self):
        x_pos = self.target[0] + self.radius * np.cos(self.theta)
        y_pos = self.target[1] + self.radius * np.sin(self.theta)
        eye = np.array([x_pos, y_pos, self.camera_height])
        self.cam_0.set_pose(pos=eye, lookat=self.target)
        self.theta += self.theta_increment

        if self._recording and len(self._recorded_frames) < 150:
            if self.show_vis:
                self.cam_0.render(rgb=True)
            frame, _, _, _ = self.cam_0.render()
            self._recorded_frames.append(frame)
        elif self.show_vis:
            self.cam_0.render(rgb=True)

    def run(self, horizon=1000):
        self.scene.build()
        for _ in range(horizon):
            self.scene.step()
            if self.args.view:
                self.update_camera_pose()
        return self.liquid.get_particles()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_water")
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=100000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    sim = GenesisSimulation(args)
    particles = sim.run()
