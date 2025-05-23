import torch
import math
import genesis as gs
# from genesis.utils.terrain import parse_terrain

from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import numpy as np
import random
import copy
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

import matplotlib.pyplot as plt
import os

PLOT_PITCH = 0
PLOT_ACC = 0
PLOT_ERROR = 0

PLOT_TILT_ERROR_VEL_ACC_HEIGHT_CMDREC = 0
RANDOM_RESAMPLE_EVAL = 0
ROLL_EVAL = 0

ACC_PROFILE_RESAMPLE = 0
ACC_PROFILE_RESAMPLE_V2 = 0
ACC_PROFILE_RESAMPLE_V3 = 0
POS_ACC = 0
NEG_ACC = 0
V_ACC = 1

HIP_REDUCTION = 1

TEACHER_STUDENT = 1

MIX_RESAMPLE = 0
TRAJECTORY_RESAMPLE = 0

PREDEFINED_RESAMPLE_EVAL = 0
PREDEFINED_RESAMPLE_TRY_EVAL = 0


RANDOM_RESAMPLE_TRAIN = 0

PITCH_COMMAND_TRAIN = 0
DESIRED_PITCH_COMMAND = 0

LATERAL_CAM = 1
TOP_CAM = 0
FRONT_CAM = 0
VIDEO_RECORD = 0

RANDOM_INIT_ROT = 0
DELAY = 0

ALIENWARE = 0

# Helper function to get quaternion from Euler angles
def quaternion_from_euler_tensor(roll_deg, pitch_deg, yaw_deg):
    """
    roll_deg, pitch_deg, yaw_deg: (N,) PyTorch tensors of angles in degrees.
    Returns a (N, 4) PyTorch tensor of quaternions in [x, y, z, w] format.
    """
    # Convert to radians
    roll_rad = torch.deg2rad(roll_deg)
    pitch_rad = torch.deg2rad(pitch_deg)
    yaw_rad = torch.deg2rad(yaw_deg)

    # Half angles
    half_r = roll_rad * 0.5
    half_p = pitch_rad * 0.5
    half_y = yaw_rad * 0.5

    # Precompute sines/cosines
    cr = half_r.cos()
    sr = half_r.sin()
    cp = half_p.cos()
    sp = half_p.sin()
    cy = half_y.cos()
    sy = half_y.sin()

    # Quaternion formula (XYZW)
    # Note: This is the standard euler->quat for 'xyz' rotation convention.
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    # Stack into (N,4)
    return torch.stack([qx, qy, qz, qw], dim=-1)

class LeggedSfEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg, folder_name, show_viewer=False, device="cuda"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        if TEACHER_STUDENT:
            self.num_domain_randomization = obs_cfg["num_domain_randomization"]
            self.num_obs_history = obs_cfg["num_obs_history"]
        
        self.num_privileged_obs = obs_cfg["num_privileged_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.folder_name = folder_name

        # print("WORKS!!!!!!!!!!!!!!!!!!!!!")
        # breakpoint()

        # self.num_props=env_cfg["n_proprio"],
        # self.num_hist=env_cfg["history_len"],

        # self.joint_limits = env_cfg["joint_limits"]
        self.simulate_action_latency = env_cfg["simulate_action_latency"]  # there is a 1 step latency on real robot
        self.dt = 1 / env_cfg['control_freq']
        self.t = 0.0
        sim_dt = self.dt / env_cfg['decimation']
        sim_substeps = 1

        if not TRAJECTORY_RESAMPLE:
            self.max_episode_length_s = env_cfg['episode_length_s']
            self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.noise_cfg = noise_cfg
        self.terrain_cfg = terrain_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.clip_obs = obs_cfg["clip_observations"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.noise_scales = noise_cfg["noise_scales"]
        self.selected_terrains = terrain_cfg["selected_terrains"]

        if DELAY:
            if self.env_cfg["randomize_delay"]:
                # 1️⃣ Define Delay Parameters
                self.min_delay, self.max_delay = self.env_cfg["delay_range"]  # Delay range in seconds
                self.max_delay_steps = int(self.max_delay / self.dt)  # Convert max delay to steps

                # 2️⃣ Initialize Delay Buffers
                self.action_delay_buffer = torch.zeros(
                    (self.num_envs, self.num_actions, self.max_delay_steps + 1), device=self.device
                )
                self.motor_delay_steps = torch.randint(
                    int(self.min_delay / self.dt), self.max_delay_steps + 1,
                    (self.num_envs, self.num_actions), device=self.device
                )
                print("Enabled random delay")

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=sim_dt,
                # dt=self.dt,
                substeps=sim_substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.env_cfg['decimation']),
                # max_FPS=int(1 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            # sph_options=gs.options.SPHOptions(
            #     lower_bound=(-0.025, -0.025, 0.0),
            #     upper_bound=(0.025, 0.025, 1.0),
            #     particle_size=0.01,
            # ),
            show_viewer=False,
        )
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

        self.show_vis = show_viewer
        self.selected_robot = 0
        if show_viewer:
            # self.cam_0 = self.scene.add_camera(
            #     res=(640, 480),
            #     pos=(5.0, 0.0, 2.5),
            #     lookat=(0.0, 0, 0.5),
            #     fov=30,
            #     GUI=show_viewer        
            # )
            self.cam_0 = self.scene.add_camera(
                # res=(640, 480),
                res=(1280, 960),
                pos=(0.0, 0.0, 60.0),  # Directly above
                lookat=(0.0, 0.0, 0.5),  # Robot or terrain center
                fov=20,  # Zoom in a bit for better detail
                GUI=show_viewer        
            )
        else:
            self.cam_0 = self.scene.add_camera(
                # res=(640, 480),
                pos=(5.0, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=show_viewer        
            )
        self._recording = False
        self._recorded_frames = []


        self.terrain_type = terrain_cfg["terrain_type"]
        if self.terrain_type != "plane":
            # # add plain
            subterrain_size = terrain_cfg["subterrain_size"]
            horizontal_scale = terrain_cfg["horizontal_scale"]
            vertical_scale = terrain_cfg["vertical_scale"]
            ########################## entities ##########################
            self.cols = terrain_cfg["cols"]
            self.rows = terrain_cfg["rows"]
            n_subterrains=(self.cols, self.rows)
            terrain_types = list(self.selected_terrains.keys())
            probs = [terrain["probability"] for terrain in self.selected_terrains.values()]
            total = sum(probs)
            normalized_probs = [p / total for p in probs]
            subterrain_grid, subterrain_center_z_values  = self.generate_subterrain_grid(self.rows, self.cols, terrain_types, normalized_probs)


            # Calculate the total width and height of the terrain
            total_width = (self.cols)* subterrain_size
            total_height =(self.rows)* subterrain_size

            # Calculate the center coordinates
            center_x = total_width / 2
            center_y = total_height / 2

            self.terrain  = gs.morphs.Terrain(
                pos=(-center_x,-center_y,0),
                subterrain_size=(subterrain_size, subterrain_size),
                n_subterrains=n_subterrains,
                horizontal_scale=horizontal_scale,
                vertical_scale=vertical_scale,
                subterrain_types=subterrain_grid
            )        
            # Get the terrain's origin position in world coordinates
            terrain_origin_x, terrain_origin_y, terrain_origin_z = self.terrain.pos

            self.terrain_min_x = - (total_width  / 2.0)
            self.terrain_max_x =   (total_width  / 2.0)
            self.terrain_min_y = - (total_height / 2.0)
            self.terrain_max_y =   (total_height / 2.0)
                        # Calculate the center of each subterrain in world coordinates
            self.subterrain_centers = []
            
            for row in range(self.rows):
                for col in range(self.cols):
                    subterrain_center_x = terrain_origin_x + (col + 0.5) * subterrain_size
                    subterrain_center_y = terrain_origin_y + (row + 0.5) * subterrain_size
                    subterrain_center_z = subterrain_center_z_values[row][col]
                    self.subterrain_centers.append((subterrain_center_x, subterrain_center_y, subterrain_center_z))

            # Print the centers
            self.spawn_counter = 0
            self.max_num_centers = len(self.subterrain_centers)

            self.scene.add_entity(self.terrain)
            self.random_pos = self.generate_random_positions()
        else:
            self.scene.add_entity(
                gs.morphs.Plane(),
            )
            self.random_pos = self.generate_positions()
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot  = self.scene.add_entity(
            gs.morphs.MJCF(
            file=self.env_cfg["robot_mjcf"],
            pos=self.base_init_pos.cpu().numpy(),
            quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        self.envs_origins = torch.zeros((self.num_envs, 7), device=self.device)

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        self.hip_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["hip_joint_names"]]

        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_contact_indices = find_link_indices(
            self.env_cfg['termination_contact_link_names']
        )
        self.penalised_contact_indices = find_link_indices(
            self.env_cfg['penalized_contact_link_names'] # ['base_link', 'thigh', 'calf'],
        )
        # self.penalised_contact_indices = torch.tensor(self.penalised_contact_indices, device=self.device)
        self.feet_indices = find_link_indices(
            self.env_cfg['feet_link_names']
        )
        print(f"motor dofs {self.motor_dofs}")
        print(self.feet_indices)
        # PD control
        stiffness = self.env_cfg['PD_stiffness']
        damping = self.env_cfg['PD_damping']
        force_limit = self.env_cfg['force_limit']

        self.p_gains, self.d_gains, self.force_limits = [], [], []
        for dof_name in self.env_cfg['dof_names']:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        for dof_name in self.env_cfg['dof_names']:
            for key in force_limit.keys():
                if key in dof_name:
                    self.force_limits.append(force_limit[key])
        print(self.p_gains)
        print(self.d_gains)
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)
        # Set the force range using the calculated force limits
        self.robot.set_dofs_force_range(
            lower=-np.array(self.force_limits),  # Negative lower limit
            upper=np.array(self.force_limits),   # Positive upper limit
            dofs_idx_local=self.motor_dofs
        )
        # Store link indices that trigger termination or penalty

        # self.termination_contact_indices = env_cfg.get("termination_contact_indices", [])
        # self.penalised_contact_indices = env_cfg.get("penalised_contact_indices", [])
        # Convert link names to indices
        # self.termination_contact_indices = [self.robot.get_link(name).idx_local  for name in self.env_cfg["termination_contact_names"]]
        # self.penalised_contact_indices = [self.robot.get_link(name).idx_local  for name in self.env_cfg["penalised_contact_names"]]
        # self.feet_indices = [self.robot.get_link(name).idx_local  for name in self.env_cfg["feet_names"]]
        self.feet_front_indices = self.feet_indices[:2]
        self.feet_rear_indices = self.feet_indices[2:]

        self.termination_exceed_degree_ignored = False
        self.termination_if_roll_greater_than_value = self.env_cfg["termination_if_roll_greater_than"]
        self.termination_if_pitch_greater_than_value = self.env_cfg["termination_if_pitch_greater_than"]
        if self.termination_if_roll_greater_than_value <= 1e-6 or self.termination_if_pitch_greater_than_value <= 1e-6:
            self.termination_exceed_degree_ignored = True

        for link in self.robot._links:
            print(link.name)
        
        print(f"termination_contact_indicies {self.termination_contact_indices}")
        print(f"penalised_contact_indices {self.penalised_contact_indices}")
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            if name=="termination":
                continue
            self.reward_functions[name] = getattr(self, "_reward_" + name)

        # initialize buffers
        self.init_buffers()

        self.init_camera_params()




    def init_buffers(self):
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.base_lin_vel_x = self.base_lin_vel[:, 0]
        self.base_lin_vel_y = self.base_lin_vel[:, 1]
        self.base_lin_vel_z = self.base_lin_vel[:, 2]
        self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_lin_vel_x = self.last_base_lin_vel[:, 0]
        self.last_base_lin_vel_z = self.last_base_lin_vel[:, 2]
        self.ax_scale = self.obs_scales["ax"]
        self.az_scale = self.obs_scales["az"]
        self.a_count = 0
        self.fixed_desired_pitch = 0
        self.base_lin_vel_x_low_freq = self.base_lin_vel[:, 0]
        self.base_lin_vel_y_low_freq = self.base_lin_vel[:, 1]
        self.base_lin_vel_z_low_freq = self.base_lin_vel[:, 2]
        self.min_pitch_num = self.reward_cfg["min_pitch_num"]
        # print("lin_vel_x_range_start: ", self.command_cfg["lin_vel_x_range_start"])
        # print("type: ", type(self.command_cfg["lin_vel_x_range_start"]))
        # print("0th element: ", self.command_cfg["lin_vel_x_range_start"][0])
        # print("1st element: ", self.command_cfg["lin_vel_x_range_start"][1])
        # self.tracking_lin_vel_rew = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        # self.tracking_lin_vel_rew_mean = torch.tensor(0.0, device=self.device, dtype=gs.tc_float)
        # self.tracking_lin_vel_rew_max = torch.full((self.num_envs,), self.reward_scales["tracking_lin_vel"], device=self.device)
        # self.tracking_lin_vel_rew_max_one = self.tracking_lin_vel_rew_max [0]
        # self.tracking_lin_vel_rew_threshold = self.tracking_lin_vel_rew_max * self.command_cfg["achieve_rate"]
        # self.tracking_lin_vel_rew_threshold_one = self.tracking_lin_vel_rew_threshold[0]
        # self.tracking_lin_vel_rew_threshold_one_ori  = self.tracking_lin_vel_rew_threshold_one
        
        # self.tracking_ang_vel_rew = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        # self.tracking_ang_vel_rew_mean = torch.tensor(0.0, device=self.device, dtype=gs.tc_float)
        # self.tracking_ang_vel_rew_max = torch.full((self.num_envs,), self.reward_scales["tracking_ang_vel"], device=self.device)
        # self.tracking_ang_vel_rew_max_one = self.tracking_ang_vel_rew_max [0]
        # self.tracking_ang_vel_rew_threshold = self.tracking_ang_vel_rew_max * self.command_cfg["achieve_rate"]
        # self.tracking_ang_vel_rew_threshold_one = self.tracking_ang_vel_rew_threshold[0]
        # self.tracking_ang_vel_rew_threshold_one_ori  = self.tracking_ang_vel_rew_threshold_one
        # self.lin_vel_x_range_min = self.command_cfg["lin_vel_x_range_start"][0]
        # self.lin_vel_x_range_max = self.command_cfg["lin_vel_x_range_start"][1]
        # self.updated_lin_vel_x_command_range = self.command_cfg["lin_vel_x_range_start"]
        # self.desired_pitch_mean = 0
        # self.first_resample_done = False
        # self.first_resample_count = 0
        # self.resample_count = 0
        # self.range_update_count = 1
        # self.pitch_count = 0
        # self.resample_updated = False
        # self.action_rate_scale = self.reward_scales["action_rate"]
        # self.linvel_update_freq = self.reward_cfg["linvel_update_freq"] # 10 Hz
        # self.linvel_update_actual_freq = (1 / self.dt) / self.linvel_update_freq
        # print("self.linvel_update_actual_freq: ", self.linvel_update_actual_freq) # %hz
        # breakpoint()
        # Suppose you keep these as class attributes:
        self.ax_filtered = torch.tensor(0.0, device=self.device)
        self.ay_filtered = torch.tensor(0.0, device=self.device)
        self.az_filtered = torch.tensor(-9.8, device=self.device)  # assuming near gravity at start
        self.az_net_filtered = 0.0
        self.desired_ax_filtered = 0.0
        # alpha controls how much new data matters vs. old data (0 < alpha < 1)
        self.alpha = self.reward_cfg["alpha"]
        self.smoothed_ax_mean = 0.0
        self.smoothed_ay_mean = 0.0
        self.smoothed_az_mean = 0.0
        self.smoothed_desired_ax_mean = 0.0
        self.last_vx_plane = torch.tensor(0.0, device=self.device)
        self.last_vy_plane = torch.tensor(0.0, device=self.device)
        self.last_vz_world = torch.tensor(0.0, device=self.device)
        self.vx_plane = torch.tensor(0.0, device=self.device)
        self.vy_plane = torch.tensor(0.0, device=self.device)
        self.vz_world = torch.tensor(0.0, device=self.device)
        if ACC_PROFILE_RESAMPLE or ACC_PROFILE_RESAMPLE_V2 or ACC_PROFILE_RESAMPLE_V3 or MIX_RESAMPLE:
            self.acc_mean = self.command_cfg["acc_mean"]
            self.acc_sigma = self.command_cfg["acc_sigma"]
            self.sign_flip_rate = self.command_cfg["sign_flip_rate"]
            # self.commanded_lin_vel_x_walking = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            # self.ax_sampled = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            self.smoothed_ax = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            self.forward_count = 0
            self.backward_count = 0

            self.switch_resample = False

            self.acc_dir = torch.full((self.num_envs,), 2.0, device=self.device, dtype=gs.tc_float)
            # self.acc_dir = torch.full((self.num_envs,), 25.0, device=self.device, dtype=gs.tc_float)

            self.acc_x_mean = self.command_cfg["acc_x_mean"]
            self.acc_x_sigma = self.command_cfg["acc_x_sigma"]
            self.acc_y_mean = self.command_cfg["acc_y_mean"]
            self.acc_y_sigma = self.command_cfg["acc_y_sigma"]
            self.acc_z_mean = self.command_cfg["acc_z_mean"]
            self.acc_z_sigma = self.command_cfg["acc_z_sigma"]
            self.acc_x_dir = torch.full((self.num_envs,), 2.0, device=self.device, dtype=gs.tc_float)
            self.acc_y_dir = torch.full((self.num_envs,), 1.0, device=self.device, dtype=gs.tc_float)
            self.acc_z_dir = torch.full((self.num_envs,), 1.0, device=self.device, dtype=gs.tc_float)

        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.zero_obs = torch.zeros(self.num_obs, device=self.device, dtype=gs.tc_float)
        self.zero_privileged_obs = torch.zeros(self.num_privileged_obs, device=self.device, dtype=gs.tc_float)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        
        
        if TRAJECTORY_RESAMPLE:
            max_episode_length_s = self.env_cfg['episode_length_s']
            self.max_episode_length_s = torch.full(
                (self.num_envs,), 
                fill_value=max_episode_length_s, 
                device=self.device, 
                dtype=gs.tc_float
            ) # torch.Size([4096])
            self.max_episode_length = torch.full(
                (self.num_envs,), 
                fill_value=np.ceil(max_episode_length_s / self.dt), 
                device=self.device, 
                dtype=gs.tc_float
            ) # torch.Size([4096])
            # self.max_episode_length = np.ceil(max_episode_length_s / self.dt)
            self.init_random_ep_len_max = np.ceil(max_episode_length_s / self.dt)

            self.traj_t = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
            self.x0 = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
            self.xf = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

            self.env0_pos_x = torch.tensor(0., device=self.device)
            self.env0_pos_y = torch.tensor(0., device=self.device)
            self.env0_pos_z = torch.tensor(0., device=self.device)
            self.env0_acc_x = torch.tensor(0., device=self.device)
            self.env0_acc_y = torch.tensor(0., device=self.device)
            self.env0_acc_z = torch.tensor(0., device=self.device)
            self.env0_jerk_x = torch.tensor(0., device=self.device)
            self.env0_jerk_y = torch.tensor(0., device=self.device)
            self.env0_jerk_z = torch.tensor(0., device=self.device)

            self.env0_x0_x = self.x0[0, 0]
            self.env0_x0_y = self.x0[0, 1]
            self.env0_x0_z = self.x0[0, 2]
            self.env0_xf_x = self.xf[0, 0]
            self.env0_xf_y = self.xf[0, 1]
            self.env0_xf_z = self.xf[0, 2]
            # print("self.env0_x0_x: ", self.env0_x0_x)
            # breakpoint()

            self.plot_save_len = 1000

        self.time_out_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.out_of_bounds_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        
        if PITCH_COMMAND_TRAIN or DESIRED_PITCH_COMMAND:
            self.commands_scale = torch.tensor(
                [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], self.obs_scales["pitch_ang"]],
                device=self.device,
                dtype=gs.tc_float,
            )
        else:
            self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.hip_actions = torch.zeros((self.num_envs, len(self.hip_dofs)), device=self.device, dtype=gs.tc_float)

        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.feet_max_height = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.last_contacts = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=gs.tc_int,
        )
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.last_actions = torch.zeros_like(self.actions)
        self.second_last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.hip_pos = torch.zeros_like(self.hip_actions)
        self.hip_vel = torch.zeros_like(self.hip_actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        self.rot_y_deg = self.base_euler[:, 1]
        self.rot_y_deg = self.rot_y_deg.unsqueeze(-1)  # Shape: [4096, 1]
        self.last_rot_y_deg = self.rot_y_deg
        self.last_desired_pitch_deg = torch.zeros((self.num_envs, 1), device=self.device)
        self.rot_y = torch.deg2rad(self.rot_y_deg)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        # print("self.default_dof_pos: ", self.default_dof_pos)
        # breakpoint()
        self.num_dof = len(self.default_dof_pos )
        self.default_hip_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["dof_names"]
                if name in self.env_cfg["hip_joint_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.contact_duration_buf = torch.zeros(
            self.num_envs, 
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # Iterate over the motor DOFs
        # self.soft_dof_vel_limit = self.env_cfg["soft_dof_vel_limit"]
        self.soft_torque_limit = self.reward_cfg["soft_torque_limit"]
        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)

        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * self.reward_cfg['soft_dof_pos_limit']
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * self.reward_cfg['soft_dof_pos_limit']
            )
        self.motor_strengths = gs.ones((self.num_envs, self.num_dof), dtype=float)
        self.motor_offsets = gs.zeros((self.num_envs, self.num_dof), dtype=float)

        self.init_foot()
        self._randomize_controls()
        self._randomize_rigids()
        # breakpoint()
        print(f"Dof_pos_limits{self.dof_pos_limits}")
        print(f"Default dof pos {self.default_dof_pos}")
        print(f"Default hip pos {self.default_hip_pos}")
        self.common_step_counter = 0
        # extras
        self.continuous_push = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int, 
        )
        self.extras = dict()  # extra information for logging

        # # Initialize episode step counter (one per environment)
        # self.episode_step = torch.zeros(
        #     (self.num_envs,), device=self.device, dtype=torch.int32
        # )

        if TEACHER_STUDENT:
            self.his_len = self.obs_cfg["history_lengh"]
            self.base_ang_vel_his = torch.zeros(self.num_envs, self.his_len*3)
            self.dof_pos_his = torch.zeros(self.num_envs, self.his_len*self.num_dof)
            self.dof_pos_delta_his = torch.zeros(self.num_envs, self.his_len*self.num_dof)
            self.dof_vel_his = torch.zeros(self.num_envs, self.his_len*self.num_dof)
            self.projected_gravity_his = torch.zeros(self.num_envs, self.his_len*3)
            self.cmd_his = torch.zeros(self.num_envs, self.his_len*3)
            self.action_his = torch.zeros(self.num_envs, self.his_len*self.num_actions)

            self.base_ang_vel_his = self.base_ang_vel_his.to(self.device)
            self.base_ang_vel_his = self.base_ang_vel_his.to(self.device)
            self.dof_pos_his = self.dof_pos_his.to(self.device)
            self.dof_pos_delta_his = self.dof_pos_delta_his.to(self.device)
            self.dof_vel_his = self.dof_vel_his.to(self.device)
            self.projected_gravity_his = self.projected_gravity_his.to(self.device)
            self.cmd_his = self.cmd_his.to(self.device)
            self.action_his = self.action_his.to(self.device)

            self.domain_randomizations_buf = torch.zeros((self.num_envs, self.num_domain_randomization), device=self.device, dtype=gs.tc_float)
            self.obs_history_buf = torch.zeros((self.num_envs, self.num_obs_history), device=self.device, dtype=gs.tc_float)

        self.env0_command_x = torch.tensor(0., device=self.device)
        self.env0_command_y = torch.tensor(0., device=self.device)
        self.env0_command_z = torch.tensor(0., device=self.device)
        self.env1_command_x = torch.tensor(0., device=self.device)
        self.env1_command_y = torch.tensor(0., device=self.device)
        self.env1_command_z = torch.tensor(0., device=self.device)
        self.env2_command_x = torch.tensor(0., device=self.device)
        self.env2_command_y = torch.tensor(0., device=self.device)
        self.env2_command_z = torch.tensor(0., device=self.device)


        if PLOT_PITCH:
            # self.base_lin_vel_x_list = []
            # self.base_lin_vel_z_list = []
            # self.last_base_lin_vel_x_list = []
            # self.last_base_lin_vel_z_list = []
            # self.ax_list = []
            # self.az_list = []
            self.desired_pitch_list = []
            self.current_pitch_list = []
            self.time_steps = []  # Track time steps
            self.max_points = 500  # Limit number of points to store for performance

            # Initialize the plot
            plt.ion()
            # self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 10))
            # self.fig, self.axs = plt.subplots(2, 1, figsize=(5, 5))
            self.fig, self.axs = plt.subplots(1, 1, figsize=(4, 4))


            # self.axs[0].set_title("Current Base Linear X Velocities")
            # self.axs[0].set_ylabel("Velocity")
            # self.axs[0].legend(["Base Lin Vel X"])

            # self.axs[1].set_title("Last Base Linear X Velocities")
            # self.axs[1].set_ylabel("Velocity")
            # self.axs[1].legend(["Last Base Lin Vel X"])
            
            # self.axs[0].set_title("Base Linear X Velocities")
            # self.axs[0].set_ylabel("Velocity")
            # self.axs[0].legend(["Base Lin Vel X", "Last Base Lin Vel X"])

            # self.axs[1].set_title("Base Linear Z Velocities")
            # self.axs[1].set_ylabel("Velocity")
            # self.axs[1].legend(["Base Lin Vel Z", "Last Base Lin Vel Z"])

            # self.axs[2].set_title("Accelerations")
            # self.axs[2].set_xlabel("Time Steps")
            # self.axs[2].set_ylabel("Acceleration")
            # self.axs[2].legend(["Ax", "Az"])

            # self.axs[1].set_title("Accelerations")
            # self.axs[1].set_xlabel("Time Steps")
            # self.axs[1].set_ylabel("Acceleration")
            # self.axs[1].legend(["Ax"])

            self.axs.set_title("Pitch")
            self.axs.set_xlabel("Time Steps")
            self.axs.set_ylabel("Pitch")
            self.axs.legend(["Desired theta", "Current Pitch"])



            # # Create line objects to update instead of replotting
            # self.line_base_lin_x, = self.axs[0].plot([], [], label="Base Lin Vel X", color="b")
            # self.axs[0].set_ylabel("Velocity")
            # self.axs[0].legend()
            # self.axs[0].set_title("Current Base Linear X Velocities")

            # self.line_last_base_lin_x, = self.axs[1].plot([], [], label="Last Base Lin Vel X", color="r")
            # self.axs[1].set_ylabel("Velocity")
            # self.axs[1].legend()
            # self.axs[1].set_title("Last Base Linear X Velocities")

            # self.line_base_lin_z, = self.axs[1].plot([], [], label="Base Lin Vel Z", color="b")
            # self.line_last_base_lin_z, = self.axs[1].plot([], [], label="Last Base Lin Vel Z", color="r")
            # self.axs[1].set_ylabel("Velocity")
            # self.axs[1].legend()
            # self.axs[1].set_title("Base Linear Z Velocities")

            # self.line_ax, = self.axs[2].plot([], [], label="Ax", color="g")
            # self.line_az, = self.axs[2].plot([], [], label="Az", color="m")
            # self.axs[2].set_ylabel("Acceleration")
            # self.axs[2].legend()
            # self.axs[2].set_title("Accelerations")

            # self.line_desired_pitch, = self.axs[2].plot([], [], label="Desired", color="g")
            # self.line_current_pitch, = self.axs[2].plot([], [], label="Current", color="m")
            # self.axs[2].set_xlabel("Time Steps")
            # self.axs[2].set_ylabel("Pitch")
            # self.axs[2].legend()
            # self.axs[2].set_title("Pitch")

        if PLOT_ACC:
            self.ax_list = []
            self.az_list = []
            self.time_steps = []  # Track time steps
            self.max_points = 500  # Limit number of points to store for performance

            # Initialize the plot
            plt.ion()
            self.fig, self.axs = plt.subplots(1, 1, figsize=(4, 4))

            self.axs.set_title("Accelerations")
            self.axs.set_xlabel("Time Steps")
            self.axs.set_ylabel("Acceleration")
            self.axs.legend(["Ax", "Az"])

        if PLOT_ERROR:
            self.error_pitch_list = []
            self.time_steps = []  # Track time steps
            self.max_points = 500  # Limit number of points to store for performance

            # Initialize the plot
            plt.ion()
            self.fig, self.axs = plt.subplots(1, 1, figsize=(4, 4))

            self.axs.set_title("Pitch Error")
            self.axs.set_xlabel("Time Steps")
            self.axs.set_ylabel("Error")
            self.axs.legend(["Error"])
        
        if PLOT_TILT_ERROR_VEL_ACC_HEIGHT_CMDREC:
            if not RANDOM_RESAMPLE_EVAL:
                self.plot_save_len = 1000
            self.desired_pitch_list = []
            self.current_pitch_list = []
            self.error_pitch_list = []
            self.desired_roll_list = []
            self.current_roll_list = []
            self.error_roll_list = []
            self.lin_vel_x_list = []
            self.last_lin_vel_x_list = []
            self.ax_list = []
            self.az_list = []
            self.heigh_list = []
            self.command_linvel_x_list = []
            self.current_linvel_x_list = []
            self.linvel_x_error_list = []
            self.command_linvel_y_list = []
            self.current_linvel_y_list = []
            self.linvel_y_error_list = []
            self.command_angvel_z_list = []
            self.current_angvel_z_list = []
            self.angvel_z_error_list = []
            self.time_steps = []  # Track time steps
            self.max_points = 1000  # Limit number of points to store for performance

            # Initialize the plot
            plt.ion()
            self.fig1, self.axs1 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs1.set_title("Desired and Current Pitch")
            self.axs1.set_xlabel("Time Steps")
            self.axs1.set_ylabel("Pitch [degrees]")
            self.axs1.legend(["Desired Pitch", "Current Pitch"])

            plt.ion()
            self.fig2, self.axs2 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs2.set_title("Pitch Error")
            self.axs2.set_xlabel("Time Steps")
            self.axs2.set_ylabel("Pitch Error [degrees]")
            self.axs2.legend(["Pitch Error"])

            plt.ion()
            self.fig3, self.axs3 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs3.set_title("Lin Vel Current and Last")
            self.axs3.set_xlabel("Time Steps")
            self.axs3.set_ylabel("Lin Vel")
            self.axs3.legend(["Current", "Last"])

            plt.ion()
            self.fig4, self.axs4 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs4.set_title("Acc x and z")
            self.axs4.set_xlabel("Time Steps")
            self.axs4.set_ylabel("Acc")
            self.axs4.legend(["Acc x", "Acc z"])

            plt.ion()
            self.fig5, self.axs5 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs5.set_title("Base Height")
            self.axs5.set_xlabel("Time Steps")
            self.axs5.set_ylabel("Height")
            self.axs5.legend(["Height"])



            plt.ion()
            self.fig6, self.axs6 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs6.set_title("Command and Current Linear X Velocity")
            self.axs6.set_xlabel("Time Steps")
            self.axs6.set_ylabel("Linear X Velocity [m/s]")
            self.axs6.legend(["Command Linear X Velocity", "Current Linear X Velocity"])

            plt.ion()
            self.fig7, self.axs7 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs7.set_title("Linear X Velocity Error")
            self.axs7.set_xlabel("Time Steps")
            self.axs7.set_ylabel("Linear X Velocity Error [m/s]")
            self.axs7.legend(["Linear X Velocity Error"])

            plt.ion()
            self.fig8, self.axs8 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs8.set_title("Command and Current Linear Y Velocity")
            self.axs8.set_xlabel("Time Steps")
            self.axs8.set_ylabel("Linear Y Velocity [m/s]")
            self.axs8.legend(["Command Linear Y Velocity", "Current Linear Y Velocity"])

            plt.ion()
            self.fig9, self.axs9 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs9.set_title("Linear Y Velocity Error")
            self.axs9.set_xlabel("Time Steps")
            self.axs9.set_ylabel("Linear Y Velocity Error [m/s]")
            self.axs9.legend(["Linear Y Velocity Error"])

            plt.ion()
            self.fig10, self.axs10 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs10.set_title("Command and Current Angular Z Velocity")
            self.axs10.set_xlabel("Time Steps")
            self.axs10.set_ylabel("Angular Z Velocity [m/s]")
            self.axs10.legend(["Command Angular Z Velocity", "Current Angular Z Velocity"])

            plt.ion()
            self.fig11, self.axs11 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs11.set_title("Angular Z Velocity Error")
            self.axs11.set_xlabel("Time Steps")
            self.axs11.set_ylabel("Angular Z Velocity Error [m/s]")
            self.axs11.legend(["Angular Z Velocity Error"])

            plt.ion()
            self.fig12, self.axs12 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs12.set_title("Desired and Current Roll")
            self.axs12.set_xlabel("Time Steps")
            self.axs12.set_ylabel("Roll [degrees]")
            self.axs12.legend(["Desired Roll", "Current Roll"])

            plt.ion()
            self.fig13, self.axs13 = plt.subplots(1, 1, figsize=(4, 4))

            self.axs13.set_title("Roll Error")
            self.axs13.set_xlabel("Time Steps")
            self.axs13.set_ylabel("Roll Error [degrees]")
            self.axs13.legend(["Roll Error"])


        if PREDEFINED_RESAMPLE_EVAL:
            self.forward_start_len = 15
            self.forward_12_len = 50
            self.forward_8_len = 50
            self.forward_4_len = 0
            self.forward_0_len = 50
            self.backward_12_len = 0
            self.backward_8_len = 0
            self.backward_4_len = 0
            self.backward_0_len = 0
            self.plot_save_len = (self.forward_start_len + self.forward_12_len + self.forward_8_len + self.forward_4_len + self.forward_0_len 
                                    + self.backward_12_len + self.backward_8_len + self.backward_4_len + self.backward_0_len)

            self.forward_start_count = 0
            self.forward_12_count = 0
            self.forward_8_count = 0
            self.forward_4_count = 0
            self.forward_0_count = 0
            self.backward_12_count = 0
            self.backward_8_count = 0
            self.backward_4_count = 0
            self.backward_0_count = 0

            self.forward_start = True
            self.forward_12 = True
            self.forward_8 = True
            self.forward_4 = True
            self.forward_0 = True
            self.backward_12 = True
            self.backward_8 = True
            self.backward_4 = True
            self.backward_0 = True

            # print("START VEDEO RECORDING")
            # self.start_recording()
        
        if PREDEFINED_RESAMPLE_TRY_EVAL:
            self.forward_start_len = 50
            self.forward_12_len = 50
            self.forward_8_len = 50
            self.forward_4_len = 0
            self.forward_0_len = 50
            self.backward_12_len = 50
            self.backward_8_len = 30
            self.backward_4_len = 0
            self.backward_0_len = 30
            self.plot_save_len = (self.forward_start_len + self.forward_12_len + self.forward_8_len + self.forward_4_len + self.forward_0_len 
                                    + self.backward_12_len + self.backward_8_len + self.backward_4_len + self.backward_0_len)

            self.forward_start_count = 0
            self.forward_12_count = 0
            self.forward_8_count = 0
            self.forward_4_count = 0
            self.forward_0_count = 0
            self.backward_12_count = 0
            self.backward_8_count = 0
            self.backward_4_count = 0
            self.backward_0_count = 0

            self.forward_start = True
            self.forward_12 = True
            self.forward_8 = True
            self.forward_4 = True
            self.forward_0 = True
            self.backward_12 = True
            self.backward_8 = True
            self.backward_4 = True
            self.backward_0 = True
        
        if RANDOM_RESAMPLE_EVAL or RANDOM_RESAMPLE_TRAIN:
            # self.run_num = 10
            self.run_num = "TEST"
            self.test_plot = False

            self.duplicate = False
            self.record_video = False

            if ALIENWARE:
                if self.duplicate:
                    duplicated_cmd_record_path = f"/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/cmd_rec/slosh_free_no_acc_profile_MLP_run{self.run_num}_cmd_rec.txt"
                    # duplicated_cmd_record_path = f"/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/cmd_rec/no_slosh_free_MLP_run{self.run_num}_cmd_rec.txt"
            else:
                if self.duplicate:
                    duplicated_cmd_record_path = f"/home/psxkf4/Genesis/logs/paper/data/cmd_rec/20250423_044137_run{self.run_num}_cmd_rec.txt"
                    # duplicated_cmd_record_path = f"/home/psxkf4/Genesis/logs/paper/data/cmd_rec/no_slosh_free_MLP_run{self.run_num}_cmd_rec.txt"


            self.start_stop_cmd = True
            self.forward1_cmd = True
            self.forward2_cmd = True
            self.forward3_cmd = True
            self.middle_stop_cmd = True
            self.backward1_cmd = True
            self.backward2_cmd = True
            self.backward3_cmd = True
            self.finish_stop_cmd = True

            self.start_stop_cmd_count = 0
            self.forward1_cmd_count = 0
            self.forward2_cmd_count = 0
            self.forward3_cmd_count = 0
            self.middle_stop_cmd_count = 0
            self.backward1_cmd_count = 0
            self.backward2_cmd_count = 0
            self.backward3_cmd_count = 0
            self.finish_stop_cmd_count = 0

            if not self.duplicate:
                self.start_stop_cmd_len = 50
                self.forward1_cmd_len = random.randint(50, 100)
                self.forward2_cmd_len = random.randint(50, 100)
                self.forward3_cmd_len = random.randint(50, 100)
                self.middle_stop_cmd_len = 50
                self.backward1_cmd_len = random.randint(50, 100)
                self.backward2_cmd_len = random.randint(50, 100)
                self.backward3_cmd_len = random.randint(50, 100)
                self.finish_stop_cmd_len = 50

                if RANDOM_RESAMPLE_TRAIN:
                    self.forward_cmd_speeds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                    self.backward_cmd_speeds = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1]
                # self.forward_cmd_speeds = [0.0]
                # self.backward_cmd_speeds = [-0.0]
                self.forward_cmd_speeds = [1.2, 0.8, 0.4]
                self.backward_cmd_speeds = [-1.2, -0.8, -0.4]
                if ROLL_EVAL:
                    self.lateral_cmd_speeds = [1.2, 0.8, 0.4, -1.2, -0.8, -0.4]
                # self.forward_cmd_speeds = [1.4]
                # self.backward_cmd_speeds = [1.4]
                # self.forward_cmd_speeds = [1.6, 1.2, 0.8, 0.4]
                # self.backward_cmd_speeds = [-1.6, -1.2, -0.8, -0.4]
                # self.forward_cmd_speeds = [1.2, 0.8, 0.4]
                # self.backward_cmd_speeds = [-1.2, -0.8, -0.4]
                # self.forward_cmd_speeds = [1.2, 0.8, 0.4]
                # self.backward_cmd_speeds = [-1.2, -0.8, -0.4]
            else:
                self.load_command_lengths_and_velocities(duplicated_cmd_record_path)

            self.plot_save_len = (self.start_stop_cmd_len + self.forward1_cmd_len + self.forward2_cmd_len + self.forward3_cmd_len 
                                    + self.middle_stop_cmd_len + self.backward1_cmd_len + self.backward2_cmd_len + self.backward3_cmd_len + self.finish_stop_cmd_len)

            if self.record_video:
                self.video_record_count = 0
                self.video_record_len = self.plot_save_len
                print("START VEDEO RECORDING")
                self.start_recording()


        self.max_accel_norm = torch.tensor(1.0, device=self.device)
        self.max_pitch_error = torch.tensor(0.1, device=self.device)  # start >0 to avoid instability
        self.mean_pitch_error_normalized = torch.tensor(0.0, device=self.device)
        self.mean_roll_error_normalized = torch.tensor(0.0, device=self.device)
        self.mean_accel_norm_normalized = torch.tensor(0.0, device=self.device)


    def load_command_lengths_and_velocities(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if "Start Stop Length" in line:
                self.start_stop_cmd_len = int(line.split(":")[1].strip())
            elif "Forward 1 Length" in line:
                self.forward1_cmd_len = int(line.split(":")[1].strip())
            elif "Forward 1 Velocity" in line:
                self.forward1_cmd_vel = float(line.split(":")[1].strip())
            elif "Forward 2 Length" in line:
                self.forward2_cmd_len = int(line.split(":")[1].strip())
            elif "Forward 2 Velocity" in line:
                self.forward2_cmd_vel = float(line.split(":")[1].strip())
            elif "Forward 3 Length" in line:
                self.forward3_cmd_len = int(line.split(":")[1].strip())
            elif "Forward 3 Velocity" in line:
                self.forward3_cmd_vel = float(line.split(":")[1].strip())
            elif "Middle Stop Length" in line:
                self.middle_stop_cmd_len = int(line.split(":")[1].strip())
            elif "Backward 1 Length" in line:
                self.backward1_cmd_len = int(line.split(":")[1].strip())
            elif "Backward 1 Velocity" in line:
                self.backward1_cmd_vel = float(line.split(":")[1].strip())
            elif "Backward 2 Length" in line:
                self.backward2_cmd_len = int(line.split(":")[1].strip())
            elif "Backward 2 Velocity" in line:
                self.backward2_cmd_vel = float(line.split(":")[1].strip())
            elif "Backward 3 Length" in line:
                self.backward3_cmd_len = int(line.split(":")[1].strip())
            elif "Backward 3 Velocity" in line:
                self.backward3_cmd_vel = float(line.split(":")[1].strip())
            elif "Finish Stop Length" in line:
                self.finish_stop_cmd_len = int(line.split(":")[1].split(",")[0].strip())

        print("=== Command Lengths ===")
        print(f"Start Stop Length: {self.start_stop_cmd_len}")
        print(f"Forward 1 Length: {self.forward1_cmd_len}")
        print(f"Forward 2 Length: {self.forward2_cmd_len}")
        print(f"Forward 3 Length: {self.forward3_cmd_len}")
        print(f"Middle Stop Length: {self.middle_stop_cmd_len}")
        print(f"Backward 1 Length: {self.backward1_cmd_len}")
        print(f"Backward 2 Length: {self.backward2_cmd_len}")
        print(f"Backward 3 Length: {self.backward3_cmd_len}")
        print(f"Finish Stop Length: {self.finish_stop_cmd_len}")

        print("\n=== Command Velocities ===")
        print(f"Forward 1 Velocity: {self.forward1_cmd_vel}")
        print(f"Forward 2 Velocity: {self.forward2_cmd_vel}")
        print(f"Forward 3 Velocity: {self.forward3_cmd_vel}")
        print(f"Backward 1 Velocity: {self.backward1_cmd_vel}")
        print(f"Backward 2 Velocity: {self.backward2_cmd_vel}")
        print(f"Backward 3 Velocity: {self.backward3_cmd_vel}")

        # breakpoint()

    def init_camera_params(self):
        self.whole_view = False

        # Initialize rotation parameters
        self.radius = 30.0  # Radius of circular path
        self.theta = 0.0  # Initial angle
        self.theta_increment = np.radians(2)  # Increment angle by 2 degrees

        # Fixed target (lookat) position
        self.target = np.array([0.0, 0.0, 0.5])  # Assume robot is at this position
        self.camera_height = 0.0  # Fixed z-axis position for top-down view
    

    def append_limited(self, lst, value):
        """ Keep list size under self.max_points to avoid performance issues. """
        if len(lst) >= self.max_points:
            lst.pop(0)
        lst.append(value)

    
    def _resample_commands_pitch(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_z_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 3] = gs_rand_float(*self.command_cfg["pitch_ang_range"], (len(envs_idx),), self.device)
        # print("Sampled Linear Velocity x: ", self.commands[envs_idx, 0])
        # print("Sampled Pitch Angle Command: ", self.commands[envs_idx, 3])

    def _resample_linvel_x_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_z_range"], (len(envs_idx),), self.device)
        # self.commands[envs_idx, 0] = 0.0
        # self.commands[envs_idx, 1] = 0.0
        # self.commands[envs_idx, 2] = 0.0

        self.env0_command_x = self.commands[0, 0]
        self.env0_command_y = self.commands[0, 1]
        self.env0_command_z = self.commands[0, 2]
        self.env1_command_x = self.commands[1, 0]
        self.env1_command_y = self.commands[1, 1]
        self.env1_command_z = self.commands[1, 2]
        self.env2_command_x = self.commands[2, 0]
        self.env2_command_y = self.commands[2, 1]
        self.env2_command_z = self.commands[2, 2]

        if PITCH_COMMAND_TRAIN:
            self.commands[envs_idx, 3] = gs_rand_float(*self.command_cfg["pitch_ang_range"], (len(envs_idx),), self.device)

    

    def _resample_desired_pitch(self, envs_idx):
        ax = (self.vx_plane - self.last_vx_plane) / self.dt
        az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        # 2. Exponential smoothing
        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        
        # 3. Use the filtered values
        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale

        desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        desired_pitch_degrees = torch.rad2deg(desired_pitch)
        desired_pitch_degrees = -desired_pitch_degrees

        self.commands[envs_idx, 3] = desired_pitch_degrees

        # self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        # self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        # self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_z_range"], (len(envs_idx),), self.device)

    def _resample_commands_gaussian_acc_v3(self, envs_idx, reset_flag):
        # if envs_idx.numel() != 0:
        #     print(f"Resample velocity for {envs_idx} at step {self.episode_length_buf}")
        #     breakpoint()
        old_lin_vel_x_command = self.commands[envs_idx, 0]
        old_lin_vel_y_command = self.commands[envs_idx, 1]
        old_ang_vel_z_command = self.commands[envs_idx, 2]

        dt = self.dt
        # sampled_ax = torch.randn((len(envs_idx),), device=self.device) * self.acc_sigma
        sampled_ax = torch.normal(mean=self.acc_x_mean, std=self.acc_x_sigma, size=(len(envs_idx),), device=self.device)
        sampled_ay = torch.normal(mean=self.acc_y_mean, std=self.acc_y_sigma, size=(len(envs_idx),), device=self.device)
        sampled_az = torch.normal(mean=self.acc_z_mean, std=self.acc_z_sigma, size=(len(envs_idx),), device=self.device)
        # sampled_ax /= 2

        if V_ACC:
            self.commands[envs_idx, 0] += self.acc_x_dir[envs_idx] * dt
            self.commands[envs_idx, 1] += self.acc_y_dir[envs_idx] * dt
            self.commands[envs_idx, 2] += self.acc_z_dir[envs_idx] * dt
        else:
            self.commands[envs_idx, 0] += sampled_ax * dt
            self.commands[envs_idx, 1] += sampled_ay * dt
            self.commands[envs_idx, 2] += sampled_az * dt
        # self.commands[envs_idx, 0] += 0
        
        if reset_flag: # when Go2 is not walking
            # resample both negative and positive velocity
            self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
            self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
            self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_z_range"], (len(envs_idx),), self.device)
        else: # while Go2 is walking
            # Define the full allowed range as provided in self.command_cfg.
            full_min_vx = torch.full_like(old_lin_vel_x_command, self.command_cfg["lin_vel_x_range"][0])
            full_max_vx = torch.full_like(old_lin_vel_x_command, self.command_cfg["lin_vel_x_range"][1])

            # For environments with a positive old velocity, we want to allow only non-negative values.
            pos_min_vx = torch.full_like(old_lin_vel_x_command, -self.sign_flip_rate)
            pos_max_vx = full_max_vx  # maximum remains the same.

            # For environments with a negative old velocity, we want to allow only non-positive values.
            neg_min_vx = full_min_vx  # minimum remains the same.
            neg_max_vx = torch.full_like(old_lin_vel_x_command, self.sign_flip_rate)

            # For those environments where the old velocity is exactly zero, allow the full range.
            # Create element-wise minimum and maximum bounds based on the sign of old_lin_vel_x_command.
            min_vx = torch.where(old_lin_vel_x_command > 0, pos_min_vx,
                                torch.where(old_lin_vel_x_command < 0, neg_min_vx, full_min_vx))
            max_vx = torch.where(old_lin_vel_x_command > 0, pos_max_vx,
                                torch.where(old_lin_vel_x_command < 0, neg_max_vx, full_max_vx))


            # Define the full allowed range as provided in self.command_cfg.
            full_min_vy = torch.full_like(old_lin_vel_y_command, self.command_cfg["lin_vel_y_range"][0])
            full_max_vy = torch.full_like(old_lin_vel_y_command, self.command_cfg["lin_vel_y_range"][1])

            # For environments with a positive old velocity, we want to allow only non-negative values.
            pos_min_vy = torch.full_like(old_lin_vel_y_command, -self.sign_flip_rate)
            pos_max_vy = full_max_vy  # maximum remains the same.

            # For environments with a negative old velocity, we want to allow only non-positive values.
            neg_min_vy = full_min_vy  # minimum remains the same.
            neg_max_vy = torch.full_like(old_lin_vel_y_command, self.sign_flip_rate)

            # For those environments where the old velocity is exactly zero, allow the full range.
            # Create element-wise minimum and maximum bounds based on the sign of old_lin_vel_x_command.
            min_vy = torch.where(old_lin_vel_y_command > 0, pos_min_vy,
                                torch.where(old_lin_vel_y_command < 0, neg_min_vy, full_min_vy))
            max_vy = torch.where(old_lin_vel_y_command > 0, pos_max_vy,
                                torch.where(old_lin_vel_y_command < 0, neg_max_vy, full_max_vy))


            # Define the full allowed range as provided in self.command_cfg.
            full_min_wz = torch.full_like(old_ang_vel_z_command, self.command_cfg["ang_vel_z_range"][0])
            full_max_wz = torch.full_like(old_ang_vel_z_command, self.command_cfg["ang_vel_z_range"][1])

            # For environments with a positive old velocity, we want to allow only non-negative values.
            pos_min_wz = torch.full_like(old_ang_vel_z_command, -self.sign_flip_rate)
            pos_max_wz = full_max_wz  # maximum remains the same.

            # For environments with a negative old velocity, we want to allow only non-positive values.
            neg_min_wz = full_min_wz  # minimum remains the same.
            neg_max_wz = torch.full_like(old_ang_vel_z_command, self.sign_flip_rate)

            # For those environments where the old velocity is exactly zero, allow the full range.
            # Create element-wise minimum and maximum bounds based on the sign of old_lin_vel_x_command.
            min_wz = torch.where(old_ang_vel_z_command > 0, pos_min_wz,
                                torch.where(old_ang_vel_z_command < 0, neg_min_wz, full_min_wz))
            max_wz = torch.where(old_ang_vel_z_command > 0, pos_max_wz,
                                torch.where(old_ang_vel_z_command < 0, neg_max_wz, full_max_wz))



            # Finally, clamp the new x-velocity command for each environment element-wise.
            self.commands[envs_idx, 0] = torch.max(torch.min(self.commands[envs_idx, 0], max_vx), min_vx)
            self.commands[envs_idx, 1] = torch.max(torch.min(self.commands[envs_idx, 1], max_vy), min_vy)
            self.commands[envs_idx, 2] = torch.max(torch.min(self.commands[envs_idx, 2], max_wz), min_wz)

            if V_ACC:
                # Get scalar values from config
                vx_max_scalar = self.command_cfg["lin_vel_x_range"][1]  # e.g., 1.0
                vx_min_scalar = self.command_cfg["lin_vel_x_range"][0]  # e.g., -1.0
                vy_max_scalar = self.command_cfg["lin_vel_y_range"][1]  # e.g., 1.0
                vy_min_scalar = self.command_cfg["lin_vel_y_range"][0]  # e.g., -1.0
                wz_max_scalar = self.command_cfg["ang_vel_z_range"][1]  # e.g., 1.0
                wz_min_scalar = self.command_cfg["ang_vel_z_range"][0]  # e.g., -1.0

                # Get the clamped velocities (already written to self.commands[envs_idx, 0])
                vx = self.commands[envs_idx, 0]
                vy = self.commands[envs_idx, 1]
                wz = self.commands[envs_idx, 2]

                # Flip to negative direction when hitting +vx_max
                flip_to_neg_vx = (vx >= vx_max_scalar - 1e-5)
                flip_to_neg_vx_indices = envs_idx[flip_to_neg_vx.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_neg_vx_indices.numel() > 0:
                    # self.acc_dir[flip_to_neg_indices] = -0.2 # -abs(sampled_ax)
                    self.acc_x_dir[flip_to_neg_vx_indices] = -abs(sampled_ax[flip_to_neg_vx.nonzero(as_tuple=False).squeeze(-1)])

                # Flip to positive direction when hitting -vx_min
                flip_to_pos_vx = (vx <= vx_min_scalar + 1e-5)
                flip_to_pos_vx_indices = envs_idx[flip_to_pos_vx.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_pos_vx_indices.numel() > 0:
                    # self.acc_dir[flip_to_pos_indices] = 0.2 # abs(sampled_ax)
                    self.acc_x_dir[flip_to_pos_vx_indices] = abs(sampled_ax[flip_to_pos_vx.nonzero(as_tuple=False).squeeze(-1)])


                # Flip to negative direction when hitting +vx_max
                flip_to_neg_vy = (vy >= vy_max_scalar - 1e-5)
                flip_to_neg_vy_indices = envs_idx[flip_to_neg_vy.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_neg_vy_indices.numel() > 0:
                    # self.acc_dir[flip_to_neg_indices] = -0.2 # -abs(sampled_ax)
                    self.acc_y_dir[flip_to_neg_vy_indices] = -abs(sampled_ay[flip_to_neg_vy.nonzero(as_tuple=False).squeeze(-1)])

                # Flip to positive direction when hitting -vx_min
                flip_to_pos_vy = (vy <= vy_min_scalar + 1e-5)
                flip_to_pos_vy_indices = envs_idx[flip_to_pos_vy.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_pos_vy_indices.numel() > 0:
                    # self.acc_dir[flip_to_pos_indices] = 0.2 # abs(sampled_ax)
                    self.acc_y_dir[flip_to_pos_vy_indices] = abs(sampled_ay[flip_to_pos_vy.nonzero(as_tuple=False).squeeze(-1)])
                
                # Flip to negative direction when hitting +vx_max
                flip_to_neg_wz = (wz >= wz_max_scalar - 1e-5)
                flip_to_neg_wz_indices = envs_idx[flip_to_neg_wz.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_neg_wz_indices.numel() > 0:
                    # self.acc_dir[flip_to_neg_indices] = -0.2 # -abs(sampled_ax)
                    self.acc_z_dir[flip_to_neg_wz_indices] = -abs(sampled_az[flip_to_neg_wz.nonzero(as_tuple=False).squeeze(-1)])

                # Flip to positive direction when hitting -vx_min
                flip_to_pos_wz = (wz <= wz_min_scalar + 1e-5)
                flip_to_pos_wz_indices = envs_idx[flip_to_pos_wz.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_pos_wz_indices.numel() > 0:
                    # self.acc_dir[flip_to_pos_indices] = 0.2 # abs(sampled_ax)
                    self.acc_z_dir[flip_to_pos_wz_indices] = abs(sampled_az[flip_to_pos_wz.nonzero(as_tuple=False).squeeze(-1)])
                

        # if RANDOM_RESAMPLE_EVAL:
        self.env0_command_x = self.commands[0, 0]
        self.env0_command_y = self.commands[0, 1]
        self.env0_command_z = self.commands[0, 2]
        self.env1_command_x = self.commands[1, 0]
        self.env1_command_y = self.commands[1, 1]
        self.env1_command_z = self.commands[1, 2]
        self.env2_command_x = self.commands[2, 0]
        self.env2_command_y = self.commands[2, 1]
        self.env2_command_z = self.commands[2, 2]

    def _resample_commands_gaussian_acc_v2(self, envs_idx, reset_flag):
        # if envs_idx.numel() != 0:
        #     print(f"Resample velocity for {envs_idx} at step {self.episode_length_buf}")
        #     breakpoint()
        old_lin_vel_x_command = self.commands[envs_idx, 0]
        old_lin_vel_y_command = self.commands[envs_idx, 1]

        # For y and yaw, keep your old logic:
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_z_range"], (len(envs_idx),), self.device)
        # self.commands[envs_idx, 2] = 0.0

        dt = self.dt
        # sampled_ax = torch.randn((len(envs_idx),), device=self.device) * self.acc_sigma
        sampled_ax = torch.normal(mean=self.acc_x_mean, std=self.acc_x_sigma, size=(len(envs_idx),), device=self.device)
        sampled_ay = torch.normal(mean=self.acc_y_mean, std=self.acc_y_sigma, size=(len(envs_idx),), device=self.device)
        # sampled_ax /= 2

        if V_ACC:
            self.commands[envs_idx, 0] += self.acc_x_dir[envs_idx] * dt
            self.commands[envs_idx, 1] += self.acc_y_dir[envs_idx] * dt
        else:
            self.commands[envs_idx, 0] += sampled_ax * dt
            self.commands[envs_idx, 1] += sampled_ay * dt
        # self.commands[envs_idx, 0] += 0
        
        if reset_flag: # when Go2 is not walking
            # resample both negative and positive velocity
            self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
            self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        else: # while Go2 is walking
            # Define the full allowed range as provided in self.command_cfg.
            full_min_vx = torch.full_like(old_lin_vel_x_command, self.command_cfg["lin_vel_x_range"][0])
            full_max_vx = torch.full_like(old_lin_vel_x_command, self.command_cfg["lin_vel_x_range"][1])

            # For environments with a positive old velocity, we want to allow only non-negative values.
            pos_min_vx = torch.full_like(old_lin_vel_x_command, -self.sign_flip_rate)
            pos_max_vx = full_max_vx  # maximum remains the same.

            # For environments with a negative old velocity, we want to allow only non-positive values.
            neg_min_vx = full_min_vx  # minimum remains the same.
            neg_max_vx = torch.full_like(old_lin_vel_x_command, self.sign_flip_rate)

            # For those environments where the old velocity is exactly zero, allow the full range.
            # Create element-wise minimum and maximum bounds based on the sign of old_lin_vel_x_command.
            min_vx = torch.where(old_lin_vel_x_command > 0, pos_min_vx,
                                torch.where(old_lin_vel_x_command < 0, neg_min_vx, full_min_vx))
            max_vx = torch.where(old_lin_vel_x_command > 0, pos_max_vx,
                                torch.where(old_lin_vel_x_command < 0, neg_max_vx, full_max_vx))


            # Define the full allowed range as provided in self.command_cfg.
            full_min_vy = torch.full_like(old_lin_vel_y_command, self.command_cfg["lin_vel_y_range"][0])
            full_max_vy = torch.full_like(old_lin_vel_y_command, self.command_cfg["lin_vel_y_range"][1])

            # For environments with a positive old velocity, we want to allow only non-negative values.
            pos_min_vy = torch.full_like(old_lin_vel_y_command, -self.sign_flip_rate)
            pos_max_vy = full_max_vy  # maximum remains the same.

            # For environments with a negative old velocity, we want to allow only non-positive values.
            neg_min_vy = full_min_vy  # minimum remains the same.
            neg_max_vy = torch.full_like(old_lin_vel_y_command, self.sign_flip_rate)

            # For those environments where the old velocity is exactly zero, allow the full range.
            # Create element-wise minimum and maximum bounds based on the sign of old_lin_vel_x_command.
            min_vy = torch.where(old_lin_vel_y_command > 0, pos_min_vy,
                                torch.where(old_lin_vel_y_command < 0, neg_min_vy, full_min_vy))
            max_vy = torch.where(old_lin_vel_y_command > 0, pos_max_vy,
                                torch.where(old_lin_vel_y_command < 0, neg_max_vy, full_max_vy))



            # Finally, clamp the new x-velocity command for each environment element-wise.
            self.commands[envs_idx, 0] = torch.max(torch.min(self.commands[envs_idx, 0], max_vx), min_vx)
            self.commands[envs_idx, 1] = torch.max(torch.min(self.commands[envs_idx, 1], max_vy), min_vy)


            if V_ACC:
                # Get scalar values from config
                vx_max_scalar = self.command_cfg["lin_vel_x_range"][1]  # e.g., 1.0
                vx_min_scalar = self.command_cfg["lin_vel_x_range"][0]  # e.g., -1.0
                vy_max_scalar = self.command_cfg["lin_vel_y_range"][1]  # e.g., 1.0
                vy_min_scalar = self.command_cfg["lin_vel_y_range"][0]  # e.g., -1.0

                # Get the clamped velocities (already written to self.commands[envs_idx, 0])
                vx = self.commands[envs_idx, 0]
                vy = self.commands[envs_idx, 1]

                # Flip to negative direction when hitting +vx_max
                flip_to_neg_vx = (vx >= vx_max_scalar - 1e-5)
                flip_to_neg_vx_indices = envs_idx[flip_to_neg_vx.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_neg_vx_indices.numel() > 0:
                    # self.acc_dir[flip_to_neg_indices] = -0.2 # -abs(sampled_ax)
                    self.acc_x_dir[flip_to_neg_vx_indices] = -abs(sampled_ax[flip_to_neg_vx.nonzero(as_tuple=False).squeeze(-1)])

                # Flip to positive direction when hitting -vx_min
                flip_to_pos_vx = (vx <= vx_min_scalar + 1e-5)
                flip_to_pos_vx_indices = envs_idx[flip_to_pos_vx.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_pos_vx_indices.numel() > 0:
                    # self.acc_dir[flip_to_pos_indices] = 0.2 # abs(sampled_ax)
                    self.acc_x_dir[flip_to_pos_vx_indices] = abs(sampled_ax[flip_to_pos_vx.nonzero(as_tuple=False).squeeze(-1)])


                # Flip to negative direction when hitting +vx_max
                flip_to_neg_vy = (vy >= vy_max_scalar - 1e-5)
                flip_to_neg_vy_indices = envs_idx[flip_to_neg_vy.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_neg_vy_indices.numel() > 0:
                    # self.acc_dir[flip_to_neg_indices] = -0.2 # -abs(sampled_ax)
                    self.acc_y_dir[flip_to_neg_vy_indices] = -abs(sampled_ay[flip_to_neg_vy.nonzero(as_tuple=False).squeeze(-1)])

                # Flip to positive direction when hitting -vx_min
                flip_to_pos_vy = (vy <= vy_min_scalar + 1e-5)
                flip_to_pos_vy_indices = envs_idx[flip_to_pos_vy.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_pos_vy_indices.numel() > 0:
                    # self.acc_dir[flip_to_pos_indices] = 0.2 # abs(sampled_ax)
                    self.acc_y_dir[flip_to_pos_vy_indices] = abs(sampled_ay[flip_to_pos_vy.nonzero(as_tuple=False).squeeze(-1)])
                

        # if RANDOM_RESAMPLE_EVAL:
        self.env0_command_x = self.commands[0, 0]
        self.env0_command_y = self.commands[0, 1]
        self.env0_command_z = self.commands[0, 2]
        self.env1_command_x = self.commands[1, 0]
        self.env1_command_y = self.commands[1, 1]
        self.env1_command_z = self.commands[1, 2]
        self.env2_command_x = self.commands[2, 0]
        self.env2_command_y = self.commands[2, 1]
        self.env2_command_z = self.commands[2, 2]
        
        # print("command x: ", self.commands[0, 0])

        # ax = (self.vx_plane - self.last_vx_plane) / self.dt
        # az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        # self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        # self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        
        # ax_smooth = self.ax_filtered * self.ax_scale
        # az_smooth = self.az_filtered * self.az_scale

        # # desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        # desired_pitch = torch.atan2(ax_smooth, -az_smooth)
        # desired_pitch_degrees = torch.rad2deg(desired_pitch)
        # desired_pitch_degrees = desired_pitch_degrees.unsqueeze(-1)

        # error = torch.mean(torch.square(desired_pitch_degrees - self.rot_y_deg))

        # print("Desired Tilt: ", desired_pitch_degrees)
        # print("error: ", abs(desired_pitch_degrees - self.rot_y_deg))

    
    def _resample_commands_gaussian_acc(self, envs_idx, reset_flag):
        # if envs_idx.numel() != 0:
        #     print(f"Resample velocity for {envs_idx} at step {self.episode_length_buf}")
        #     breakpoint()
        old_lin_vel_x_command = self.commands[envs_idx, 0]

        # For y and yaw, keep your old logic:
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_z_range"], (len(envs_idx),), self.device)
        # self.commands[envs_idx, 1] = 0.0
        # self.commands[envs_idx, 2] = 0.0

        # self.linvel_update_actual_freq = (1 / self.dt) / self.linvel_update_freq # 5
        # dt = 1 / self.linvel_update_actual_freq # 0.2
        # dt = self.dt * 10
        dt = self.dt
        # sampled_ax = torch.randn((len(envs_idx),), device=self.device) * self.acc_sigma
        sampled_ax = torch.normal(mean=self.acc_mean, std=self.acc_sigma, size=(len(envs_idx),), device=self.device)
        # sampled_ax /= 2

        if POS_ACC:
            self.commands[envs_idx, 0] += abs(sampled_ax) * dt
        elif NEG_ACC:
            self.commands[envs_idx, 0] += -abs(sampled_ax) * dt
        elif V_ACC:
            # self.acc_dir[envs_idx] = abs(sampled_ax)
            self.commands[envs_idx, 0] += self.acc_dir[envs_idx] * dt
        else:
            self.commands[envs_idx, 0] += sampled_ax * dt
        # self.commands[envs_idx, 0] += 0
        
        if reset_flag: # when Go2 is not walking
            # resample both negative and positive velocity
            self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        else: # while Go2 is walking
            # Define the full allowed range as provided in self.command_cfg.
            full_min = torch.full_like(old_lin_vel_x_command, self.command_cfg["lin_vel_x_range"][0])
            full_max = torch.full_like(old_lin_vel_x_command, self.command_cfg["lin_vel_x_range"][1])

            # For environments with a positive old velocity, we want to allow only non-negative values.
            pos_min = torch.full_like(old_lin_vel_x_command, -self.sign_flip_rate)
            pos_max = full_max  # maximum remains the same.

            # For environments with a negative old velocity, we want to allow only non-positive values.
            neg_min = full_min  # minimum remains the same.
            neg_max = torch.full_like(old_lin_vel_x_command, self.sign_flip_rate)

            # For those environments where the old velocity is exactly zero, allow the full range.
            # Create element-wise minimum and maximum bounds based on the sign of old_lin_vel_x_command.
            min_vx = torch.where(old_lin_vel_x_command > 0, pos_min,
                                torch.where(old_lin_vel_x_command < 0, neg_min, full_min))
            max_vx = torch.where(old_lin_vel_x_command > 0, pos_max,
                                torch.where(old_lin_vel_x_command < 0, neg_max, full_max))
                                
            # Finally, clamp the new x-velocity command for each environment element-wise.
            self.commands[envs_idx, 0] = torch.max(torch.min(self.commands[envs_idx, 0], max_vx), min_vx)



            if POS_ACC:
                # Get scalar values from config
                vx_max_scalar = self.command_cfg["lin_vel_x_range"][1]  # e.g., 1.0
                vx_min_scalar = self.command_cfg["lin_vel_x_range"][0]  # e.g., -1.0

                # Get the clamped velocities (already written to self.commands[envs_idx, 0])
                vx = self.commands[envs_idx, 0]

                # Detect elements where vx was clamped to max (or exceeded max slightly)
                flip_mask = vx >= vx_max_scalar - 1e-5  # small epsilon to handle float errors

                # Get the global indices of environments where this happened
                flip_indices = envs_idx[flip_mask.nonzero(as_tuple=False).squeeze(-1)]

                # Perform the flip to -1.0
                if flip_indices.numel() > 0:
                    self.commands[flip_indices, 0] = vx_min_scalar
                    # self.commands[flip_indices, 0] = 0.0
                    # print(f"Flipped {flip_indices.numel()} velocities to {vx_min_scalar}")
            elif NEG_ACC:
                # Get scalar values from config
                vx_max_scalar = self.command_cfg["lin_vel_x_range"][1]  # e.g., 1.0
                vx_min_scalar = self.command_cfg["lin_vel_x_range"][0]  # e.g., -1.0

                # Get the clamped velocities (already written to self.commands[envs_idx, 0])
                vx = self.commands[envs_idx, 0]

                # Detect elements where vx was clamped to max (or exceeded max slightly)
                flip_mask = vx <= vx_min_scalar + 1e-5  # small epsilon to handle float errors

                # Get the global indices of environments where this happened
                flip_indices = envs_idx[flip_mask.nonzero(as_tuple=False).squeeze(-1)]

                # Perform the flip to -1.0
                if flip_indices.numel() > 0:
                    self.commands[flip_indices, 0] = vx_max_scalar
                    # self.commands[flip_indices, 0] = 0.0
                    # print(f"Flipped {flip_indices.numel()} velocities to {vx_min_scalar}")
            elif V_ACC:
                # Get scalar values from config
                vx_max_scalar = self.command_cfg["lin_vel_x_range"][1]  # e.g., 1.0
                vx_min_scalar = self.command_cfg["lin_vel_x_range"][0]  # e.g., -1.0

                # Get the clamped velocities (already written to self.commands[envs_idx, 0])
                vx = self.commands[envs_idx, 0]

                # Flip to negative direction when hitting +vx_max
                flip_to_neg = (vx >= vx_max_scalar - 1e-5)
                flip_to_neg_indices = envs_idx[flip_to_neg.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_neg_indices.numel() > 0:
                    # self.acc_dir[flip_to_neg_indices] = -0.2 # -abs(sampled_ax)
                    self.acc_dir[flip_to_neg_indices] = -abs(sampled_ax[flip_to_neg.nonzero(as_tuple=False).squeeze(-1)])

                # Flip to positive direction when hitting -vx_min
                flip_to_pos = (vx <= vx_min_scalar + 1e-5)
                flip_to_pos_indices = envs_idx[flip_to_pos.nonzero(as_tuple=False).squeeze(-1)]
                if flip_to_pos_indices.numel() > 0:
                    # self.acc_dir[flip_to_pos_indices] = 0.2 # abs(sampled_ax)
                    self.acc_dir[flip_to_pos_indices] = abs(sampled_ax[flip_to_pos.nonzero(as_tuple=False).squeeze(-1)])

        # if RANDOM_RESAMPLE_EVAL:
        self.env0_command_x = self.commands[0, 0]
        self.env0_command_y = self.commands[0, 1]
        self.env0_command_z = self.commands[0, 2]
        self.env1_command_x = self.commands[1, 0]
        self.env1_command_y = self.commands[1, 1]
        self.env1_command_z = self.commands[1, 2]
        self.env2_command_x = self.commands[2, 0]
        self.env2_command_y = self.commands[2, 1]
        self.env2_command_z = self.commands[2, 2]
        
        # print("command x: ", self.commands[0, 0])

        # ax = (self.vx_plane - self.last_vx_plane) / self.dt
        # az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        # self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        # self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        
        # ax_smooth = self.ax_filtered * self.ax_scale
        # az_smooth = self.az_filtered * self.az_scale

        # # desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        # desired_pitch = torch.atan2(ax_smooth, -az_smooth)
        # desired_pitch_degrees = torch.rad2deg(desired_pitch)
        # desired_pitch_degrees = desired_pitch_degrees.unsqueeze(-1)

        # error = torch.mean(torch.square(desired_pitch_degrees - self.rot_y_deg))

        # print("Desired Tilt: ", desired_pitch_degrees)
        # print("error: ", abs(desired_pitch_degrees - self.rot_y_deg))

    def _resample_trajectory(self, envs_idx, reset_flag):
        # T = desired_duration  # Chosen based on aggressiveness (shorter T → more aggressive, faster motion)
        # x0 = current_position
        # xf = target_position
        # t = 0
        # dt = control_loop_time_step
        # s = t / T
        # v = (xf - x0) * (30*s**2 - 60*s**3 + 30*s**4) / T
        # send_velocity_command(v)
        # t += self.dt
        
        # x0 = self.base_pos[envs_idx].clone()
        # s = self.traj_t[envs_idx] / self.max_episode_length
        # scaling = (30*s**2 - 60*s**3 + 30*s**4).unsqueeze(-1)  # Shape: [N, 1]
        # max_episode_length_tensor = torch.tensor(self.max_episode_length, device=self.device, dtype=self.xf.dtype).unsqueeze(-1)
        # v = (self.xf[envs_idx] - x0) * scaling / max_episode_length_tensor
        # self.traj_t[envs_idx] += self.dt

        # x0: starting positions for environments being resampled
        self.x0 = self.base_pos[envs_idx].clone()
        # s: normalized time for each environment (only for the ones being resampled)
        s = self.traj_t[envs_idx] / self.max_episode_length_s[envs_idx]
        
        # pos_scaling = (10*s**3 - 15*s**4 + 6*s**5).unsqueeze(-1)
        # pos = self.x0 * (self.xf[envs_idx] - self.x0) * pos_scaling
        
        vel_scaling = (30*s**2 - 60*s**3 + 30*s**4).unsqueeze(-1)
        vel = (self.xf[envs_idx] - self.x0) * vel_scaling / self.max_episode_length_s[envs_idx].unsqueeze(-1)
        
        # acc_scaling = (60*s - 180*s**2 + 120*s**3).unsqueeze(-1)
        # acc = (self.xf[envs_idx] - self.x0) * acc_scaling / (self.max_episode_length_s[envs_idx]**2).unsqueeze(-1)

        # jerk_scaling = (60 - 360*s + 360*s**2).unsqueeze(-1)
        # jerk = (self.xf[envs_idx] - self.x0) * jerk_scaling / (self.max_episode_length_s[envs_idx]**3).unsqueeze(-1)

        # Increment trajectory timer for these envs
        # self.traj_t[envs_idx] += self.dt
        self.traj_t[envs_idx] += self.env_cfg["resampling_time_s"]

        if not reset_flag: # every step resample, not reset due to failure or time out
            # print("-----------------------------------------------------------------------------------")
            # print("envs_idx (should be all): ", envs_idx)
            # print("x0[envs_idx] (current pos): ", x0)
            # print("max_episode_length[envs_idx] (should not change until timeout): ", self.max_episode_length[envs_idx])
            # print("traj_t[envs_idx = all] (should be incremented every step): ", self.traj_t[envs_idx])
            # print("s: should be incremented every step", s)
            # print("xf[envs_idx = all] (should not change until timeout): ", self.xf[envs_idx])
            # print("v should be incremented every step: ", v)
            # print("-----------------------------------------------------------------------------------")
            # print("xf: ", self.xf[envs_idx].shape)
            # print("x0: ", self.x0.shape)
            # print("scaling: ", scaling.shape)
            # print("T: ", self.max_episode_length_s[envs_idx].unsqueeze(-1).shape)
            # print("v: ", v.shape)
            # breakpoint()
            pass

        self.commands[envs_idx, 0] = vel[:, 0]
        self.commands[envs_idx, 1] = vel[:, 0]
        self.commands[envs_idx, 2] = 0.0


        s_log = self.traj_t[0] / self.max_episode_length_s[0]

        pos_scaling_log = (10*s_log**3 - 15*s_log**4 + 6*s_log**5).unsqueeze(-1)
        pos_log = self.base_pos[0] * (self.xf[0] - self.base_pos[0]) * pos_scaling_log

        vel_scaling_log = (30*s_log**2 - 60*s_log**3 + 30*s_log**4).unsqueeze(-1)
        vel_log = (self.xf[0] - self.base_pos[0]) * vel_scaling_log / self.max_episode_length_s[0].unsqueeze(-1)

        acc_scaling_log = (60*s_log - 180*s_log**2 + 120*s_log**3).unsqueeze(-1)
        acc_log = (self.xf[0] - self.base_pos[0]) * acc_scaling_log / (self.max_episode_length_s[0]**2).unsqueeze(-1)

        jerk_scaling_log = (60 - 360*s_log + 360*s_log**2).unsqueeze(-1)
        jerk_log = (self.xf[0] - self.base_pos[0]) * jerk_scaling_log / (self.max_episode_length_s[0]**3).unsqueeze(-1)

        self.env0_command_x = vel_log[0].item()
        self.env0_command_y = vel_log[1].item()
        self.env0_command_z = 0.0
        self.env0_x0_x = self.base_pos[0, 0].item()
        self.env0_x0_y = self.base_pos[0, 1].item()
        self.env0_x0_z = self.base_pos[0, 2].item()

        self.env0_pos_x = pos_log[0].item()
        self.env0_pos_y = pos_log[1].item()
        self.env0_pos_z = pos_log[2].item()
        self.env0_acc_x = acc_log[0].item()
        self.env0_acc_y = acc_log[1].item()
        self.env0_acc_z = acc_log[2].item()
        self.env0_jerk_x = jerk_log[0].item()
        self.env0_jerk_y = jerk_log[1].item()
        self.env0_jerk_z = jerk_log[2].item()


        # print("self.env0_command_x: ", self.env0_command_x)

    def _resample_predefined_commands(self):
        if self.forward_start:
            print("Starting resample")
            self.commands[0, 0] = 0.0
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.forward_start_count == self.forward_start_len:
                self.forward_start = False
            self.forward_start_count += 1
        elif self.forward_12:
            print("Lin Vel X: 1.2")
            self.commands[0, 0] = 1.2
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.forward_12_count == self.forward_12_len:
                self.forward_12 = False
            self.forward_12_count += 1
        elif self.forward_8:
            print("Lin Vel X: 0.8")
            self.commands[0, 0] = 0.8
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.forward_8_count == self.forward_8_len:
                self.forward_8 = False
            self.forward_8_count += 1
        elif self.forward_4:
            print("Lin Vel X: 0.4")
            self.commands[0, 0] = 0.4
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.forward_4_count == self.forward_4_len:
                self.forward_4 = False
            self.forward_4_count += 1
        elif self.forward_0:
            print("Lin Vel X: 0.0")
            self.commands[0, 0] = 0.0
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.forward_0_count == self.forward_0_len:
                self.forward_0 = False
            self.forward_0_count += 1
        elif self.backward_12:
            print("Lin Vel X: -1.2")
            self.commands[0, 0] = -1.2
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.backward_12_count == self.backward_12_len:
                self.backward_12 = False
            self.backward_12_count += 1
        elif self.backward_8:
            print("Lin Vel X: -0.8")
            self.commands[0, 0] = -0.8
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.backward_8_count == self.backward_8_len:
                self.backward_8 = False
            self.backward_8_count += 1
        elif self.backward_4:
            print("Lin Vel X: -0.4")
            self.commands[0, 0] = -0.4
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.backward_4_count == self.backward_4_len:
                self.backward_4 = False
            self.backward_4_count += 1
        elif self.backward_0:
            print("Lin Vel X: 0.0")
            self.commands[0, 0] = 0.0
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.backward_0_count == self.backward_0_len:
                self.backward_0 = False
            self.backward_0_count += 1

            if self.record_video:
                print("STOP VIDEO RECORDING")
                base_path = "/home/psxkf4/Genesis/logs/paper/videos"
                file_name = self.folder_name + ".mp4"
                full_path = os.path.join(base_path, file_name)
                self.stop_recording(full_path)
    
    def _resample_random_commands_train(self):
        if self.start_stop_cmd:
            self.lin_vel_x_cmd = 0.0
            self.commands[0, 0] = self.lin_vel_x_cmd
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.start_stop_cmd_count == self.start_stop_cmd_len:
                self.start_stop_cmd = False  # End stop phase
            self.start_stop_cmd_count += 1
        elif self.forward1_cmd:
            if self.forward1_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.forward_cmd_speeds)
                    self.forward1_cmd_vel = self.lin_vel_x_cmd
            print("Resampling forward velocity randomly: ", self.forward1_cmd_vel)
            self.commands[0, 0] = self.forward1_cmd_vel
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.forward1_cmd_count == self.forward1_cmd_len:
                self.forward1_cmd = False
            self.forward1_cmd_count += 1
        elif self.forward2_cmd:
            if self.forward2_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.forward_cmd_speeds)
                    self.forward2_cmd_vel = self.lin_vel_x_cmd
            print("Resampling forward velocity randomly: ", self.forward2_cmd_vel)
            self.commands[0, 0] = self.forward2_cmd_vel
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.forward2_cmd_count == self.forward2_cmd_len:
                self.forward2_cmd = False
            self.forward2_cmd_count += 1
        elif self.forward3_cmd:
            if self.forward3_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.forward_cmd_speeds)
                    self.forward3_cmd_vel = self.lin_vel_x_cmd
            print("Resampling forward velocity randomly: ", self.forward3_cmd_vel)
            self.commands[0, 0] = self.forward3_cmd_vel
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.forward3_cmd_count == self.forward3_cmd_len:
                self.forward3_cmd = False
            self.forward3_cmd_count += 1
        elif self.middle_stop_cmd:
            self.lin_vel_x_cmd = 0.0
            print("Middle-stop phase active: commanding stop (0 m/s).")
            self.commands[0, 0] = self.lin_vel_x_cmd
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.middle_stop_cmd_count == self.middle_stop_cmd_len:
                self.middle_stop_cmd = False  # End stop phase
            self.middle_stop_cmd_count += 1
        elif self.backward1_cmd:
            if self.backward1_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.backward_cmd_speeds)
                    self.backward1_cmd_vel = self.lin_vel_x_cmd
            print("Resampling forward velocity randomly: ", self.backward1_cmd_vel)
            self.commands[0, 0] = self.backward1_cmd_vel
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.backward1_cmd_count == self.backward1_cmd_len:
                self.backward1_cmd = False
            self.backward1_cmd_count += 1
        elif self.backward2_cmd:
            if self.backward2_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.backward_cmd_speeds)
                    self.backward2_cmd_vel = self.lin_vel_x_cmd
            print("Resampling forward velocity randomly: ", self.backward2_cmd_vel)
            self.commands[0, 0] = self.backward2_cmd_vel
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.backward2_cmd_count == self.backward2_cmd_len:
                self.backward2_cmd = False
            self.backward2_cmd_count += 1
        elif self.backward3_cmd:
            if self.backward3_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.backward_cmd_speeds)
                    self.backward3_cmd_vel = self.lin_vel_x_cmd
            print("Resampling forward velocity randomly: ", self.backward3_cmd_vel)
            self.commands[0, 0] = self.backward3_cmd_vel
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.backward3_cmd_count == self.backward3_cmd_len:
                self.backward3_cmd = False
            self.backward3_cmd_count += 1
        elif self.finish_stop_cmd:
            self.lin_vel_x_cmd = 0.0
            print("Finish-stop phase active: commanding stop (0 m/s).")
            self.commands[0, 0] = self.lin_vel_x_cmd
            self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.finish_stop_cmd_count == self.finish_stop_cmd_len:
                self.fnish_cmd_stop = False  # End stop phase
            self.finish_stop_cmd_count += 1

    def _resample_random_commands_eval(self):
        if self.start_stop_cmd:
            self.lin_vel_x_cmd = 0.0
            if ROLL_EVAL:
                self.lin_vel_y_cmd = 0.0
            print("Start-stop phase active: commanding stop (0 m/s).")
            self.commands[0, 0] = self.lin_vel_x_cmd
            if ROLL_EVAL: 
                self.commands[0, 1] = self.lin_vel_y_cmd
            else:
                self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.start_stop_cmd_count == self.start_stop_cmd_len:
                self.start_stop_cmd = False  # End stop phase
            self.start_stop_cmd_count += 1
        elif self.forward1_cmd:
            if self.forward1_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.forward_cmd_speeds)
                    self.forward1_cmd_vel = self.lin_vel_x_cmd
                    if ROLL_EVAL:
                        self.lin_vel_y_cmd = random.choice(self.lateral_cmd_speeds)
                        self.lateral1_cmd_vel = self.lin_vel_y_cmd
            print("Resampling forward velocity randomly: ", self.forward1_cmd_vel)
            if ROLL_EVAL:
                print("Resampling lateral velocity randomly: ", self.lateral1_cmd_vel)
            self.commands[0, 0] = self.forward1_cmd_vel
            if ROLL_EVAL: 
                self.commands[0, 1] = self.lateral1_cmd_vel
            else:
                self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.forward1_cmd_count == self.forward1_cmd_len:
                self.forward1_cmd = False
            self.forward1_cmd_count += 1
        elif self.forward2_cmd:
            if self.forward2_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.forward_cmd_speeds)
                    self.forward2_cmd_vel = self.lin_vel_x_cmd
                    if ROLL_EVAL:
                        self.lin_vel_y_cmd = random.choice(self.lateral_cmd_speeds)
                        self.lateral2_cmd_vel = self.lin_vel_y_cmd
            print("Resampling forward velocity randomly: ", self.forward2_cmd_vel)
            if ROLL_EVAL:
                print("Resampling lateral velocity randomly: ", self.lateral2_cmd_vel)
            self.commands[0, 0] = self.forward2_cmd_vel
            if ROLL_EVAL: 
                self.commands[0, 1] = self.lateral2_cmd_vel
            else:
                self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.forward2_cmd_count == self.forward2_cmd_len:
                self.forward2_cmd = False
            self.forward2_cmd_count += 1
        elif self.forward3_cmd:
            if self.forward3_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.forward_cmd_speeds)
                    self.forward3_cmd_vel = self.lin_vel_x_cmd
                    if ROLL_EVAL:
                        self.lin_vel_y_cmd = random.choice(self.lateral_cmd_speeds)
                        self.lateral3_cmd_vel = self.lin_vel_y_cmd
            print("Resampling forward velocity randomly: ", self.forward3_cmd_vel)
            if ROLL_EVAL:
                print("Resampling lateral velocity randomly: ", self.lateral3_cmd_vel)
            self.commands[0, 0] = self.forward3_cmd_vel
            if ROLL_EVAL: 
                self.commands[0, 1] = self.lateral3_cmd_vel
            else:
                self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.forward3_cmd_count == self.forward3_cmd_len:
                self.forward3_cmd = False
            self.forward3_cmd_count += 1
        elif self.middle_stop_cmd:
            self.lin_vel_x_cmd = 0.0
            if ROLL_EVAL:
                self.lin_vel_y_cmd = 0.0
            print("Middle-stop phase active: commanding stop (0 m/s).")
            self.commands[0, 0] = self.lin_vel_x_cmd
            if ROLL_EVAL: 
                self.commands[0, 1] = self.lin_vel_y_cmd
            else:
                self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.middle_stop_cmd_count == self.middle_stop_cmd_len:
                self.middle_stop_cmd = False  # End stop phase
            self.middle_stop_cmd_count += 1
        elif self.backward1_cmd:
            if self.backward1_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.backward_cmd_speeds)
                    self.backward1_cmd_vel = self.lin_vel_x_cmd
                    if ROLL_EVAL:
                        self.lin_vel_y_cmd = random.choice(self.lateral_cmd_speeds)
                        self.lateral4_cmd_vel = self.lin_vel_y_cmd
            print("Resampling forward velocity randomly: ", self.backward1_cmd_vel)
            if ROLL_EVAL:
                print("Resampling lateral velocity randomly: ", self.lateral4_cmd_vel)
            self.commands[0, 0] = self.backward1_cmd_vel
            if ROLL_EVAL: 
                self.commands[0, 1] = self.lateral4_cmd_vel
            else:
                self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.backward1_cmd_count == self.backward1_cmd_len:
                self.backward1_cmd = False
            self.backward1_cmd_count += 1
        elif self.backward2_cmd:
            if self.backward2_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.backward_cmd_speeds)
                    self.backward2_cmd_vel = self.lin_vel_x_cmd
                    if ROLL_EVAL:
                        self.lin_vel_y_cmd = random.choice(self.lateral_cmd_speeds)
                        self.lateral5_cmd_vel = self.lin_vel_y_cmd
            print("Resampling forward velocity randomly: ", self.backward2_cmd_vel)
            if ROLL_EVAL:
                print("Resampling lateral velocity randomly: ", self.lateral5_cmd_vel)
            self.commands[0, 0] = self.backward2_cmd_vel
            if ROLL_EVAL: 
                self.commands[0, 1] = self.lateral5_cmd_vel
            else:
                self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.backward2_cmd_count == self.backward2_cmd_len:
                self.backward2_cmd = False
            self.backward2_cmd_count += 1
        elif self.backward3_cmd:
            if self.backward3_cmd_count == 0:
                if not self.duplicate:
                    self.lin_vel_x_cmd = random.choice(self.backward_cmd_speeds)
                    self.backward3_cmd_vel = self.lin_vel_x_cmd
                    if ROLL_EVAL:
                        self.lin_vel_y_cmd = random.choice(self.lateral_cmd_speeds)
                        self.lateral6_cmd_vel = self.lin_vel_y_cmd
            print("Resampling forward velocity randomly: ", self.backward3_cmd_vel)
            if ROLL_EVAL:
                print("Resampling lateral velocity randomly: ", self.lateral6_cmd_vel)
            self.commands[0, 0] = self.backward3_cmd_vel
            if ROLL_EVAL: 
                self.commands[0, 1] = self.lateral6_cmd_vel
            else:
                self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.backward3_cmd_count == self.backward3_cmd_len:
                self.backward3_cmd = False
            self.backward3_cmd_count += 1
        elif self.finish_stop_cmd:
            self.lin_vel_x_cmd = 0.0
            if ROLL_EVAL:
                self.lin_vel_y_cmd = 0.0
            print("Finish-stop phase active: commanding stop (0 m/s).")
            self.commands[0, 0] = self.lin_vel_x_cmd
            if ROLL_EVAL: 
                self.commands[0, 1] = self.lin_vel_y_cmd
            else:
                self.commands[0, 1] = 0.0
            self.commands[0, 2] = 0.0
            if self.finish_stop_cmd_count == self.finish_stop_cmd_len:
                self.fnish_cmd_stop = False  # End stop phase
            self.finish_stop_cmd_count += 1
        
            if self.record_video:
                print("STOP VIDEO RECORDING")
                base_path = "/home/psxkf4/Genesis/logs/paper/videos"
                file_name = self.folder_name + ".mp4"
                full_path = os.path.join(base_path, file_name)
                self.stop_recording(full_path)

        if PLOT_TILT_ERROR_VEL_ACC_HEIGHT_CMDREC:

            linvel_x_error = abs(self.commands[0, 0] - self.base_lin_vel_x)
            linvel_y_error = abs(self.commands[0, 1] - self.base_lin_vel_y)
            angvel_z_error = abs(self.commands[0, 2] - self.base_ang_vel[0, 2])


            self.append_limited(self.command_linvel_x_list, self.commands[0, 0].item())
            self.append_limited(self.current_linvel_x_list, self.base_lin_vel_x.item())
            self.append_limited(self.linvel_x_error_list, linvel_x_error)
            self.append_limited(self.command_linvel_y_list, self.commands[0, 1].item())
            self.append_limited(self.current_linvel_y_list, self.base_lin_vel_y.item())
            self.append_limited(self.linvel_y_error_list, linvel_y_error)
            self.append_limited(self.command_angvel_z_list, self.commands[0, 2].item())
            self.append_limited(self.current_angvel_z_list, self.base_ang_vel[0, 2].item())
            self.append_limited(self.angvel_z_error_list, angvel_z_error)

        ax = (self.vx_plane - self.last_vx_plane) / self.dt
        az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        
        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale

        # desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        desired_pitch = torch.atan2(ax_smooth, -az_smooth)
        desired_pitch_degrees = torch.rad2deg(desired_pitch)
        desired_pitch_degrees = desired_pitch_degrees.unsqueeze(-1)

        error_1 = torch.mean(torch.square(desired_pitch_degrees - self.rot_y_deg))
    
    def generate_subterrain_grid(self, rows, cols, terrain_types, weights):
        """
        Generate a 2D grid (rows x cols) of terrain strings chosen randomly
        based on 'weights', but do NOT place 'pyramid_sloped_terrain' adjacent 
        to another 'pyramid_sloped_terrain'.
        """
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        sub_terrain_z_values = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                terrain_choice = random.choices(terrain_types, weights=weights, k=1)[0]
                if terrain_choice == "pyramid_sloped_terrain":
                    terrain_choice = random.choice(["pyramid_sloped_terrain", "pyramid_down_sloped_terrain"])
                elif terrain_choice == "pyramid_stairs_terrain":
                    # Define terrain options and their corresponding probabilities
                    terrain_options = ["pyramid_stairs_terrain", "pyramid_down_stairs_terrain"]
                    terrain_weights = [0.0, 1.0]  # climb up priority
                    # Choose terrain based on the weights
                    terrain_choice = random.choices(terrain_options, weights=terrain_weights, k=1)[0]

                z_value = self.check_terrain_type_and_return_value(terrain_choice)
                grid[i][j] = terrain_choice
                sub_terrain_z_values[i][j] = z_value
        return grid, sub_terrain_z_values


    def check_terrain_type_and_return_value(self, terrain_choice):

        if terrain_choice == "flat_terrain":
            return 0.0
        elif terrain_choice == "random_uniform_terrain":
            return 0.5
        elif terrain_choice == "discrete_obstacles_terrain":
            return 0.5
        elif terrain_choice == "pyramid_sloped_terrain":
            return 3.0
        elif terrain_choice == "pyramid_down_sloped_terrain":
            return -0.1
        elif terrain_choice == "pyramid_stairs_terrain":
            return 5.0
        elif terrain_choice == "pyramid_down_stairs_terrain":
            return -0.1
        elif terrain_choice == "pyramid_steep_down_stairs_terrain":
            return -3.0
        elif terrain_choice == "wave_terrain":
            return 0.5
        else:
            return 1.0

    def init_foot(self):
        self.feet_num = len(self.feet_indices)
       
        self.step_period = self.reward_cfg["step_period"]
        self.step_offset = self.reward_cfg["step_offset"]
        self.step_height_for_front = self.reward_cfg["front_feet_relative_height_from_base"]
        self.step_height_for_front_from_world = self.reward_cfg["front_feet_relative_height_from_world"]
        self.step_height_for_rear = self.reward_cfg["rear_feet_relative_height_from_base"]
        #todo get he first feet_pos here 
        # Get positions for all links and slice using indices
        all_links_pos = self.robot.get_links_pos()
        all_links_vel = self.robot.get_links_vel()

        self.feet_pos = all_links_pos[:, self.feet_indices, :]
        self.feet_front_pos = all_links_pos[:, self.feet_front_indices, :]
        self.feet_rear_pos = all_links_pos[:, self.feet_rear_indices, :]
        self.feet_vel = all_links_vel[:, self.feet_indices, :]
        self.front_feet_pos_base = self._world_to_base_transform(self.feet_front_pos, self.base_pos, self.base_quat)
        self.rear_feet_pos_base = self._world_to_base_transform(self.feet_rear_pos, self.base_pos, self.base_quat)

    def update_feet_state(self):
        # Get positions for all links and slice using indices
        all_links_pos = self.robot.get_links_pos()
        all_links_vel = self.robot.get_links_vel()

        self.feet_pos = all_links_pos[:, self.feet_indices, :]
        self.feet_front_pos = all_links_pos[:, self.feet_front_indices, :]
        self.feet_rear_pos = all_links_pos[:, self.feet_rear_indices, :]
        self.feet_vel = all_links_vel[:, self.feet_indices, :]
        self.front_feet_pos_base = self._world_to_base_transform(self.feet_front_pos, self.base_pos, self.base_quat)
        self.rear_feet_pos_base = self._world_to_base_transform(self.feet_rear_pos, self.base_pos, self.base_quat)

    def _quaternion_to_matrix(self, quat):
        w, x, y, z = quat.unbind(dim=-1)
        R = torch.stack([
            1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)
        ], dim=-1).reshape(-1, 3, 3)
        return R

    def _world_to_base_transform(self, points_world, base_pos, base_quat):
        # Convert quaternion to rotation matrix
        R = self._quaternion_to_matrix(base_quat)

        # Subtract base position to get relative position
        points_relative = points_world - base_pos.unsqueeze(1)

        # Apply rotation to transform to base frame
        points_base = torch.einsum('bij,bkj->bki', R.transpose(1, 2), points_relative)
        return points_base

    def post_physics_step_callback(self):
        self.update_feet_state()
        self.phase = (self.episode_length_buf * self.dt) % self.step_period / self.step_period
        # Assign phases for quadruped legs
        """
        small_offset = 0.05  # tweak as needed, 0 < small_offset < step_offset typically
        self.phase_FL_RR = self.phase
        self.phase_FR_RL = (self.phase + self.step_offset) % 1

        # Now offset one leg in each diagonal pair slightly
        phase_FL = self.phase_FL_RR
        phase_RR = (self.phase_FL_RR + small_offset) % 1     # shifted by small_offset

        phase_FR = self.phase_FR_RL
        phase_RL = (self.phase_FR_RL + small_offset) % 1     # shifted by small_offset

        # Concatenate in the order (FL, FR, RL, RR)
        self.leg_phase = torch.cat([
            phase_FL.unsqueeze(1),
            phase_FR.unsqueeze(1),
            phase_RL.unsqueeze(1),
            phase_RR.unsqueeze(1)
        ], dim=-1)
        """

        # Assign phases for quadruped legs
        self.phase_FL_RR = self.phase  # Front-left (FL) and Rear-right (RR) in sync
        self.phase_FR_RL = (self.phase + self.step_offset) % 1  # Front-right (FR) and Rear-left (RL) offset

        # Assign phases to legs based on their indices (FL, FR, RL, RR) order matters
        self.leg_phase = torch.cat([
            self.phase_FL_RR.unsqueeze(1),  # FL
            self.phase_FR_RL.unsqueeze(1),  # FR
            self.phase_FR_RL.unsqueeze(1),  # RL
            self.phase_FL_RR.unsqueeze(1)   # RR
        ], dim=-1)
        
    # Function to update camera position and lookat
    def update_camera_pose(self):

        # Compute new camera position
        x_pos = self.target[0] + self.radius * np.cos(self.theta)
        y_pos = self.target[1] + self.radius * np.sin(self.theta)
        eye = np.array([x_pos, y_pos, self.camera_height])  # Camera position

        # Update the camera pose
        self.cam_0.set_pose(
            pos=eye,
            lookat=self.target
        )

        # Increment theta for the next frame
        self.theta += self.theta_increment

        if self._recording and len(self._recorded_frames) < 150:
            if self.show_vis:
                self.cam_0.render(
                    rgb=True,
                )
            frame, _, _, _ = self.cam_0.render()
            self._recorded_frames.append(frame)
        elif self.show_vis:
            self.cam_0.render(
                rgb=True,
            )

    def get_data(self):
        # return self.lin_vel_x_range_min, self.lin_vel_x_range_max, self.tracking_lin_vel_rew_mean, self.tracking_lin_vel_rew_threshold_one, self.desired_pitch_mean, self.pitch_count, self.action_rate_scale
        if TRAJECTORY_RESAMPLE:
            return self.env0_command_x, self.env0_command_y, self.env0_command_z, self.env0_x0_x, self.env0_x0_y, self.env0_x0_z, self.env0_xf_x, self.env0_xf_y, self.env0_xf_z, self.max_episode_length_s[0], self.env0_pos_x, self.env0_pos_y, self.env0_pos_z, self.env0_acc_x, self.env0_acc_y, self.env0_acc_z, self.env0_jerk_x, self.env0_jerk_y, self.env0_jerk_z
        elif ACC_PROFILE_RESAMPLE_V2 or ACC_PROFILE_RESAMPLE_V3:
            return self.mean_pitch_error_normalized, self.mean_roll_error_normalized, self.mean_accel_norm_normalized, self.smoothed_ax_mean, self.smoothed_az_mean, self.env0_command_x, self.env0_command_y, self.env0_command_z, self.env1_command_x, self.env1_command_y, self.env1_command_z, self.env2_command_x, self.env2_command_y, self.env2_command_z
        else:
            return self.mean_pitch_error_normalized, self.mean_accel_norm_normalized, self.smoothed_ax_mean, self.smoothed_az_mean, self.env0_command_x, self.env0_command_y, self.env0_command_z, self.env1_command_x, self.env1_command_y, self.env1_command_z, self.env2_command_x, self.env2_command_y, self.env2_command_z

    def step(self, actions):
        # self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        if DELAY:
            if self.env_cfg["randomize_delay"]:
                # 3️⃣ Store new actions in delay buffer (Shift the buffer)
                self.action_delay_buffer[:, :, :-1] = self.action_delay_buffer[:, :, 1:].clone()
                self.action_delay_buffer[:, :, -1] = self.actions  # Insert latest action

                # 3) Vectorized gather for delayed actions

                T = self.action_delay_buffer.shape[-1]  # T = max_delay_steps + 1
                # (num_envs, num_actions)
                delayed_indices = (T - 1) - self.motor_delay_steps
                # Expand to (num_envs, num_actions, 1)
                gather_indices = delayed_indices.unsqueeze(-1)

                # Gather from last dimension
                delayed_actions = self.action_delay_buffer.gather(dim=2, index=gather_indices).squeeze(-1)

                exec_actions = delayed_actions
            else:
                exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        else:
            exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        dof_pos_list = []
        dof_vel_list = []
        for i in range(self.env_cfg['decimation']):
            self.torques = self._compute_torques(exec_actions)
            if self.num_envs == 0:
                torques = self.torques.squeeze()
                self.robot.control_dofs_force(torques, self.motor_dofs)
            else:
                self.robot.control_dofs_force(self.torques, self.motor_dofs)
            self.scene.step()
            self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
            self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

            if i == 0 or i == 2:
                dof_pos_list.append(self.robot.get_dofs_position().detach().cpu())
                dof_vel_list.append(self.robot.get_dofs_velocity().detach().cpu())

        # target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.dof_pos_list = dof_pos_list
        self.dof_vel_list = dof_vel_list
        # self.torques = self._compute_torques(exec_actions)
        # self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        
        # self.scene.step()
        # Check for NaNs in base pose and quat
        pos_after_step = self.robot.get_pos()
        quat_after_step = self.robot.get_quat()

        # Identify bad environments
        bad_envs = torch.isnan(pos_after_step).any(dim=1) | torch.isnan(quat_after_step).any(dim=1)
        if bad_envs.any():
            print(f"NaN detected in {bad_envs.sum().item()} envs. Removing from batch.")
            print(f"bad actions {self.actions[bad_envs]}")
            self.reset_buf[bad_envs] = True

        # 2a. Check for NaNs in base state
        # if torch.isnan(self.robot.get_pos()).any() or torch.isnan(self.robot.get_quat()).any():
        #     print("NaN detected right after scene.step() in base pos/quat!")
        #     print("Base pos:", self.robot.get_pos())
        #     print("Base quat:", self.robot.get_quat())
        #     raise ValueError("NaNs in base pose after scene step.")
        # 2b. Check for NaNs in DOF states
        # dof_pos_check = self.robot.get_dofs_position(self.motor_dofs)
        # dof_vel_check = self.robot.get_dofs_velocity(self.motor_dofs)
        # if torch.isnan(dof_pos_check).any() or torch.isnan(dof_vel_check).any():
        #     print("NaN detected right after scene.step() in DOF pos/vel!")
        #     print("DOF pos:", dof_pos_check)
        #     print("DOF vel:", dof_vel_check)
        #     raise ValueError("NaNs in DOF states after scene step.")



        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        self.rot_x_deg = self.base_euler[:, 0].unsqueeze(-1) 
        self.rot_y_deg = self.base_euler[:, 1].unsqueeze(-1)
        self.rot_z_deg = self.base_euler[:, 2].unsqueeze(-1) 
        # print("Roll [deg]: ", self.rot_x_deg)
        # print("Pitch [deg]: ", self.rot_y_deg)
        # print("Yaw [deg]: ", self.rot_z_deg)
        self.rot_x = torch.deg2rad(self.rot_x_deg)
        self.rot_y = torch.deg2rad(self.rot_y_deg)

        # print("Pitch: ", self.rot_y_deg)
        inv_base_quat = inv_quat(self.base_quat)
        # print("self.robot.get_vel(): ", self.robot.get_vel())
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        # print("self.base_lin_vel[:]: ", self.base_lin_vel[:])
        # breakpoint()

        self.base_lin_vel_x = self.base_lin_vel[:, 0]
        self.base_lin_vel_y = self.base_lin_vel[:, 1]
        self.base_lin_vel_z = self.base_lin_vel[:, 2]
        # print("Vx_body: ", self.base_lin_vel_x)
        # print("Vz_body: ", self.base_lin_vel_z)

        cos_r = torch.cos(self.rot_x)
        sin_r = torch.sin(self.rot_x)
        cos_r_flat = cos_r.squeeze(-1)   # shape [4096]
        sin_r_flat = sin_r.squeeze(-1)   # shape [4096]

        cos_p = torch.cos(self.rot_y)
        sin_p = torch.sin(self.rot_y)
        cos_p_flat = cos_p.squeeze(-1)   # shape [4096]
        sin_p_flat = sin_p.squeeze(-1)   # shape [4096]

        vx_plane_temp = self.vx_plane.clone()
        vy_plane_temp = self.vy_plane.clone()
        vz_world_temp = self.vz_world.clone()
        self.last_vx_plane = vx_plane_temp
        self.last_vy_plane = vy_plane_temp
        self.last_vz_world = vz_world_temp

        self.vx_plane = self.base_lin_vel_x * cos_p_flat + self.base_lin_vel_z * sin_p_flat
        self.vy_plane = self.base_lin_vel_y * cos_r_flat + self.base_lin_vel_z * sin_r_flat
        self.vz_world = -self.base_lin_vel_x * sin_p_flat + self.base_lin_vel_z * cos_p_flat

        # print("self.base_lin_vel_x: ", self.base_lin_vel_x)
        # print("self.base_lin_vel_z: ", self.base_lin_vel_z)
        # print("cos_p: ", cos_p_flat)
        # print("sin_p: ", sin_p_flat)
        # print("Vx_plane: ", self.vx_plane)
        # print("Vz_world: ", self.vz_world)
        # print("Last_Vx_plane: ", self.last_vx_plane)
        # print("Last_vz_world: ", self.last_vz_world)
        self.ax = (self.vx_plane - self.last_vx_plane) / self.dt
        self.ay = (self.vy_plane - self.last_vy_plane) / self.dt
        self.az = (self.vz_world - self.last_vz_world) / self.dt

        # print("ax: ", ax)
        # print("az: ", az)

        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_pos_delta = (self.dof_pos - self.default_dof_pos)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.hip_pos[:] = self.robot.get_dofs_position(self.hip_dofs)
        self.hip_vel[:] = self.robot.get_dofs_velocity(self.hip_dofs)
        self.contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )       

        if TEACHER_STUDENT:
            self.base_ang_vel_his =  torch.cat((self.base_ang_vel_his[:, 3:], self.base_ang_vel), dim=-1)
            self.dof_pos_his = torch.cat((self.dof_pos_his[:, 12:], self.dof_pos), dim=-1)
            self.dof_pos_delta_his = torch.cat((self.dof_pos_delta_his[:, 12:], self.dof_pos_delta), dim=-1)
            self.dof_vel_his = torch.cat((self.dof_vel_his[:, 12:], self.dof_vel), dim=-1)
            self.cmd_his =  torch.cat((self.cmd_his[:, 3:], self.commands), dim=-1)
            self.projected_gravity_his = torch.cat((self.projected_gravity_his[:, 3:], self.projected_gravity), dim=-1)
            self.action_his = torch.cat((self.action_his[:, 12:], self.actions), dim=-1)


        # contact_force_tensor = self.robot.get_links_net_contact_force()
        # # Make sure it's a torch tensor
        # if not torch.is_tensor(contact_force_tensor):
        #     contact_force_tensor = torch.tensor(contact_force_tensor, device=self.device, dtype=gs.tc_float)
        # else:
        #     contact_force_tensor = contact_force_tensor.to(device=self.device, dtype=gs.tc_float)
        # # Optional: add shape check
        # assert contact_force_tensor.shape == self.contact_forces.shape, \
        #     f"Shape mismatch: {contact_force_tensor.shape} vs {self.contact_forces.shape}"
        # self.contact_forces.copy_(contact_force_tensor)
 
        
        # print("episode length[0]: ", self.episode_length_buf[0])
        # episode length:  tensor([450, 500, 185,  ..., 350, 725, 704], device='cuda:0',
        # dtype=torch.int32)
        # print(int(self.env_cfg["resampling_time_s"] / self.dt)) # 1
        # breakpoint()
        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        ) # This number (200) represents the interval—every 200 simulation steps, new high-level commands should be resampled.

        self.post_physics_step_callback()
        if ACC_PROFILE_RESAMPLE:
            reset_flag = False
            self._resample_commands_gaussian_acc(envs_idx, reset_flag)
        elif ACC_PROFILE_RESAMPLE_V2:
            reset_flag = False
            self._resample_commands_gaussian_acc_v2(envs_idx, reset_flag)
        elif ACC_PROFILE_RESAMPLE_V3:
            reset_flag = False
            self._resample_commands_gaussian_acc_v3(envs_idx, reset_flag)
        elif DESIRED_PITCH_COMMAND:
            self._resample_desired_pitch(envs_idx)
        elif PREDEFINED_RESAMPLE_EVAL or PREDEFINED_RESAMPLE_TRY_EVAL:
            self._resample_predefined_commands()
        elif RANDOM_RESAMPLE_TRAIN:
            self._resample_random_commands_train()
        elif RANDOM_RESAMPLE_EVAL:
            self._resample_random_commands_eval()
        elif TRAJECTORY_RESAMPLE:
            reset_flag = False
            self._resample_trajectory(envs_idx, reset_flag)
        elif MIX_RESAMPLE:
            if self.switch_resample:
                reset_flag = False
                self._resample_commands_gaussian_acc(envs_idx, reset_flag)
            else:
                self._resample_commands(envs_idx)
        else:
            self._resample_commands(envs_idx)
        # self._randomize_rigids(envs_idx)
        # random push
        self.common_step_counter += 1
        push_interval_s = self.env_cfg['push_interval_s']
        if push_interval_s > 0:
            max_push_vel_xy = self.env_cfg['max_push_vel_xy']
            dofs_vel = self.robot.get_dofs_velocity() # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            push_vel[((self.common_step_counter + self.env_identities) % int(push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            self.robot.set_dofs_velocity(dofs_vel)

        # print("self.episode_length_buf: ", self.episode_length_buf)
        # print("self.max_episode_length: ", self.max_episode_length)
        self.check_termination()
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        # print("time_out_idx: ", time_out_idx)

        # print("reset envs because of timeout and failures: ", self.reset_buf)
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.compute_rewards()

        self.compute_observations()

        if self.whole_view:
            self.update_camera_pose()
        else:
            self._render_headless()

        if VIDEO_RECORD:
            self.video_record_count += 1
            if self.video_record_count == self.video_record_len:
                print("STOP VIDEO RECORDING")
                self.stop_recording("/home/psxkf4/Genesis/logs/paper/videos/run_001.mp4")

        if TEACHER_STUDENT:
            return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
        else:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def compute_observations(self):
        sin_phase = torch.sin(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)
        cos_phase = torch.cos(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)
        
        # ax = (self.vx_plane - self.last_vx_plane) / self.dt
        # az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt
        # self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        # self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        # ax_smooth = self.ax_filtered * self.ax_scale
        # az_smooth = self.az_filtered * self.az_scale
        # desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        # desired_pitch_degrees = torch.rad2deg(desired_pitch)
        # desired_pitch_degrees = desired_pitch_degrees.unsqueeze(-1)
        # # print("desired_pitch_degrees: ", desired_pitch_degrees.shape)
        # tilt_error = desired_pitch_degrees - self.rot_y_deg

        # # Right before building self.obs_buf
        # if torch.isnan(self.base_lin_vel).any() or torch.isnan(self.base_ang_vel).any():
        #     print("NaN in base_lin_vel or base_ang_vel before obs!")
        #     print("base_lin_vel:", self.base_lin_vel)
        #     print("base_ang_vel:", self.base_ang_vel)
        #     raise ValueError("NaNs in velocity terms before building obs.")

        # # If you're computing sin/cos phases, check them too:
        # if torch.isnan(self.leg_phase).any():
        #     print("NaN in leg_phase before obs!")
        #     print("leg_phase:", self.leg_phase)
        #     raise ValueError("NaNs in leg_phase.")
        # print("shape of self.base_ang_vel: ", self.base_ang_vel.shape)
        # print("shape of self.rot_y_deg: ", self.rot_y_deg.shape)
        # print("shape of self.ax: ", self.ax.unsqueeze(-1).shape)
        # print("shape of self.az: ", self.az.unsqueeze(-1).shape)
        # print("shape of self.rot_y: ", self.rot_y.shape)
        # print("shape of tilt_error: ", tilt_error.shape)
        # print("shape of self.frictions ", self.frictions.shape)
        # print("shape of self.added_masses: ", self.added_masses.shape)
        # print("shape of self.motor_strengths: ", self.motor_strengths.shape)
        # breakpoint()
        # compute observations
        self.obs_buf = torch.cat(
            [
                # self.ax.unsqueeze(-1), # 1
                # self.az.unsqueeze(-1), # 1
                # self.rot_y_deg, # 1
                # self.rot_y, # 1
                # tilt_error, # 1
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                # self.xf[:, 0].unsqueeze(1), # 1
                # self.max_episode_length.unsqueeze(1), # 1
                self.commands * self.commands_scale,  # 3
                # (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_pos_delta * self.obs_scales["dof_pos"],  # 12
                # self.dof_pos * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                # self.rot_y_deg, # 1
                # sin_phase, #4
                # cos_phase #4
            ],
            axis=-1,
        )
                    
        if TEACHER_STUDENT:
            self.domain_randomizations_buf = torch.cat(
                [
                    self.frictions, # 24
                    self.added_masses, # 1
                    self.motor_strengths, # 12
                ],
                axis=-1,
            )
            self.obs_history_buf = torch.cat(
                [
                    self.base_ang_vel_his, # 15*3
                    self.projected_gravity_his, # 15*3
                    self.cmd_his, # 15*3
                    self.dof_pos_delta_his, # 15*12
                    # self.dof_pos_his, # 15*12
                    self.dof_vel_his, # 15*12
                    self.action_his # 15*12
                ],
                axis=-1,
            )
        # else:
        # compute observations
        self.privileged_obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                # self.ax.unsqueeze(-1), # 1
                # self.az.unsqueeze(-1), # 1
                # self.rot_y_deg, # 1
                # self.rot_y, # 1
                # tilt_error, # 1
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                # self.xf[:, 0].unsqueeze(1), # 1
                # self.max_episode_length.unsqueeze(1), # 1
                self.commands * self.commands_scale,  # 3
                # (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_pos_delta * self.obs_scales["dof_pos"],  # 12
                # self.dof_pos * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                # self.rot_y_deg, # 1
                # sin_phase, #4
                # cos_phase #4
            ],
            axis=-1,
        )
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -self.clip_obs, self.clip_obs)

        self.obs_buf = torch.clip(self.obs_buf, -self.clip_obs, self.clip_obs)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        self.check_and_sanitize_observations()
        # self.check_and_reset_observations()

        self.second_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.a_count += 1
        # if self.a_count % (1 / (self.dt)) == 0:
        # if self.a_count % self.linvel_update_freq == 0:
        #     # Since self.base_lin_vel_x_low_freq is a reference to self.base_lin_vel[:, 0], 
        #     # it implicitly updates whenever self.base_lin_vel[:, 0] is updated.
        #     # base_lin_vel_x_temp = self.base_lin_vel_x_low_freq.clone()
        #     # base_lin_vel_y_temp = self.base_lin_vel_y_low_freq.clone()
        #     # base_lin_vel_z_temp = self.base_lin_vel_z_low_freq.clone()
        #     # self.last_base_lin_vel_x = base_lin_vel_x_temp
        #     # self.last_base_lin_vel_y = base_lin_vel_y_temp
        #     # self.last_base_lin_vel_z = base_lin_vel_z_temp
        #     vx_plane_temp = self.vx_plane.clone()
        #     vz_world_temp = self.vz_world.clone()
        #     self.last_vx_plane = vx_plane_temp
        #     self.last_vz_world = vz_world_temp

        if PLOT_PITCH:
            # 1. Compute raw a_x, a_z
            ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) / (1 / self.linvel_update_actual_freq)
            az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) / (1 / self.linvel_update_actual_freq)
            # ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) / self.dt
            # az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) / self.dt
            
            
            # 2. Exponential smoothing
            self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
            self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
            
            # 3. Use the filtered values
            ax_smooth = self.ax_filtered * self.ax_scale
            az_smooth = self.az_filtered * self.az_scale

            desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
            desired_pitch_degrees = torch.rad2deg(desired_pitch)

            # Convert tensors to numbers (if using PyTorch)
            # base_lin_vel_x = self.base_lin_vel_x_low_freq.item()
            # base_lin_vel_z = self.base_lin_vel_z_low_freq.item()
            # last_base_lin_vel_x = self.last_base_lin_vel_x.item()
            # last_base_lin_vel_z = self.last_base_lin_vel_z.item()
            # ax_val = ax.item()
            # az_val = az.item()
            desired_pitch_degrees = desired_pitch_degrees.item()
            rot_y = self.rot_y_deg.item()

            # print("current: ", base_lin_vel_x)
            # print("last: ", last_base_lin_vel_x)

            # Store values with a limit to prevent memory overload
            # self.append_limited(self.base_lin_vel_x_list, base_lin_vel_x)
            # self.append_limited(self.base_lin_vel_z_list, base_lin_vel_z)
            # self.append_limited(self.last_base_lin_vel_x_list, last_base_lin_vel_x)
            # self.append_limited(self.last_base_lin_vel_z_list, last_base_lin_vel_z)
            # self.append_limited(self.ax_list, ax_val)
            # self.append_limited(self.az_list, az_val)
            self.append_limited(self.desired_pitch_list, desired_pitch_degrees)
            self.append_limited(self.current_pitch_list, rot_y)
            self.append_limited(self.time_steps, self.a_count)

            # # Only update the plot every 10 iterations for performance
            # if self.a_count % 10 == 0:
            # self.update_plot()
            self.update_plot_pitch()

            if self.a_count == 10000:
                self.show_plot()
        
        if PLOT_ACC:
            # 1. Compute raw a_x, a_z
            ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) / (1 / self.linvel_update_actual_freq)
            az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) / (1 / self.linvel_update_actual_freq)
            # ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) / self.dt
            # az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) / self.dt
            
            # 2. Exponential smoothing
            self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
            self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
            
            # 3. Use the filtered values
            ax_smooth = self.ax_filtered * self.ax_scale
            az_smooth = self.az_filtered * self.az_scale

            desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
            desired_pitch_degrees = torch.rad2deg(desired_pitch)

            ax_val = ax.item()
            az_val = az.item()

            self.append_limited(self.ax_list, ax_val)
            self.append_limited(self.az_list, az_val)
            self.append_limited(self.time_steps, self.a_count)

            self.update_plot_acc()

            if self.a_count == 10000:
                self.show_plot()
        
        if PLOT_ERROR:
            # 1. Compute raw a_x, a_z
            ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) / (1 / self.linvel_update_actual_freq)
            az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) / (1 / self.linvel_update_actual_freq)
            # ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) / self.dt
            # az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) / self.dt
            
            # 2. Exponential smoothing
            self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
            self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
            
            # 3. Use the filtered values
            ax_smooth = self.ax_filtered * self.ax_scale
            az_smooth = self.az_filtered * self.az_scale

            desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
            desired_pitch_degrees = torch.rad2deg(desired_pitch)

            desired_pitch_degrees = desired_pitch_degrees.item()
            rot_y = self.rot_y_deg.item()
            error = abs(desired_pitch_degrees - rot_y)

            print("Desired_pitch_degrees: ", desired_pitch_degrees)
            print("Current pitch degrees: ", rot_y)
            print("Pitch Error: ", error)

            self.append_limited(self.error_pitch_list, error)
            self.append_limited(self.time_steps, self.a_count)

            self.upate_plot_error()

            if self.a_count == 10000:
                self.show_plot()
        
        if PLOT_TILT_ERROR_VEL_ACC_HEIGHT_CMDREC:
            # 1. Compute raw a_x, a_z
            ax = (self.vx_plane - self.last_vx_plane) / self.dt
            ay = (self.vy_plane - self.last_vy_plane) / self.dt
            az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

            # print("raw ax: ", ax)
            # self.t += self.dt
            # print("Times step: ", self.t)
            # print("Ax plane: ", ax)
            # print("Az world: ", az)

            # 2. Exponential smoothing
            self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
            self.ay_filtered = self.alpha * self.ay_filtered + (1.0 - self.alpha) * ay
            self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
            

            # 3. Use the filtered values
            ax_smooth = self.ax_filtered * self.ax_scale
            ay_smooth = self.ay_filtered
            az_smooth = self.az_filtered * self.az_scale

            # print("ax: ", ax_smooth)

            # desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
            desired_pitch = torch.atan2(ax_smooth, -az_smooth)
            desired_pitch_degrees = torch.rad2deg(desired_pitch)
            # print("Before: ", desired_pitch_degrees)
            # desired_pitch_degrees = -desired_pitch_degrees
            # print("After: ", desired_pitch_degrees)
            desired_roll = torch.atan2(-ay_smooth, -az_smooth)
            desired_roll_degrees = torch.rad2deg(desired_roll)

            # print("Desired tilt [deg]: ", desired_pitch_degrees)
            # print("Current tilt [deg]: ", self.rot_y_deg)

            desired_pitch_degrees = desired_pitch_degrees.item()
            rot_y = self.rot_y_deg.item()
            error_pitch = abs(desired_pitch_degrees - rot_y)
            desired_roll_degrees = desired_roll_degrees.item()
            rot_x = self.rot_x_deg.item()
            error_roll = abs(desired_roll_degrees - rot_x)
            linvel_x = self.vx_plane.item()
            last_linvel_x = self.last_vx_plane.item()
            height = self.base_pos[:, 2].item()

            # print("Tilt Error: ", error)


            self.append_limited(self.desired_pitch_list, desired_pitch_degrees)
            self.append_limited(self.current_pitch_list, rot_y)
            self.append_limited(self.error_pitch_list, error_pitch)
            self.append_limited(self.desired_roll_list, desired_roll_degrees)
            self.append_limited(self.current_roll_list, rot_x)
            self.append_limited(self.error_roll_list, error_roll)
            self.append_limited(self.lin_vel_x_list, linvel_x)
            self.append_limited(self.last_lin_vel_x_list, last_linvel_x)
            self.append_limited(self.ax_list, ax_smooth)
            self.append_limited(self.az_list, az_smooth)
            self.append_limited(self.heigh_list, height)
            self.append_limited(self.time_steps, self.a_count)

            if self.a_count == self.plot_save_len:
                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/pitch"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/pitch"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/pitch"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/pitch"
                # file_name = self.folder_name + "_tilt.png"
                file_name = f"{self.folder_name}_run{self.run_num}_pitch.png"
                pitch_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/pitch_error"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/pitch_error"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/pitch_error"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/pitch_error"
                # file_name = self.folder_name + "_error.png"
                file_name = f"{self.folder_name}_run{self.run_num}_pitch_error.png"
                pitch_error_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/roll"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/roll"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/roll"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/roll"
                # file_name = self.folder_name + "_tilt.png"
                file_name = f"{self.folder_name}_run{self.run_num}_roll.png"
                roll_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/roll_error"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/roll_error"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/roll_error"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/roll_error"
                # file_name = self.folder_name + "_error.png"
                file_name = f"{self.folder_name}_run{self.run_num}_roll_error.png"
                roll_error_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/linvel_x_check"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/linvel_x_check"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/linvel_x_check"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/linvel_x_check"
                # file_name = self.folder_name + "_linvel_x.png"
                file_name = f"{self.folder_name}_run{self.run_num}_linvel_x_check.png"
                linvel_x_check_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/acc"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/acc"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/acc"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/acc"
                # file_name = self.folder_name + "_acc.png"
                file_name = f"{self.folder_name}_run{self.run_num}_acc.png"
                acc_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/height"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/height"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/height"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/height"
                # file_name = self.folder_name + "_height.png"
                file_name = f"{self.folder_name}_run{self.run_num}_height.png"
                height_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/cmd_rec"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/cmd_rec"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/cmd_rec"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/cmd_rec"
                # file_name = self.folder_name + "_cmd_rec.txt"
                file_name = f"{self.folder_name}_run{self.run_num}_cmd_rec.txt"
                cmd_rec_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/pitch_stats"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/pitch_stats"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/pitch_stats"
                    else:
                        # base_path = "/home/psxkf4/Genesis/logs/paper/data/pitch_stats"
                        base_path = f"/home/psxkf4/Genesis/results/{self.folder_name}/pitch_stats"
                os.makedirs(base_path, exist_ok=True)
                file_name = f"{self.folder_name}_run{self.run_num}_pitch_stats.txt"
                pitch_stats_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/roll_stats"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/roll_stats"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/roll_stats"
                    else:
                        # base_path = "/home/psxkf4/Genesis/logs/paper/data/pitch_stats"
                        base_path = f"/home/psxkf4/Genesis/results/{self.folder_name}/roll_stats"
                os.makedirs(base_path, exist_ok=True)
                file_name = f"{self.folder_name}_run{self.run_num}_roll_stats.txt"
                roll_stats_path = os.path.join(base_path, file_name)
                
                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/linvel_x"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/linvel_x"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/linvel_x"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/linvel_x"

                file_name = f"{self.folder_name}_run{self.run_num}_linvel_x.png"
                linvel_x_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/linvel_x_error"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/linvel_x_error"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/linvel_x_error"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/linvel_x_error"

                file_name = f"{self.folder_name}_run{self.run_num}_linvel_x_error.png"
                linvel_x_error_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/linvel_x_stats"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/linvel_x_stats"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/linvel_x_stats"
                    else:
                        # base_path = "/home/psxkf4/Genesis/logs/paper/data/linvel_x_stats"
                        base_path = f"/home/psxkf4/Genesis/results/{self.folder_name}/linvel_x_stats"
                os.makedirs(base_path, exist_ok=True)
                file_name = f"{self.folder_name}_run{self.run_num}_linvel_x_stats.txt"
                linvel_x_stats_path = os.path.join(base_path, file_name)


                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/linvel_y"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/linvel_y"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/linvel_y"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/linvel_y"

                file_name = f"{self.folder_name}_run{self.run_num}_linvel_y.png"
                linvel_y_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/linvel_y_error"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/linvel_y_error"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/linvel_y_error"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/linvel_y_error"

                file_name = f"{self.folder_name}_run{self.run_num}_linvel_y_error.png"
                linvel_y_error_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/linvel_y_stats"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/linvel_y_stats"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/linvel_y_stats"
                    else:
                        # base_path = "/home/psxkf4/Genesis/logs/paper/data/linvel_y_stats"
                        base_path = f"/home/psxkf4/Genesis/results/{self.folder_name}/linvel_y_stats"
                os.makedirs(base_path, exist_ok=True)
                file_name = f"{self.folder_name}_run{self.run_num}_linvel_y_stats.txt"
                linvel_y_stats_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/angvel_z"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/angvel_z"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/angvel_z"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/angvel_z"

                file_name = f"{self.folder_name}_run{self.run_num}_angvel_z.png"
                angvel_z_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/angvel_z_error"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/angvel_z_error"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/angvel_z_error"
                    else:
                        base_path = "/home/psxkf4/Genesis/logs/paper/data/angvel_z_error"

                file_name = f"{self.folder_name}_run{self.run_num}_angvel_z_error.png"
                angvel_z_error_path = os.path.join(base_path, file_name)

                if ALIENWARE:
                    if self.test_plot:
                        base_path = "/home/alienware/koyo_ws/Genesis/test_data/angvel_z_stats"
                    else:
                        base_path = "/home/alienware/koyo_ws/Genesis/Slosh-Free-Go2-Logs/data/angvel_z_stats"
                else:
                    if self.test_plot:
                        base_path = "/home/psxkf4/Genesis/logs/paper/test_data/angvel_z_stats"
                    else:
                        # base_path = "/home/psxkf4/Genesis/logs/paper/data/angvel_z_stats"
                        base_path = f"/home/psxkf4/Genesis/results/{self.folder_name}/angvel_z_stats"
                os.makedirs(base_path, exist_ok=True)
                file_name = f"{self.folder_name}_run{self.run_num}_angvel_z_stats.txt"
                angvel_z_stats_path = os.path.join(base_path, file_name)

                self.update_eval_plot(pitch_path, pitch_error_path, linvel_x_check_path, 
                                      acc_path, height_path, cmd_rec_path, pitch_stats_path,
                                      linvel_x_path, linvel_x_error_path, linvel_x_stats_path,
                                      linvel_y_path, linvel_y_error_path, linvel_y_stats_path,
                                      angvel_z_path, angvel_z_error_path, angvel_z_stats_path,
                                      roll_path, roll_error_path, roll_stats_path)

                # if self.a_count == 10000:
                #     self.show_plot()
        

    def to_numpy(self, data):
        """Convert a tensor or a list of tensors to a NumPy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()  # Move tensor to CPU and convert to NumPy
        
        if isinstance(data, list):  # If it's a list, check if it contains tensors
            return np.array([x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in data])

        return np.array(data)  # Convert normal lists to NumPy

    def update_plot_pitch(self):
        self.axs.cla()

        # Convert to NumPy before plotting
        desired_theta_np = self.to_numpy(self.desired_pitch_list)
        current_pitch_np = self.to_numpy(self.current_pitch_list)

        self.axs.plot(self.time_steps, desired_theta_np, label="Desired", color="g")
        self.axs.plot(self.time_steps, current_pitch_np, label="Current", color="m")
        
        self.axs.set_xlabel("Time Steps")
        self.axs.set_ylabel("Pitch")
        self.axs.legend()
        self.axs.set_title("Pitch [degrees]: When forward[backward] start, should be negative[positive]. When forward[backward] stop, should be positive[negative].")

        plt.pause(0.01)  # Refresh the plot in real-time
    
    def update_plot_acc(self):
        self.axs.cla()
        self.axs.plot(self.time_steps, self.ax_list, label="Ax", color="g")
        self.axs.plot(self.time_steps, self.az_list, label="Az", color="m")
        self.axs.set_xlabel("Time Steps")
        self.axs.set_ylabel("Acceleration")
        self.axs.legend()
        self.axs.set_title("Accelerations")

        plt.pause(0.01)  # Refresh the plot in real-time
    
    def upate_plot_error(self):
        self.axs.cla()
        self.axs.plot(self.time_steps, self.error_pitch_list, label="Pitch Error", color="r")
        self.axs.set_xlabel("Time Steps")
        self.axs.set_ylabel("Error [degrees]")
        self.axs.legend()
        self.axs.set_title("Pitch Error")

        plt.pause(0.01)  # For real-time updates

    def update_eval_plot(self, pitch_path=None, pitch_error_path=None,linvel_x_check_path=None, 
                         acc_path=None, height_path=None, cmd_rec_path=None, pitch_stats_path=None,
                         linvel_x_path=None, linvel_x_error_path=None, linvel_x_stats_path=None,
                         linvel_y_path=None, linvel_y_error_path=None, linvel_y_stats_path=None,
                         angvel_z_path=None, angvel_z_error_path=None, angvel_z_stats_path=None,
                         roll_path=None, roll_error_path=None, roll_stats_path=None):
        
        tick_durations = [
            self.start_stop_cmd_len,
            self.forward1_cmd_len,
            self.forward2_cmd_len,
            self.forward3_cmd_len,
            self.middle_stop_cmd_len,
            self.backward1_cmd_len,
            self.backward2_cmd_len,
            self.backward3_cmd_len,
            self.finish_stop_cmd_len
        ]
        tick_positions = np.cumsum(tick_durations)
        tick_positions = np.insert(tick_positions, 0, 0)
        tick_labels = [
            "Start",
            "Fwd1",
            "Fwd2",
            "Fwd3",
            "Stop",
            "Bwd1",
            "Bwd2",
            "Bwd3",
            "End",
            "Finish"
        ]
        
        self.axs1.cla()

        # Convert to NumPy before plotting
        desired_pitch_np = self.to_numpy(self.desired_pitch_list)
        current_pitch_np = self.to_numpy(self.current_pitch_list)

        self.axs1.plot(self.time_steps, desired_pitch_np, label="Desired", color="g", linestyle='--')
        self.axs1.plot(self.time_steps, current_pitch_np, label="Current", color="m")
        self.axs1.set_xlabel("Time Steps")
        self.axs1.set_ylabel("Pitch [degrees]")
        self.axs1.set_ylim(-14.0, 14.0)
        self.axs1.set_xticks(tick_positions)
        # self.axs1.set_xticklabels(tick_labels)
        # self.axs1.tick_params(axis='x', labelsize=6)
        self.axs1.set_xticklabels([])   
        self.axs1.legend()
        # self.axs1.set_title("Desired and Current Pitch")

        self.axs2.cla()
        self.axs2.plot(self.time_steps, self.error_pitch_list, label="Pitch Error", color="r")
        self.axs2.set_xlabel("Time Steps")
        self.axs2.set_ylabel("Pitch Error [degrees]")
        self.axs2.set_ylim(-2.0, 10.0)
        self.axs2.set_xticks(tick_positions)
        self.axs2.set_xticklabels([]) 
        self.axs2.legend()
        # self.axs2.set_title("Pitch Error")

        # Convert to NumPy before plotting
        lin_vel_x_np = self.to_numpy(self.lin_vel_x_list)
        last_lin_vel_x_np = self.to_numpy(self.last_lin_vel_x_list)

        self.axs3.cla()
        self.axs3.plot(self.time_steps, lin_vel_x_np, label="Current", color="r", linestyle='--')
        self.axs3.plot(self.time_steps, last_lin_vel_x_np, label="Last", color="b")
        self.axs3.set_xlabel("Time Steps")
        self.axs3.set_ylabel("Velocity [m/s]")
        self.axs3.set_ylim(-5.0, 5.0)
        self.axs3.set_xticks(tick_positions)
        self.axs3.set_xticklabels([])
        self.axs3.legend()

        # Convert to NumPy before plotting
        ax_np = self.to_numpy(self.ax_list)
        az_np = self.to_numpy(self.az_list)

        self.axs4.cla()
        self.axs4.plot(self.time_steps, ax_np.squeeze(), label="Acc x", color="b", linestyle='--')
        self.axs4.plot(self.time_steps, az_np.squeeze(), label="Acc z", color="y")
        self.axs4.set_xlabel("Time Steps")
        self.axs4.set_ylabel("Acc [m/s^2]")
        self.axs4.set_ylim(-14.0, 6.0)
        self.axs4.set_xticks(tick_positions)
        self.axs4.set_xticklabels([])
        self.axs4.legend()

        self.axs5.cla()
        self.axs5.plot(self.time_steps, self.heigh_list, label="Height", color="k")
        self.axs5.set_xlabel("Time Steps")
        self.axs5.set_ylabel("Base Height [m]")
        self.axs5.set_ylim(0.0, 0.75)
        self.axs5.set_xticks(tick_positions)
        self.axs5.set_xticklabels([])
        self.axs5.legend()


        # Convert to NumPy before plotting
        command_linvel_x_np = self.to_numpy(self.command_linvel_x_list)
        current_linvel_x_np = self.to_numpy(self.current_linvel_x_list)
        linvel_x_error_np = self.to_numpy(self.linvel_x_error_list)
        command_linvel_y_np = self.to_numpy(self.command_linvel_y_list)
        current_linvel_y_np = self.to_numpy(self.current_linvel_y_list)
        linvel_y_error_np = self.to_numpy(self.linvel_x_error_list)
        command_angvel_z_np = self.to_numpy(self.command_angvel_z_list)
        current_angvel_z_np = self.to_numpy(self.current_angvel_z_list)
        angvel_z_error_np = self.to_numpy(self.linvel_x_error_list)

        self.axs6.cla()
        self.axs6.plot(self.time_steps, command_linvel_x_np, label="Command Linear X Velocity", color="b", linestyle='--')
        self.axs6.plot(self.time_steps, current_linvel_x_np, label="Current Linear X Velocity", color="c")
        self.axs6.set_xlabel("Time Steps")
        self.axs6.set_ylabel("Linear X Velocity [m/s]")
        self.axs6.set_ylim(-1.5, 1.5)
        self.axs6.set_xticks(tick_positions)
        self.axs6.set_xticklabels([])   
        self.axs6.legend()

        self.axs7.cla()
        self.axs7.plot(self.time_steps, linvel_x_error_np, label="Linear X Velocity Error", color="r")
        self.axs7.set_xlabel("Time Steps")
        self.axs7.set_ylabel("Linear X Velocity Error [m/s]")
        self.axs7.set_ylim(-1.0, 2.0)
        self.axs7.set_xticks(tick_positions)
        self.axs7.set_xticklabels([]) 
        self.axs7.legend()

        self.axs8.cla()
        self.axs8.plot(self.time_steps, command_linvel_y_np, label="Command Linear Y Velocity", color="b", linestyle='--')
        self.axs8.plot(self.time_steps, current_linvel_y_np, label="Current Linear Y Velocity", color="c")
        self.axs8.set_xlabel("Time Steps")
        self.axs8.set_ylabel("Linear Y Velocity [m/s]")
        self.axs6.set_ylim(-1.5, 1.5)
        self.axs8.set_xticks(tick_positions)
        self.axs8.set_xticklabels([])   
        self.axs8.legend()

        self.axs9.cla()
        self.axs9.plot(self.time_steps, linvel_y_error_np, label="Linear Y Velocity Error", color="r")
        self.axs9.set_xlabel("Time Steps")
        self.axs9.set_ylabel("Linear Y Velocity Error [m/s]")
        self.axs9.set_ylim(-1.0, 2.0)
        self.axs9.set_xticks(tick_positions)
        self.axs9.set_xticklabels([]) 
        self.axs9.legend()

        self.axs10.cla()
        self.axs10.plot(self.time_steps, command_angvel_z_np, label="Command Angular Z Velocity", color="b", linestyle='--')
        self.axs10.plot(self.time_steps, current_angvel_z_np, label="Current Angular Z Velocity", color="c")
        self.axs10.set_xlabel("Time Steps")
        self.axs10.set_ylabel("Angular Z Velocity [m/s]")
        self.axs6.set_ylim(-1.5, 1.5)
        self.axs10.set_xticks(tick_positions)
        self.axs10.set_xticklabels([])   
        self.axs10.legend()

        self.axs11.cla()
        self.axs11.plot(self.time_steps, angvel_z_error_np, label="Angular Z Velocity Error", color="r")
        self.axs11.set_xlabel("Time Steps")
        self.axs11.set_ylabel("Angular Z Velocity Error [m/s]")
        self.axs11.set_ylim(-1.0, 2.0)
        self.axs11.set_xticks(tick_positions)
        self.axs11.set_xticklabels([]) 
        self.axs11.legend()

        # Convert to NumPy before plotting
        desired_roll_np = self.to_numpy(self.desired_roll_list)
        current_roll_np = self.to_numpy(self.current_roll_list)

        self.axs12.plot(self.time_steps, desired_roll_np, label="Desired", color="g", linestyle='--')
        self.axs12.plot(self.time_steps, current_roll_np, label="Current", color="m")
        self.axs12.set_xlabel("Time Steps")
        self.axs12.set_ylabel("Roll [degrees]")
        self.axs12.set_ylim(-14.0, 14.0)
        self.axs12.set_xticks(tick_positions)
        # self.axs1.set_xticklabels(tick_labels)
        # self.axs1.tick_params(axis='x', labelsize=6)
        self.axs12.set_xticklabels([])   
        self.axs12.legend()
        # self.axs1.set_title("Desired and Current Pitch")

        self.axs13.cla()
        self.axs13.plot(self.time_steps, self.error_roll_list, label="Roll Error", color="r")
        self.axs13.set_xlabel("Time Steps")
        self.axs13.set_ylabel("Roll Error [degrees]")
        self.axs13.set_ylim(-2.0, 10.0)
        self.axs13.set_xticks(tick_positions)
        self.axs13.set_xticklabels([]) 
        self.axs13.legend()
        # self.axs2.set_title("Pitch Error")


        # Compute stats for pitch error
        pitch_error_np = self.to_numpy(self.error_pitch_list)
        pitch_mean_error = pitch_error_np.mean()
        pitch_std_error = pitch_error_np.std()
        pitch_max_error = pitch_error_np.max()

        # Compute stats for roll error
        roll_error_np = self.to_numpy(self.error_roll_list)
        roll_mean_error = roll_error_np.mean()
        roll_std_error = roll_error_np.std()
        roll_max_error = roll_error_np.max()

        linvel_x_error_np = self.to_numpy(self.linvel_x_error_list)
        linvel_x_mean_error = linvel_x_error_np.mean()
        linvel_x_std_error = linvel_x_error_np.std()
        linvel_x_max_error = linvel_x_error_np.max()

        linvel_y_error_np = self.to_numpy(self.linvel_y_error_list)
        linvel_y_mean_error = linvel_y_error_np.mean()
        linvel_y_std_error = linvel_y_error_np.std()
        linvel_y_max_error = linvel_y_error_np.max()

        angvel_z_error_np = self.to_numpy(self.angvel_z_error_list)
        angvel_z_mean_error = angvel_z_error_np.mean()
        angvel_z_std_error = angvel_z_error_np.std()
        angvel_z_max_error = angvel_z_error_np.max()

        if cmd_rec_path:
            with open(cmd_rec_path, "w") as f:
                f.write(f"Randomly Sampled Command V Length Run {self.run_num}\n")
                f.write(f"-----------------------\n")
                f.write(f"Start Stop Length: {self.start_stop_cmd_len}\n")
                f.write(f"Forward 1 Length: {self.forward1_cmd_len}\n")
                f.write(f"Forward 1 Velocity: {self.forward1_cmd_vel}\n")
                f.write(f"Forward 2 Length: {self.forward2_cmd_len}\n")
                f.write(f"Forward 2 Velocity: {self.forward2_cmd_vel}\n")
                f.write(f"Forward 3 Length: {self.forward3_cmd_len}\n")
                f.write(f"Forward 3 Velocity: {self.forward3_cmd_vel}\n")
                f.write(f"Middle Stop Length: {self.middle_stop_cmd_len}\n")
                f.write(f"Backward 1 Length: {self.backward1_cmd_len}\n")
                f.write(f"Backward 1 Velocity: {self.backward1_cmd_vel}\n")
                f.write(f"Backward 2 Length: {self.backward2_cmd_len}\n")
                f.write(f"Backward 2 Velocity: {self.backward2_cmd_vel}\n")
                f.write(f"Backward 3 Length: {self.backward3_cmd_len}\n")
                f.write(f"Backward 3 Velocity: {self.backward3_cmd_vel}\n")
                f.write(f"Finish Stop Length: {self.finish_stop_cmd_len}\n")
                f.write(f"\n")
                f.write(f"Total Step Length: {self.plot_save_len}")
            print(f"Command length saved to {cmd_rec_path}")

        # Save stats to text file
        if pitch_stats_path:
            with open(pitch_stats_path, "w") as f:
                f.write(f"Pitch Error Statistics Run {self.run_num}\n")
                f.write(f"-----------------------\n")
                f.write(f"Mean (μ_{self.run_num}): {pitch_mean_error:.6f}\n")
                f.write(f"Standard Deviation (σ_{self.run_num}): {pitch_std_error:.6f}\n")
                f.write(f"Maximum (M_{self.run_num}): {pitch_max_error:.6f}\n")
                f.write(f"\n")
                f.write(f"Total Step Length (n_{self.run_num}): {self.plot_save_len}")
            print(f"Pitch error stats saved to {pitch_stats_path}")
        
        if roll_stats_path:
            with open(roll_stats_path, "w") as f:
                f.write(f"Roll Error Statistics Run {self.run_num}\n")
                f.write(f"-----------------------\n")
                f.write(f"Mean (μ_{self.run_num}): {roll_mean_error:.6f}\n")
                f.write(f"Standard Deviation (σ_{self.run_num}): {roll_std_error:.6f}\n")
                f.write(f"Maximum (M_{self.run_num}): {roll_max_error:.6f}\n")
                f.write(f"\n")
                f.write(f"Total Step Length (n_{self.run_num}): {self.plot_save_len}")
            print(f"Roll error stats saved to {roll_stats_path}")

        if linvel_x_stats_path:
            with open(linvel_x_stats_path, "w") as f:
                f.write(f"Linear X Velocity Error Statistics Run {self.run_num}\n")
                f.write(f"-----------------------\n")
                f.write(f"Mean (μ_{self.run_num}): {linvel_x_mean_error:.6f}\n")
                f.write(f"Standard Deviation (σ_{self.run_num}): {linvel_x_std_error:.6f}\n")
                f.write(f"Maximum (M_{self.run_num}): {linvel_x_max_error:.6f}\n")
                f.write(f"\n")
                f.write(f"Total Step Length (n_{self.run_num}): {self.plot_save_len}")
            print(f"Linear x velocity error stats saved to {linvel_x_stats_path}")
        
        if linvel_y_stats_path:
            with open(linvel_y_stats_path, "w") as f:
                f.write(f"Linear Y Velocity Error Statistics Run {self.run_num}\n")
                f.write(f"-----------------------\n")
                f.write(f"Mean (μ_{self.run_num}): {linvel_y_mean_error:.6f}\n")
                f.write(f"Standard Deviation (σ_{self.run_num}): {linvel_y_std_error:.6f}\n")
                f.write(f"Maximum (M_{self.run_num}): {linvel_y_max_error:.6f}\n")
                f.write(f"\n")
                f.write(f"Total Step Length (n_{self.run_num}): {self.plot_save_len}")
            print(f"Linear y velocity error stats saved to {linvel_y_stats_path}")
        
        if angvel_z_stats_path:
            with open(angvel_z_stats_path, "w") as f:
                f.write(f"Angular Z Velocity Error Statistics Run {self.run_num}\n")
                f.write(f"-----------------------\n")
                f.write(f"Mean (μ_{self.run_num}): {angvel_z_mean_error:.6f}\n")
                f.write(f"Standard Deviation (σ_{self.run_num}): {angvel_z_std_error:.6f}\n")
                f.write(f"Maximum (M_{self.run_num}): {angvel_z_max_error:.6f}\n")
                f.write(f"\n")
                f.write(f"Total Step Length (n_{self.run_num}): {self.plot_save_len}")
            print(f"Angular z velocity error stats saved to {angvel_z_stats_path}")

        if pitch_path and pitch_error_path and roll_path and roll_error_path and linvel_x_check_path and acc_path and height_path and linvel_x_path and linvel_x_error_path and linvel_y_path and linvel_y_error_path and angvel_z_path and angvel_z_error_path:
            print("PLOTS ARE SAVED")
            self.fig1.savefig(pitch_path, dpi=300, bbox_inches='tight')
            self.fig2.savefig(pitch_error_path, dpi=300, bbox_inches='tight')
            self.fig3.savefig(linvel_x_check_path, dpi=300, bbox_inches='tight')
            self.fig4.savefig(acc_path, dpi=300, bbox_inches='tight')
            self.fig5.savefig(height_path, dpi=300, bbox_inches='tight')
            self.fig6.savefig(linvel_x_path, dpi=300, bbox_inches='tight')
            self.fig7.savefig(linvel_x_error_path, dpi=300, bbox_inches='tight')
            self.fig8.savefig(linvel_y_path, dpi=300, bbox_inches='tight')
            self.fig9.savefig(linvel_y_error_path, dpi=300, bbox_inches='tight')
            self.fig10.savefig(angvel_z_path, dpi=300, bbox_inches='tight')
            self.fig11.savefig(angvel_z_error_path, dpi=300, bbox_inches='tight')
            self.fig12.savefig(roll_path, dpi=300, bbox_inches='tight')
            self.fig13.savefig(roll_error_path, dpi=300, bbox_inches='tight')

        plt.pause(0.01)

    def show_plot(self):
        plt.ioff()
        plt.show()

    def check_and_sanitize_observations(self):
        """
        Detect NaN/Inf in self.obs_buf / self.privileged_obs_buf.
        Reset those environments and replace only the rows with NaN/Inf values.
        """
        # 1) Find which envs have NaN or Inf in either buffer.
        bad_envs = torch.any(~torch.isfinite(self.obs_buf), dim=1) | torch.any(~torch.isfinite(self.privileged_obs_buf), dim=1)

        if bad_envs.any():
            num_bad = bad_envs.sum().item()
            print(f"WARNING: {num_bad} envs have invalid observations -> resetting them.")
            
            # Find the indices of NaN values in self.obs_buf
            nan_indices = torch.isnan(self.obs_buf).nonzero(as_tuple=False)
            # for idx in nan_indices:
            #     env_idx, obs_idx = idx
            #     print(f"NaN detected at env {env_idx}, observation {obs_idx}: {self.obs_buf[env_idx, obs_idx]}")
            #     print(f"base pose {self.base_pos[env_idx]}")
            
            # Reset those environments
            # self.reset_idx(bad_envs.nonzero(as_tuple=False).flatten())

            # 2) Replace rows with NaN values in obs_buf and privileged_obs_buf
            for env_idx in bad_envs.nonzero(as_tuple=False).flatten():
                self.random_pos[env_idx] = self.random_pos[0]
                self.obs_buf[env_idx] =  copy.deepcopy(self.zero_obs)
                self.privileged_obs_buf[env_idx] =  copy.deepcopy(self.zero_privileged_obs)

    def check_and_reset_observations(self):
        """
        Detect NaN/Inf in self.obs_buf / self.privileged_obs_buf.
        If found, mark those environments as done so that they get reset.
        Optionally replace the offending observations to prevent downstream issues.
        """
        # 1) Identify environments with any NaN or Inf entries
        bad_envs = torch.any(~torch.isfinite(self.obs_buf), dim=1) \
                | torch.any(~torch.isfinite(self.privileged_obs_buf), dim=1)

        # 2) If we find any invalid entries, handle them
        if bad_envs.any():
            num_bad = bad_envs.sum().item()
            print(f"WARNING: {num_bad} envs have invalid observations -> resetting them.")

            # Mark these environments as done so they will be reset at the next step call
            self.reset_buf[bad_envs] = True

            # Optionally, overwrite their observations to avoid propagating NaNs
            self.obs_buf[bad_envs] = self.zero_obs
            self.privileged_obs_buf[bad_envs] = self.zero_privileged_obs


    def compute_rewards(self):
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            
            # print("reward name: ", name)
            rew = reward_func() * self.reward_scales[name]
            # print("reward: ", rew)

            if name == "tracking_lin_vel":
                # print("tracking lin vel reward: ", rew)
                # if torch.all(rew > self.reward_scales["tracking_lin_vel"] * 0.7):
                if torch.mean(rew) > (self.reward_scales["tracking_lin_vel"] * 0.7):
                    self.switch_resample = True
                
                # breakpoint()
    
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.reward_cfg["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
        # self.resample_updated = False

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf
    
    def get_critic_observations(self):
        return self.privileged_obs_buf
    
    def get_domain_randomizations(self):
        return self.domain_randomizations_buf
    
    def get_observations_history(self):
        return self.obs_history_buf
    



    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.noise_cfg["add_noise"]
        noise_level =self.noise_cfg["noise_level"]
        noise_vec[:3] = self.noise_scales["ang_vel"] * noise_level * self.obs_scales["ang_vel"]
        noise_vec[3:6] = self.noise_scales["gravity"] * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = self.noise_scales["dof_pos"] * noise_level * self.obs_scales["dof_pos"]
        noise_vec[9+self.num_actions:9+2*self.num_actions] = self.noise_scales["dof_vel"] * noise_level * self.obs_scales["dof_vel"]
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+8] = 0. # sin/cos phase
        return noise_vec


    def check_termination(self):
        """Check if environments need to be reset."""
        # (n_envs, n_links, 3) tensor of net contact forces
        contact_threshold_exceeded = (torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        ) > 1.0)
        # For each environment, if ANY contact index exceeds force threshold, treat it as contact
        in_contact = torch.any(contact_threshold_exceeded, dim=1)
        self.contact_duration_buf[in_contact] += self.dt
        self.reset_buf = self.contact_duration_buf > self.env_cfg["termination_duration"]
        #pitch and roll degree exceed termination
        if not self.termination_exceed_degree_ignored: 
            self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.termination_if_pitch_greater_than_value
            self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.termination_if_roll_greater_than_value
        # Timeout termination
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg['termination_if_height_lower_than']
        # -------------------------------------------------------
        #  Add out-of-bounds check using terrain_min_x, etc.
        # -------------------------------------------------------
        # min_x, max_x, min_y, max_y = self.terrain_bounds  # or however you store them
        
        # We assume base_pos[:, 0] is x, base_pos[:, 1] is y
        if self.terrain_type != "plane":
            self.out_of_bounds_buf = (
                (self.base_pos[:, 0] < self.terrain_min_x) |
                (self.base_pos[:, 0] > self.terrain_max_x) |
                (self.base_pos[:, 1] < self.terrain_min_y) |
                (self.base_pos[:, 1] > self.terrain_max_y)
            )
            self.reset_buf |= self.out_of_bounds_buf
        # For those that are out of bounds, penalize by marking episode_length_buf = max
        # self.episode_length_buf[out_of_bounds] = self.max_episode_length

        # If an environment has been running longer than its maximum episode length 
        # (i.e. the number of steps exceeds self.max_episode_length), 
        # then it is marked for reset.
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        # Environments that have timed out are included in self.reset_buf.
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        # print("envs_idx in reset_idx function", envs_idx)
        # breakpoint()
        # # Reset episode step counter for the environments being reset
        # self.episode_step[envs_idx] = 0  # Ensures rewards transition properly

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.hip_pos[envs_idx] = self.default_hip_pos
        self.hip_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        # reset base
        # Check if the new_base_pos contains any NaNs
        # random_index = random.randrange(len(self.random_pos))
        # Randomly choose positions from pre-generated random_pos for each environment
        # random_indices = torch.randint(0, self.num_envs, (len(envs_idx),), device=self.device)
        self.base_pos[envs_idx] = self.random_pos[envs_idx] + self.base_init_pos
        if torch.isnan(self.base_pos[envs_idx]).any():
            print(f"WARNING: NaN detected in base_pos for envs {envs_idx}. Skipping assignment.")
        else:
            self.base_pos[envs_idx] = self.random_pos[0] + self.base_init_pos

        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        # print("self.base_quat[envs_idx]: ", self.base_quat[envs_idx])
        # self.base_quat[envs_idx]:  tensor([[1., 0., 0., 0.],
        # [1., 0., 0., 0.],
        # [1., 0., 0., 0.]], device='cuda:0')
        # breakpoint()
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        
        if RANDOM_INIT_ROT:
            if self.env_cfg["randomize_rot"]:
                # 1) Get random roll, pitch, yaw (in degrees) for each environment.
                roll = gs_rand_float(*self.env_cfg["roll_range"],  (len(envs_idx),), self.device)
                pitch = gs_rand_float(*self.env_cfg["pitch_range"], (len(envs_idx),), self.device)
                yaw = gs_rand_float(*self.env_cfg["yaw_range"],    (len(envs_idx),), self.device)

                # 2) Convert them all at once into a (N,4) quaternion tensor [x, y, z, w].
                quats_torch = quaternion_from_euler_tensor(roll, pitch, yaw)  # (N, 4)

                # 3) Move to CPU if needed and assign into self.base_quat in one shot
                #    (assuming self.base_quat is a numpy array of shape [num_envs, 4]).
                self.base_quat[envs_idx] = quats_torch
            else:
                self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)

        if DELAY:
            if self.env_cfg["randomize_delay"]:
                self.motor_delay_steps[envs_idx] = torch.randint(
                    int(self.min_delay / self.dt),
                    self.max_delay_steps + 1,
                    (len(envs_idx), self.num_actions),
                    device=self.device
                )

        # 1a. Check right after setting position
        if torch.isnan(self.base_pos[envs_idx]).any():
            print("NaN in base_pos right after setting it in reset_idx()")
            print("envs_idx:", envs_idx)
            print("base_pos:", self.base_pos[envs_idx])
            raise ValueError("NaNs in base_pos during reset.")

        # 1b. Check DOFs
        dof_pos = self.robot.get_dofs_position(self.motor_dofs)
        if torch.isnan(dof_pos[envs_idx]).any():
            print("NaN in dof_pos right after reset_idx()")
            print("envs_idx:", envs_idx)
            print("dof_pos:", dof_pos[envs_idx])
            raise ValueError("NaNs in dof_pos during reset.")


        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.last_base_lin_vel_x[envs_idx] = 0
        self.last_base_lin_vel_z[envs_idx] = 0

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.second_last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.feet_air_time[envs_idx] = 0.0
        self.feet_max_height[envs_idx] = 0.0
        self.reset_buf[envs_idx] = True
        self.contact_duration_buf[envs_idx] = 0.0

        if TRAJECTORY_RESAMPLE:
            # print("reset envs due to failure or timeout: ", envs_idx)
            max_episode_length_s = random.randint(self.env_cfg["min_T"], self.env_cfg["max_T"])
            self.max_episode_length_s[envs_idx] = max_episode_length_s
            self.max_episode_length[envs_idx] = np.ceil(max_episode_length_s / self.dt)

            self.xf[envs_idx] = self.base_pos[envs_idx].clone()
            # update_dis = random.uniform(self.env_cfg["min_delta_x"], self.env_cfg["max_delta_x"])
            # self.xf[envs_idx, 0] += update_dis
            update_dis_x = random.uniform(self.env_cfg["min_delta_x"], self.env_cfg["max_delta_x"])
            self.xf[envs_idx, 0] += update_dis_x
            update_dis_y = random.uniform(self.env_cfg["min_delta_x"], self.env_cfg["max_delta_x"])
            self.xf[envs_idx, 1] += update_dis_y
            self.traj_t[envs_idx] = 0.0
            
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # print("max_episode_length (T) is updated", self.max_episode_length)
            # print("xf is updated", self.xf[envs_idx])
            # print("traj_t is reset to zero")
            # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            # breakpoint()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
        if ACC_PROFILE_RESAMPLE:
            reset_flag = True
            self._resample_commands_gaussian_acc(envs_idx, reset_flag)
        elif ACC_PROFILE_RESAMPLE_V2:
            reset_flag = True
            self._resample_commands_gaussian_acc_v2(envs_idx, reset_flag)
        elif ACC_PROFILE_RESAMPLE_V3:
            reset_flag = True
            self._resample_commands_gaussian_acc_v3(envs_idx, reset_flag)
        elif DESIRED_PITCH_COMMAND:
            self._resample_desired_pitch(envs_idx)
        elif PREDEFINED_RESAMPLE_EVAL or PREDEFINED_RESAMPLE_TRY_EVAL:
            self._resample_predefined_commands()
        elif RANDOM_RESAMPLE_TRAIN:
            self._resample_random_commands_train()
        elif RANDOM_RESAMPLE_EVAL:
            # self._resample_random_commands_eval()
            pass
        elif TRAJECTORY_RESAMPLE:
            reset_flag = True
            self._resample_trajectory(envs_idx, reset_flag)
        elif MIX_RESAMPLE:
            if self.switch_resample:
                reset_flag = False
                self._resample_commands_gaussian_acc(envs_idx, reset_flag)
            else:
                self._resample_commands(envs_idx)
        else:
            self._resample_commands(envs_idx)
        if self.env_cfg['send_timeouts']:
            self.extras['time_outs'] = self.time_out_buf
        

    def generate_random_positions(self):
        """
        Use the _random_robot_position() method to generate unique random positions
        for each environment.
        """
        positions = torch.zeros((self.num_envs, 3), device=self.device)
        for i in range(self.num_envs):
            x, y, z = self._random_robot_position()
            # positions[i] = torch.tensor([0, 0, z], device=self.device)
            positions[i] = torch.tensor([x, y, z], device=self.device)
        return positions

    def generate_positions(self):
        """
        Use the _random_robot_position() method to generate unique random positions
        for each environment.
        """
        positions = torch.zeros((self.num_envs, 3), device=self.device)
        for i in range(self.num_envs):
            positions[i] = torch.tensor([0, 0, 0], device=self.device)
        return positions

    def _random_robot_position(self):
        # 1. Sample random row, col(a subterrain)
        # 0.775 ~ l2_norm(0.7, 0.31)
        # go2_size_xy = 0.775
        # row = np.random.randint(int((self.rows * self.terrain.subterrain_size[0]-go2_size_xy)/self.terrain.horizontal_scale))
        # col = np.random.randint(int((self.cols * self.terrain.subterrain_size[1]-go2_size_xy)/self.terrain.horizontal_scale))
        center = self.subterrain_centers[self.spawn_counter]
        x, y, z = center[0], center[1], center[2]
        self.spawn_counter+= 1
        if self.spawn_counter == len(self.subterrain_centers):
            self.spawn_counter = 0
       
        return x, y, z


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        
        actions_scaled = actions * self.env_cfg['action_scale']
        if HIP_REDUCTION:
            hip_indices = torch.tensor([0, 3, 6, 9], device=actions.device)
            actions_scaled[:, hip_indices] *= self.env_cfg["hip_reduction_scale"]
        torques = (
            self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets)
            - self.batched_d_gains * self.dof_vel
        )
        torques =  torques * self.motor_strengths

            # torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if TEACHER_STUDENT:
            return self.obs_buf, self.privileged_obs_buf, self.obs_history_buf
        else:
            return self.obs_buf, self.privileged_obs_buf


    def _render_headless(self):
        if LATERAL_CAM:
            # if self._recording and len(self._recorded_frames) < 150:
            #     x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
            #     self.cam_0.set_pose(pos=(x+5.0, y, z), lookat=(x, y, z+0.5))
            #     if self.show_vis:
            #         self.cam_0.render(
            #             rgb=True,
            #         )
            #     frame, _, _, _ = self.cam_0.render()
            #     self._recorded_frames.append(frame)
            # elif self.show_vis:
            #     x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
            #     self.cam_0.set_pose(pos=(x+5.0, y, z), lookat=(x, y, z+0.5))
            #     self.cam_0.render(
            #         rgb=True,
            #     )
            if self._recording and len(self._recorded_frames) < 150:
                x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
                self.cam_0.set_pose(pos=(x, y+4.0, z), lookat=(x, y, z+0.2))
                if self.show_vis:
                    self.cam_0.render(
                        rgb=True,
                    )
                frame, _, _, _ = self.cam_0.render()
                self._recorded_frames.append(frame)
            elif self.show_vis:
                x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
                self.cam_0.set_pose(pos=(x, y+4.0, z), lookat=(x, y, z+0.2))
                self.cam_0.render(
                    rgb=True,
                )
        elif TOP_CAM:
            if self._recording and len(self._recorded_frames) < 150:
                x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
                self.cam_0.set_pose(pos=(x, y, z+4.0), lookat=(x, y, z+0.2))
                if self.show_vis:
                    self.cam_0.render(
                        rgb=True,
                    )
                frame, _, _, _ = self.cam_0.render()
                self._recorded_frames.append(frame)
            elif self.show_vis:
                x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
                self.cam_0.set_pose(pos=(x, y, z+4.0), lookat=(x, y, z+0.2))
                self.cam_0.render(
                    rgb=True,
                )
        elif FRONT_CAM:
            if self._recording and len(self._recorded_frames) < 150:
                x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
                self.cam_0.set_pose(pos=(x+4.0, y, z), lookat=(x, y, z+0.2))
                if self.show_vis:
                    self.cam_0.render(
                        rgb=True,
                    )
                frame, _, _, _ = self.cam_0.render()
                self._recorded_frames.append(frame)
            elif self.show_vis:
                x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
                self.cam_0.set_pose(pos=(x+4.0, y, z), lookat=(x, y, z+0.2))
                self.cam_0.render(
                    rgb=True,
                )
        else:
            if self._recording and len(self._recorded_frames) < 150:
                x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
                self.cam_0.set_pose(pos=(x+5.0, y, z+5.5), lookat=(x, y, z+0.5))
                if self.show_vis:
                    self.cam_0.render(
                        rgb=True,
                    )
                frame, _, _, _ = self.cam_0.render()
                self._recorded_frames.append(frame)
            elif self.show_vis:
                x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
                self.cam_0.set_pose(pos=(x+5.0, y, z+5.5), lookat=(x, y, z+0.5))
                self.cam_0.render(
                    rgb=True,
                )
    

    def get_recorded_frames(self):
        if len(self._recorded_frames) == 150:
            frames = self._recorded_frames
            self._recorded_frames = []
            self._recording = False
            return frames
        else:
            return None

    def start_recording(self, record_internal=False):
        self._recorded_frames = []
        self._recording = True
        if record_internal:
            self._record_frames = True
        else:
            self.cam_0.start_recording()

    def stop_recording(self, save_path=None):
        self._recorded_frames = []
        self._recording = False
        if save_path is not None:
            print("fps", int(1 / self.dt))
            self.cam_0.stop_recording(save_path, fps = int(1 / self.dt))

    # ------------ domain randomization----------------

    def _randomize_rigids(self, env_ids=None):

        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['randomize_friction']:
            self._randomize_link_friction(env_ids)
        if self.env_cfg['randomize_base_mass']:
            self._randomize_base_mass(env_ids)
        if self.env_cfg['randomize_com_displacement']:
            self._randomize_com_displacement(env_ids)

    def _randomize_controls(self, env_ids=None):

        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['randomize_motor_strength']:
            self._randomize_motor_strength(env_ids)
        if self.env_cfg['randomize_motor_offset']:
            self._randomize_motor_offset(env_ids)
        if self.env_cfg['randomize_kp_scale']:
            self._randomize_kp(env_ids)
        if self.env_cfg['randomize_kd_scale']:
            self._randomize_kd(env_ids)

    def _randomize_link_friction(self, env_ids):

        min_friction, max_friction = self.env_cfg['friction_range']

        solver = self.rigid_solver

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) \
                 * (max_friction - min_friction) + min_friction
        if torch.isnan(ratios).any():
            print("NaN in friction ratios before applying them!")
            print("ratios:", ratios)
            raise ValueError("NaNs in friction ratios.")
        
        # print("friction ratios: ", ratios)
        # print("shape of ratios: ", ratios.shape)
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)
        
        self.frictions = ratios
        print("friction ratios: ", self.frictions.shape)

    def _randomize_base_mass(self, env_ids):

        min_mass, max_mass = self.env_cfg['added_mass_range']
        base_link_id = 1

        added_mass = gs.rand((len(env_ids), 1), dtype=float) \
                        * (max_mass - min_mass) + min_mass

        self.rigid_solver.set_links_mass_shift(added_mass, [base_link_id,], env_ids)

        self.added_masses = added_mass
        print("added mass: ", self.added_masses.shape)

    def _randomize_com_displacement(self, env_ids):

        min_displacement, max_displacement = self.env_cfg['com_displacement_range']
        base_link_id = 1

        com_displacement = gs.rand((len(env_ids), 1, 3), dtype=float) \
                            * (max_displacement - min_displacement) + min_displacement
        # com_displacement[:, :, 0] -= 0.02

        self.rigid_solver.set_links_COM_shift(com_displacement, [base_link_id,], env_ids)

        self.com_displacement = com_displacement
        print("com displacement: ", self.com_displacement.shape)

    def _randomize_motor_strength(self, env_ids):

        min_strength, max_strength = self.env_cfg['motor_strength_range']
        self.motor_strengths[env_ids, :] = gs.rand((len(env_ids), 1), dtype=float) \
                                           * (max_strength - min_strength) + min_strength
        print("self.motor_strengths: ", self.motor_strengths.shape)


    def _randomize_motor_offset(self, env_ids):

        min_offset, max_offset = self.env_cfg['motor_offset_range']
        self.motor_offsets[env_ids, :] = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                                         * (max_offset - min_offset) + min_offset

    def _randomize_kp(self, env_ids):

        min_scale, max_scale = self.env_cfg['kp_scale_range']
        kp_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                    * (max_scale - min_scale) + min_scale
        self.batched_p_gains[env_ids, :] = kp_scales * self.p_gains[None, :]

    def _randomize_kd(self, env_ids):

        min_scale, max_scale = self.env_cfg['kd_scale_range']
        kd_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                    * (max_scale - min_scale) + min_scale
        self.batched_d_gains[env_ids, :] = kd_scales * self.d_gains[None, :]




    # ------------ reward functions----------------

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_lin_vel_world(self):
        # print("self.vx_plane: ", self.vx_plane)
        print("shape of self.vx_plane: ", self.vx_plane.shape)
        # print("shape of self.commands[:, 0]: ", self.commands[:, 0].shape)
        # breakpoint()
        # Tracking of linear velocity commands (xy axes)
        # vx_error = torch.square(self.commands[:, 0] - self.vx_plane)
        # vy_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        # lin_vel_error = vx_error + vy_error
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_collision(self):
        """
        Penalize collisions on selected bodies.
        Returns the per-env penalty value as a 1D tensor of shape (n_envs,).
        """
        # print("self.penalised_contact_indices: ", self.penalised_contact_indices)
        # print("self.robot.n_links: ", self.robot.n_links)
        # # self.penalised_contact_indices:  [0, 5, 6, 7, 8, 9, 10, 11, 12]
        # # self.robot.n_links:  17
        # assert torch.max(torch.tensor(self.penalised_contact_indices)) < self.robot.n_links, (
        #     f"Invalid penalised_contact_indices! Max index: {max(self.penalised_contact_indices)}, "
        #     f"n_links: {self.robot.n_links}"
        # )

        undesired_forces = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
        collisions = (undesired_forces > 0.1).float()  # shape (n_envs, len(...))
        # print("collisions: ", collisions)
        
        # Sum over those links to get # of collisions per environment
        return collisions.sum(dim=1)


    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)


    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_slosh_free(self):
        # 1. Compute raw a_x, a_z
        ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) / (1 / self.linvel_update_actual_freq)
        az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) / (1 / self.linvel_update_actual_freq)
        
        # 2. Exponential smoothing
        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        
        # 3. Use the filtered values
        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale

        self.smoothed_ax_mean = torch.mean(ax_smooth)
        self.smoothed_az_mean = torch.mean(az_smooth)

        desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        desired_pitch_degrees = torch.rad2deg(desired_pitch)

        # Compute the squared error between desired and current pitch angles
        # error = torch.sum(torch.square(desired_pitch_degrees - self.rot_y_deg)) # sum -> too gig
        error = torch.mean(torch.square(desired_pitch_degrees - self.rot_y_deg))

        # return torch.exp(-error / self.reward_cfg["tracking_sigma"]) # zero -> not noticed
        return error
    
    def _reward_slosh_free_world(self):
        ax = (self.vx_plane - self.last_vx_plane) / self.dt
        az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        
        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale

        accel_norm = torch.sqrt(ax_smooth**2 + az_smooth**2 + 1e-8)

        # desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        desired_pitch = torch.atan2(ax_smooth, -az_smooth)
        desired_pitch_degrees = torch.rad2deg(desired_pitch)
        desired_pitch_degrees = desired_pitch_degrees.unsqueeze(-1)

        # desired_pitch_degrees = - desired_pitch_degrees # Add this for correct tilt direction

        error = torch.mean(torch.square(desired_pitch_degrees - self.rot_y_deg))

        # Logging for monitoring purposes
        self.smoothed_ax_mean = torch.mean(ax_smooth)
        self.smoothed_az_mean = torch.mean(az_smooth)
        self.mean_pitch_error_normalized = torch.mean(desired_pitch_degrees - self.rot_y_deg)
        self.mean_accel_norm_normalized = torch.mean(accel_norm)

        return error
    
    def _reward_slosh_free_world_v2(self):
        ax = (self.vx_plane - self.last_vx_plane) / self.dt
        az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        
        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale

        accel_norm = torch.sqrt(ax_smooth**2 + az_smooth**2 + 1e-8)

        desired_pitch = torch.atan2(ax_smooth, -az_smooth)
        desired_pitch_degrees = torch.rad2deg(desired_pitch)
        desired_pitch_degrees = desired_pitch_degrees.unsqueeze(-1)

        error = torch.square(desired_pitch_degrees - self.rot_y_deg)
        error = error.squeeze(-1)

        # Logging for monitoring purposes
        self.smoothed_ax_mean = torch.mean(ax_smooth)
        self.smoothed_az_mean = torch.mean(az_smooth)
        self.mean_pitch_error_normalized = torch.mean(desired_pitch_degrees - self.rot_y_deg)
        self.mean_accel_norm_normalized = torch.mean(accel_norm)

        return error
    
    def _reward_hip_vel(self):
        return torch.square(self.hip_vel)

    def _reward_hip_pos(self):
        # .squeeze(-1)
        error = torch.square(self.hip_pos- self.default_hip_pos)
        print(f"error shape {error.shape} and value {error}")
        breakpoint()
        return error
        # return torch.square(self.hip_pos- self.default_hip_pos)

    
    def _reward_slosh_free_world_v3(self):
        ax = (self.vx_plane - self.last_vx_plane) / self.dt
        ay = (self.vy_plane - self.last_vy_plane) / self.dt
        az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.ay_filtered = self.alpha * self.ay_filtered + (1.0 - self.alpha) * ay
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        
        ax_smooth = self.ax_filtered * self.ax_scale
        ay_smooth = self.ay_filtered
        az_smooth = self.az_filtered * self.az_scale

        desired_pitch = torch.atan2(ax_smooth, -az_smooth)
        desired_pitch_degrees = torch.rad2deg(desired_pitch)
        desired_pitch_degrees = desired_pitch_degrees.unsqueeze(-1)

        desired_roll = torch.atan2(-ay_smooth, -az_smooth)
        desired_roll_degrees = torch.rad2deg(desired_roll)
        desired_roll_degrees = desired_roll_degrees.unsqueeze(-1)

        error_pitch = torch.square(desired_pitch_degrees - self.rot_y_deg)
        error_pitch = error_pitch.squeeze(-1)

        error_roll = torch.square(desired_roll_degrees - self.rot_x_deg)
        error_roll = error_roll.squeeze(-1)

        error = error_pitch + error_roll

        # Logging for monitoring purposes
        self.smoothed_ax_mean = torch.mean(ax_smooth)
        self.smoothed_az_mean = torch.mean(az_smooth)
        self.mean_pitch_error_normalized = torch.mean(desired_pitch_degrees - self.rot_y_deg)
        self.mean_roll_error_normalized = torch.mean(desired_roll_degrees - self.rot_x_deg)

        return error
    
    def _reward_slosh_free_world_xz(self):
        ax = (self.vx_plane - self.last_vx_plane) / self.dt
        az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        
        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale

        desired_pitch = torch.atan2(ax_smooth, -az_smooth)
        desired_pitch_degrees = torch.rad2deg(desired_pitch)
        desired_pitch_degrees = desired_pitch_degrees.unsqueeze(-1)

        error_pitch = torch.square(desired_pitch_degrees - self.rot_y_deg)
        error_pitch = error_pitch.squeeze(-1)

        error = error_pitch

        # Logging for monitoring purposes
        self.smoothed_ax_mean = torch.mean(ax_smooth)
        self.smoothed_az_mean = torch.mean(az_smooth)
        self.mean_pitch_error_normalized = torch.mean(desired_pitch_degrees - self.rot_y_deg)

        return error
    
    def _reward_slosh_free_world_yz(self):
        ay = (self.vy_plane - self.last_vy_plane) / self.dt
        # az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        self.ay_filtered = self.alpha * self.ay_filtered + (1.0 - self.alpha) * ay
        # self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        
        ay_smooth = self.ay_filtered
        az_smooth = self.az_filtered * self.az_scale

        desired_roll = torch.atan2(ay_smooth, -az_smooth)
        desired_roll_degrees = torch.rad2deg(desired_roll)
        desired_roll_degrees = desired_roll_degrees.unsqueeze(-1)

        error_roll = torch.square(desired_roll_degrees - self.rot_x_deg)
        error_roll = error_roll.squeeze(-1)

        error = error_roll

        # Logging for monitoring purposes
        self.smoothed_az_mean = torch.mean(az_smooth)
        self.mean_roll_error_normalized = torch.mean(desired_roll_degrees - self.rot_x_deg)

        return error

    def _reward_slosh_free_first_derivative(self):
        # 1. Compute raw accelerations
        ax = (self.vx_plane - self.last_vx_plane) / self.dt
        az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        # Exponential smoothing for ax and az
        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az

        # Use smoothed values
        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale

        # Compute desired pitch (theta_d)
        desired_pitch_rad = torch.atan2(-ax_smooth, -az_smooth)
        desired_pitch_deg = torch.rad2deg(desired_pitch_rad).unsqueeze(-1)
        desired_pitch_deg = - desired_pitch_deg

        # Compute measured angular velocity dθ/dt
        dtheta_dt = (self.rot_y_deg - self.last_rot_y_deg) / self.dt

        # Compute desired angular velocity dθd/dt discretely
        dtheta_d_dt = (desired_pitch_deg - self.last_desired_pitch_deg) / self.dt

        # Compute spillnot angle error
        spillnot = torch.mean((desired_pitch_deg - self.rot_y_deg)**2)

        # Compute spillnot velocity error
        spillnotvel = torch.mean((dtheta_d_dt - dtheta_dt)**2)

        # Combine the errors into total reward (you may use weighted sum)
        total_error = spillnot + spillnotvel

        # Logging useful values for monitoring
        self.mean_pitch_error_normalized = torch.mean(desired_pitch_deg - self.rot_y_deg)
        self.mean_spillnotvel_error = spillnotvel

        # Update previous states for next iteration
        self.last_rot_y_deg = self.rot_y_deg.clone()
        self.last_desired_pitch_deg = desired_pitch_deg.clone()

        return total_error
    
    def _reward_slosh_free_lateral_acc(self):
        # 1. Compute acceleration
        ax = (self.vx_plane - self.last_vx_plane) / self.dt
        az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        # 2. Smooth
        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az

        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale

        # 3. Desired pitch and pitch error (in radians)
        desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        desired_pitch_degrees = torch.rad2deg(desired_pitch)
        desired_pitch_degrees = desired_pitch_degrees.unsqueeze(-1)
        desired_pitch_degrees = - desired_pitch_degrees
        pitch_error = desired_pitch_degrees - self.rot_y_deg

        # 7. Final reward (penalty)
        reward = torch.mean(torch.square(pitch_error)) * torch.abs(ax)


        return reward

    def _reward_slosh_free_by_acc(self):
        # 1. Compute acceleration
        ax = (self.vx_plane - self.last_vx_plane) / self.dt
        az = -9.8 + (self.vz_world - self.last_vz_world) / self.dt

        # 2. Smooth
        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az

        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale

        reward = torch.abs(ax_smooth)

        return reward
    
    def _reward_tracking_acc_x(self):
        """
        Encourage the robot to match a desired acceleration in x-direction.
        """
        # 1. Compute current base acceleration in x
        ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) / (1 / self.linvel_update_actual_freq)

        # 2. Optional: smooth it
        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        ax_smooth = self.ax_filtered * self.ax_scale
        self.smoothed_ax_mean = torch.mean(ax_smooth)

        # 3. Get desired acceleration.
        desired_ax = (self.commands[:, 0] - self.base_lin_vel_x) / (1 / self.linvel_update_actual_freq)
        self.desired_ax_filtered = self.alpha * self.desired_ax_filtered + (1.0 - self.alpha) * desired_ax
        self.smoothed_desired_ax_mean = torch.mean(self.desired_ax_filtered)

        # 4. Compute squared error
        acc_error = torch.square(ax_smooth - self.desired_ax_filtered)

        # 5. Return reward (higher when error is smaller)
        return torch.exp(-acc_error / self.reward_cfg["tracking_sigma"])


    def _reward_slosh_free_lateral_acc_condition_acc(self):
        # 1. Compute acceleration (using differences in linear velocity)
        ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) * self.linvel_update_actual_freq
        az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) * self.linvel_update_actual_freq

        # 2. Smooth the raw acceleration values
        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az

        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale

        self.smoothed_ax_mean = torch.mean(ax_smooth)
        self.smoothed_az_mean = torch.mean(az_smooth)

        # 3. Compute desired pitch and pitch error (in radians), then convert error to degrees
        desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        pitch_error = desired_pitch - self.rot_y.view(-1)  # assuming self.rot_y is in radians
        pitch_error_deg = torch.rad2deg(pitch_error)

        # 4. Compute lateral acceleration norm
        accel_norm = torch.sqrt(ax_smooth**2 + az_smooth**2 + 1e-8)

        # 5. Update max acceleration for normalization (use no_grad so gradients are not affected)
        with torch.no_grad():
            self.max_accel_norm = torch.maximum(self.max_accel_norm, accel_norm.max())
        # Normalize lateral acceleration (this is your Norm(Acc_c))
        # accel_norm_normalized = accel_norm / (self.max_accel_norm + 1e-4)
        accel_norm_normalized = accel_norm

        # 6. Define COST1 as the absolute tilt error (in degrees)
        cost1 = torch.abs(pitch_error_deg)
        # Define N (maximum error weight) from your reward configuration
        N = self.reward_cfg["max_pitch_error_weight"]
        # Clip COST1 so that any error above N is treated as just N
        clipped_cost1 = torch.clamp(cost1, 0.0, N)

        # 7. Compute the final reward:
        #    The higher the tilt error (up to N), the lower the term (N - clipped_cost1)
        #    and hence the lower the reward. It is multiplied by the normalized lateral acceleration.
        reward = (N - clipped_cost1) * accel_norm_normalized

        # Logging for monitoring purposes
        self.mean_pitch_error_normalized = torch.mean(cost1)
        self.mean_accel_norm_normalized = torch.mean(accel_norm_normalized)

        return reward
    
    def _reward_slosh_free_lateral_acc_condition_acc_noclip(self):
        # 1. Compute acceleration (using differences in linear velocity)
        ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) * self.linvel_update_actual_freq
        az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) * self.linvel_update_actual_freq
        az_net = (self.base_lin_vel_z - self.last_base_lin_vel_z) * self.linvel_update_actual_freq

        # 2. Smooth the raw acceleration values
        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        self.az_net_filtered = self.alpha * self.az_net_filtered + (1.0 - self.alpha) * az_net

        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale
        az_net_smooth = self.az_net_filtered * self.az_scale

        self.smoothed_ax_mean = torch.mean(ax_smooth)
        self.smoothed_az_mean = torch.mean(az_smooth)
        self.smoothed_az_mean = torch.mean(az_net_smooth)

        # 3. Compute desired pitch and pitch error (in radians), then convert error to degrees
        desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        pitch_error = desired_pitch - self.rot_y.view(-1)  # assuming self.rot_y is in radians
        pitch_error_deg = torch.rad2deg(pitch_error)

        # 4. Compute lateral acceleration norm
        accel_norm = torch.sqrt(ax_smooth**2 + az_smooth**2 + 1e-8)
        # accel_net_norm = torch.sqrt(ax_smooth**2 + az_net_smooth**2 + 1e-8)
        accel_net_norm = torch.sqrt((2*ax_smooth)**2 + (az_net_smooth)**2 + 1e-8)

        # 5. Update max acceleration for normalization (use no_grad so gradients are not affected)
        with torch.no_grad():
            self.max_accel_norm = torch.maximum(self.max_accel_norm, accel_norm.max())
        # Normalize lateral acceleration (this is your Norm(Acc_c))
        # accel_norm_normalized = accel_norm / (self.max_accel_norm + 1e-4)
        accel_norm_normalized = accel_norm
        # accel_norm_normalized = accel_net_norm

        # 6. Define COST1 as the absolute tilt error (in degrees)
        cost1 = torch.abs(pitch_error_deg)
        # Define N (maximum error weight) from your reward configuration
        N = self.reward_cfg["max_pitch_error_weight"]

        # 7. Compute the final reward:
        #    The higher the tilt error (up to N), the lower the term (N - clipped_cost1)
        #    and hence the lower the reward. It is multiplied by the normalized lateral acceleration.
        reward = (N - cost1) * accel_norm_normalized

        # Logging for monitoring purposes
        self.mean_pitch_error_normalized = torch.mean(cost1)
        self.mean_accel_norm_normalized = torch.mean(accel_norm_normalized)

        return reward
    
    def _reward_slosh_free_lateral_acc_condition_acc_noclip_world(self):
        # 1. Compute acceleration (using differences in linear velocity)
        ax = (self.vx_plane - self.last_vx_plane) / (1 / self.linvel_update_actual_freq)
        az = -9.8 + (self.vz_world - self.last_vz_world) / (1 / self.linvel_update_actual_freq)
        az_net = (self.vz_world - self.last_vz_world) * self.linvel_update_actual_freq

        # 2. Smooth the raw acceleration values
        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az
        self.az_net_filtered = self.alpha * self.az_net_filtered + (1.0 - self.alpha) * az_net

        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale
        az_net_smooth = self.az_net_filtered * self.az_scale

        self.smoothed_ax_mean = torch.mean(ax_smooth)
        self.smoothed_az_mean = torch.mean(az_smooth)
        self.smoothed_az_mean = torch.mean(az_net_smooth)

        # 3. Compute desired pitch and pitch error (in radians), then convert error to degrees
        desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        pitch_error = desired_pitch - self.rot_y.view(-1)  # assuming self.rot_y is in radians
        pitch_error_deg = torch.rad2deg(pitch_error)

        # 4. Compute lateral acceleration norm
        accel_norm = torch.sqrt(ax_smooth**2 + az_smooth**2 + 1e-8)
        # accel_net_norm = torch.sqrt(ax_smooth**2 + az_net_smooth**2 + 1e-8)
        accel_net_norm = torch.sqrt((2*ax_smooth)**2 + (az_net_smooth)**2 + 1e-8)

        # 5. Update max acceleration for normalization (use no_grad so gradients are not affected)
        with torch.no_grad():
            self.max_accel_norm = torch.maximum(self.max_accel_norm, accel_norm.max())
        # Normalize lateral acceleration (this is your Norm(Acc_c))
        # accel_norm_normalized = accel_norm / (self.max_accel_norm + 1e-4)
        accel_norm_normalized = accel_norm
        # accel_norm_normalized = accel_net_norm

        # 6. Define COST1 as the absolute tilt error (in degrees)
        cost1 = torch.abs(pitch_error_deg)
        # Define N (maximum error weight) from your reward configuration
        N = self.reward_cfg["max_pitch_error_weight"]

        # 7. Compute the final reward:
        #    The higher the tilt error (up to N), the lower the term (N - clipped_cost1)
        #    and hence the lower the reward. It is multiplied by the normalized lateral acceleration.
        reward = (N - cost1) * accel_norm_normalized

        # Logging for monitoring purposes
        self.mean_pitch_error_normalized = torch.mean(cost1)
        self.mean_accel_norm_normalized = torch.mean(accel_norm_normalized)

        return reward

    def _reward_slosh_free_lateral_acc_div_by_tilt(self):
        # 1. Compute acceleration
        ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) * self.linvel_update_actual_freq
        az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) * self.linvel_update_actual_freq

        # 2. Smooth
        self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
        self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az

        ax_smooth = self.ax_filtered * self.ax_scale
        az_smooth = self.az_filtered * self.az_scale

        # 3. Desired pitch and pitch error (in radians)
        desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
        pitch_error = desired_pitch - self.rot_y.view(-1)
        pitch_error_deg = torch.rad2deg(pitch_error)

        # 4. Acceleration norm
        accel_norm = torch.sqrt(ax_smooth**2 + az_smooth**2 + 1e-8)

        # 5. Update max values (no grad)
        with torch.no_grad():
            self.max_accel_norm = torch.maximum(self.max_accel_norm, accel_norm.max())
            # self.max_pitch_error = torch.maximum(self.max_pitch_error, pitch_error_abs.max())

        # 6. Normalize both
        accel_norm_normalized = accel_norm / (self.max_accel_norm + 1e-4)
        # pitch_error_normalized = pitch_error_abs / (self.max_pitch_error + 1e-4)
        
        # reward = torch.exp(accel_norm_normalized / (torch.abs(pitch_error_deg)+ 1e-4)) # too big
        # reward = torch.log1p((accel_norm + 1) / (torch.abs(pitch_error_deg) + 1e-3))
        reward = torch.log1p(torch.square(ax_smooth + 1) / (torch.abs(pitch_error_deg) + 1e-3))

        # rew1 = torch.tensor(0.0, device=self.device)
        # rew2 = (accel_norm_normalized / pitch_error_deg)

        # mask = ax_smooth > self.reward_cfg["ax_threshold"]
        # reward = torch.where(mask, rew1, rew2)

        self.mean_pitch_error_normalized = torch.mean(pitch_error_deg)
        self.mean_accel_norm_normalized = torch.mean(accel_norm)

        return reward



    # def _reward_slosh_free(self):
    #     # 1. Compute raw a_x, a_z
    #     ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) / (1 / self.linvel_update_actual_freq)
    #     az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) / (1 / self.linvel_update_actual_freq)

    #     # 2. Exponential smoothing
    #     self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
    #     self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az

    #     # 3. Apply scaling
    #     ax_smooth = self.ax_filtered * self.ax_scale
    #     az_smooth = self.az_filtered * self.az_scale

    #     # 4. Compute desired pitch from filtered acceleration
    #     desired_pitch = torch.atan2(-ax_smooth, -az_smooth)

    #     # 5. Compute pitch error
    #     pitch_error = desired_pitch - self.rot_y.view(-1)  # assuming self.rot_y_deg is in degrees
    #     pitch_error_deg = torch.rad2deg(pitch_error)

    #     # 6. Compute reward: negative squared sine-weighted error
    #     # reward = torch.abs(torch.sin(pitch_error))
    #     # reward = torch.abs(pitch_error)
    #     reward = torch.square(pitch_error_deg)
    #     # print("shape: ", reward)
    #     # breakpoint()

    #     return reward # want this to be smaller -> want pitch_error to be small
    
    # def _reward_slosh_free(self):
    #     # 1. Compute raw a_x, a_z
    #     ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) / (1 / self.linvel_update_actual_freq)
    #     az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) / (1 / self.linvel_update_actual_freq)

    #     # 2. Exponential smoothing
    #     self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
    #     self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az

    #     # 3. Apply scaling
    #     ax_smooth = self.ax_filtered * self.ax_scale
    #     az_smooth = self.az_filtered * self.az_scale

    #     # 4. Compute desired pitch from filtered acceleration
    #     desired_pitch = torch.atan2(-ax_smooth, -az_smooth)  # in radians
    #     desired_pitch_degrees = torch.rad2deg(desired_pitch)

    #     # 5. Compute pitch error
    #     pitch_error = desired_pitch_degrees - self.rot_y_deg.view(-1)  # assuming self.rot_y_deg is in degrees

    #     # 6. Compute norm of acceleration in x-z plane
    #     accel_norm = torch.sqrt(ax_smooth**2 + az_smooth**2 + 1e-8)  # add epsilon to prevent NaN

    #     # 7. Compute reward: negative squared sine-weighted error
    #     # reward = - (torch.sin(pitch_error) * accel_norm)**2
    #     reward = torch.abs(torch.sin(pitch_error) * accel_norm)

    #     return reward # want this to be smaller -> want pitch_error to be small
    
    
    # def _reward_slosh_free(self):
    #     # 1. Compute acceleration
    #     ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) * self.linvel_update_actual_freq
    #     az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) * self.linvel_update_actual_freq

    #     # 2. Smooth
    #     self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
    #     self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az

    #     ax_smooth = self.ax_filtered * self.ax_scale
    #     az_smooth = self.az_filtered * self.az_scale

    #     # 3. Desired pitch and pitch error (in radians)
    #     desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
    #     pitch_error = desired_pitch - self.rot_y.view(-1)
    #     pitch_error = torch.rad2deg(pitch_error)
    #     pitch_error_abs = torch.abs(pitch_error)
    #     pitch_error_square = pitch_error**2

    #     # 4. Acceleration norm
    #     accel_norm = torch.sqrt(ax_smooth**2 + az_smooth**2 + 1e-8)

    #     # 5. Update max values (no grad)
    #     with torch.no_grad():
    #         self.max_accel_norm = torch.maximum(self.max_accel_norm, accel_norm.max())
    #         self.max_pitch_error = torch.maximum(self.max_pitch_error, pitch_error_abs.max())

    #     # 6. Normalize both
    #     accel_norm_normalized = accel_norm / (self.max_accel_norm + 1e-4)
    #     # pitch_error_normalized = pitch_error_abs / (self.max_pitch_error + 1e-4)
    #     pitch_error_normalized = 1.0 + (pitch_error_abs / (self.max_pitch_error + 1e-4))


    #     # 7. Final reward (penalty)
    #     reward = pitch_error_square * accel_norm_normalized

    #     # For logging
    #     # self.mean_pitch_error_normalized = torch.mean(pitch_error_normalized)
    #     self.mean_pitch_error_normalized = torch.mean(pitch_error_square)
    #     self.mean_accel_norm_normalized = torch.mean(accel_norm_normalized)

    #     return reward

    # def _reward_slosh_free(self):
    #     # Acceleration
    #     ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) * self.linvel_update_actual_freq
    #     az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) * self.linvel_update_actual_freq

    #     self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
    #     self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az

    #     ax_smooth = self.ax_filtered * self.ax_scale
    #     az_smooth = self.az_filtered * self.az_scale

    #     # Pitch error
    #     desired_pitch = torch.atan2(-ax_smooth, -az_smooth)
    #     pitch_error = desired_pitch - self.rot_y.view(-1)
    #     pitch_error = torch.rad2deg(pitch_error)
    #     pitch_error_abs = torch.abs(pitch_error)

    #     # Acc norm
    #     accel_norm = torch.sqrt(ax_smooth**2 + az_smooth**2 + 1e-8)

    #     # Normalize acceleration using sigmoid
    #     accel_norm_normalized = torch.tanh(accel_norm/19.6)

    #     reward = pitch_error_abs * accel_norm_normalized

    #     self.mean_pitch_error_normalized = torch.mean(pitch_error_abs)
    #     self.mean_accel_norm_normalized = torch.mean(accel_norm_normalized)

    #     return reward
    
    # def _reward_tracking_acc_x(self):



    # def _reward_slosh_free(self):
    #     # 1. Compute raw a_x, a_z
    #     ax = (self.base_lin_vel_x - self.last_base_lin_vel_x) / (1 / self.linvel_update_actual_freq)
    #     az = -9.8 + (self.base_lin_vel_z - self.last_base_lin_vel_z) / (1 / self.linvel_update_actual_freq)

    #     # 2. Exponential smoothing
    #     self.ax_filtered = self.alpha * self.ax_filtered + (1.0 - self.alpha) * ax
    #     self.az_filtered = self.alpha * self.az_filtered + (1.0 - self.alpha) * az

    #     # 3. Apply scaling
    #     ax_smooth = self.ax_filtered * self.ax_scale
    #     az_smooth = self.az_filtered * self.az_scale

    #     # 4. Compute desired pitch from filtered acceleration
    #     desired_pitch = torch.atan2(-ax_smooth, -az_smooth)  # in radians
    #     desired_pitch_degrees = torch.rad2deg(desired_pitch)

    #     # 5. Compute pitch error
    #     pitch_error = desired_pitch_degrees - self.rot_y_deg.view(-1)  # assuming self.rot_y_deg is in degrees

    #     # 6. Compute norm of acceleration in x-z plane
    #     accel_norm = torch.sqrt(ax_smooth**2 + az_smooth**2 + 1e-8)  # add epsilon to prevent NaN

    #     # 7. Compute reward: negative squared sine-weighted error
    #     # reward = - (pitch_error * accel_norm)**2
    #     reward = torch.abs(pitch_error * accel_norm)

    #     return reward # want this to be smaller -> want pitch_error to be small



    
    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)


    # def _reward_action_rate_2nd_order(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.second_last_actions - 2*self.last_actions + self.actions), dim=1)

    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        # print(contact)
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        # print(contact_feet_vel)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_contact(self): # max 1
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Iterate over legs (order: FL, FR, RL, RR)
        for i in range(self.feet_num):
            # Determine if the current phase indicates a stance phase (< 0.55)
            is_stance = self.leg_phase[:, i] < 0.55

            # Check if the foot is in contact with the ground
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1

            # Reward correct contact behavior (stance matches contact)
            res += ~(contact ^ is_stance)  # XOR for mismatch, negate for correct match

            # 1 (reward) if the foot is correctly placed (correct stance-contact match).
            # 0 (no reward) if the foot is incorrectly placed.

        return res

    # def _reward_hip_vel(self):
    #     return torch.sum(torch.square(self.hip_vel), dim=(1))

    # def _reward_hip_pos(self):
    #     return torch.sum(torch.abs(self.hip_pos- self.default_hip_pos), dim=(1))


    # def _reward_front_feet_swing_height_from_base(self):
    #     # Get contact forces and determine which feet are in contact
    #     contact = torch.norm(self.contact_forces[:, self.feet_front_indices, :3], dim=2) > 1.0
    #     pos_error = torch.square((self.step_height_for_front - self.front_feet_pos_base[:, :, 2]) * ~contact)
    #     return torch.sum(pos_error, dim=1)


    # def _reward_front_feet_swing_height_from_world(self):
    #     contact = torch.norm(self.contact_forces[:, self.feet_front_indices, :3], dim=2) > 1.0
    #     pos_error = torch.square(self.feet_front_pos[:, :, 2] - self.step_height_for_front_from_world) * ~contact
    #     return torch.sum(pos_error, dim=(1))

    # def _reward_rear_feet_swing_height(self):
    #     # Get contact forces and determine which feet are in contact
    #     contact = torch.norm(self.contact_forces[:, self.feet_rear_indices, :3], dim=2) > 1.0
    #     pos_error = torch.square((self.step_height_for_rear - self.rear_feet_pos_base[:, :, 2]) * ~contact)
    #     return torch.sum(pos_error, dim=1)



    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)


    # def _reward_dof_vel(self):
    #     # Penalize dof velocities
    #     return torch.sum(torch.square(self.dof_vel), dim=1)


    # def _reward_torque_limits(self):
    #     # penalize torques too close to the limit
    #     return torch.sum((torch.abs(self.torques) - self.torque_limits*self.soft_torque_limit).clip(min=0.), dim=1)

    # def _reward_termination(self):
    #     # Terminal reward / penalty
    #     return self.reset_buf & ~self.time_out_buf & ~self.out_of_bounds_buf

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    # def _reward_feet_contact_forces(self):
    #     # penalize high contact forces
    #     return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.reward_cfg["max_contact_force"]).clip(min=0.), dim=1)
    


    def _reward_tracking_pitch_ang(self):
        # print("shape of command: ", self.commands[:, 3].shape)
        # print("shape of rot_y_deg: ", self.rot_y_deg.shape)
        # breakpoint()
        self.rot_y_deg = self.rot_y_deg.squeeze(-1)
        pitch_ang_error = torch.square(self.commands[:, 3] - self.rot_y_deg)
        return torch.exp(-pitch_ang_error / self.reward_cfg["tracking_sigma"])
    

    def _reward_constrained_slosh_free(self): # max 0
        ax = (self.last_base_lin_vel_x - self.base_lin_vel_x) / self.dt
        az = (self.last_base_lin_vel_z - self.base_lin_vel_z) / self.dt
        ax *= self.ax_scale
        az *= self.az_scale
        # Compute the desired pitch angle (theta) in radians
        desired_pitch = torch.atan2(ax, az)

        self.desired_pitch_mean = torch.mean(desired_pitch)

        # if torch.logical_and(-self.min_pitch_num < desired_pitch, desired_pitch < self.min_pitch_num).all():
        if -self.min_pitch_num < self.desired_pitch_mean and self.desired_pitch_mean < self.min_pitch_num:
            # desired_pitch = 0
            return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
        else:
            self.pitch_count += 1
            desired_pitch = desired_pitch
            # Compute the squared error between desired and current pitch angles
            return torch.sum(torch.square(desired_pitch - self.rot_y_deg))
        
