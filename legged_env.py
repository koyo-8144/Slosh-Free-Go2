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

# Turn off during high level control
PITCH_CMD = 0

# After training 12 joints, try high level control
PITCH_HIGHLEVEL = 1
PLOT = 1




class LeggedEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = obs_cfg["num_privileged_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        # self.num_props=env_cfg["n_proprio"],
        # self.num_hist=env_cfg["history_len"],

        # self.joint_limits = env_cfg["joint_limits"]
        self.simulate_action_latency = env_cfg["simulate_action_latency"]  # there is a 1 step latency on real robot
        self.dt = 1 / env_cfg['control_freq']
        sim_dt = self.dt / env_cfg['decimation']
        sim_substeps = 1
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
                res=(640, 480),
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

        # self.liquid = self.scene.add_entity(
        #     # viscous liquid
        #     # material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
        #     material=gs.materials.SPH.Liquid(),
        #     morph=gs.morphs.Cylinder(
        #         pos=(0.0, 0.0, self.base_init_pos[2].cpu().numpy() + 0.025),  # Lowered initial position
        #         height=0.04,  # Reduced height to fit inside boundary
        #         radius=0.025,  # Adjusted radius for better containment
        #     ),
        #     surface=gs.surfaces.Default(
        #         color=(0.4, 0.8, 1.0),
        #         vis_mode="particle",
        #         # vis_mode="recon",
        #     ),
        # )


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
            self.env_cfg['penalized_contact_link_names']
        )
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

        # input("ready?")


    def init_buffers(self):
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        if PITCH_CMD or PITCH_HIGHLEVEL:
            self.base_lin_vel_x = self.base_lin_vel[:, 0]
            self.base_lin_vel_z = self.base_lin_vel[:, 2]
            self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
            self.last_base_lin_vel_x = self.last_base_lin_vel[:, 0]
            self.last_base_lin_vel_z = self.last_base_lin_vel[:, 2]
            self.ax_scale = self.obs_scales["ax"]
            self.az_scale = self.obs_scales["az"]
            self.a_count = 0
            self.fixed_desired_pitch = 0
            self.base_lin_vel_x_low_freq = self.base_lin_vel[:, 0]
            self.base_lin_vel_z_low_freq = self.base_lin_vel[:, 2]
            self.min_pitch_num = self.reward_cfg["min_pitch_num"]
            # print("lin_vel_x_range_start: ", self.command_cfg["lin_vel_x_range_start"])
            # print("type: ", type(self.command_cfg["lin_vel_x_range_start"]))
            # print("0th element: ", self.command_cfg["lin_vel_x_range_start"][0])
            # print("1st element: ", self.command_cfg["lin_vel_x_range_start"][1])
            self.tracking_lin_vel_rew = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
            self.tracking_lin_vel_rew_mean = torch.tensor(0.0, device=self.device, dtype=gs.tc_float)
            self.tracking_lin_vel_rew_max = torch.full((self.num_envs,), self.reward_scales["tracking_lin_vel"], device=self.device)
            self.tracking_lin_vel_rew_max_one = self.tracking_lin_vel_rew_max [0]
            self.tracking_lin_vel_rew_threshold = self.tracking_lin_vel_rew_max * self.command_cfg["achieve_rate"]
            self.tracking_lin_vel_rew_threshold_one = self.tracking_lin_vel_rew_threshold[0]
            self.tracking_lin_vel_rew_threshold_one_ori  = self.tracking_lin_vel_rew_threshold_one
            self.lin_vel_x_range_min = self.command_cfg["lin_vel_x_range_start"][0]
            self.lin_vel_x_range_max = self.command_cfg["lin_vel_x_range_start"][1]
            self.updated_lin_vel_x_command_range = self.command_cfg["lin_vel_x_range_start"]
            self.desired_pitch_mean = 0
            self.first_resample_done = False
            self.first_resample_count = 0
            self.resample_count = 0
            self.range_update_count = 1
            self.pitch_count = 0
            self.resample_updated = False
            self.action_rate_scale = self.reward_scales["action_rate"]

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
        self.time_out_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.out_of_bounds_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        if PITCH_CMD or PITCH_HIGHLEVEL:
            self.commands_scale = torch.tensor(
                # [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
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
        self.rot_y = self.base_euler[:, 1]
        self.rot_y = self.rot_y.unsqueeze(-1)  # Shape: [4096, 1]
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
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


        if PLOT:
            self.base_lin_vel_x_list = []
            self.base_lin_vel_z_list = []
            self.last_base_lin_vel_x_list = []
            self.last_base_lin_vel_z_list = []
            self.ax_list = []
            self.az_list = []
            self.desired_theta_list = []
            self.current_pitch_list = []
            self.time_steps = []  # Track time steps
            self.max_points = 500  # Limit number of points to store for performance

            # Initialize the plot
            plt.ion()
            # self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 10))
            # self.fig, self.axs = plt.subplots(2, 1, figsize=(5, 5))
            self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 8))


            # self.axs[0].set_title("Current Base Linear X Velocities")
            # self.axs[0].set_ylabel("Velocity")
            # self.axs[0].legend(["Base Lin Vel X"])

            # self.axs[1].set_title("Last Base Linear X Velocities")
            # self.axs[1].set_ylabel("Velocity")
            # self.axs[1].legend(["Last Base Lin Vel X"])
            
            self.axs[0].set_title("Base Linear X Velocities")
            self.axs[0].set_ylabel("Velocity")
            self.axs[0].legend(["Base Lin Vel X", "Last Base Lin Vel X"])

            self.axs[1].set_title("Base Linear Z Velocities")
            self.axs[1].set_ylabel("Velocity")
            self.axs[1].legend(["Base Lin Vel Z", "Last Base Lin Vel Z"])

            self.axs[2].set_title("Accelerations")
            self.axs[2].set_xlabel("Time Steps")
            self.axs[2].set_ylabel("Acceleration")
            self.axs[2].legend(["Ax", "Az"])

            # self.axs[1].set_title("Accelerations")
            # self.axs[1].set_xlabel("Time Steps")
            # self.axs[1].set_ylabel("Acceleration")
            # self.axs[1].legend(["Ax"])

            self.axs[3].set_title("Pitch")
            self.axs[3].set_xlabel("Time Steps")
            self.axs[3].set_ylabel("Pitch")
            self.axs[3].legend(["Desired theta", "Current Pitch"])



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

            self.last_base_lin_vel_x_cmd = 0

    def init_camera_params(self):
        self.whole_view = False

        # Initialize rotation parameters
        self.radius = 30.0  # Radius of circular path
        self.theta = 0.0  # Initial angle
        self.theta_increment = np.radians(2)  # Increment angle by 2 degrees

        # Fixed target (lookat) position
        self.target = np.array([0.0, 0.0, 0.5])  # Assume robot is at this position
        self.camera_height = 0.0  # Fixed z-axis position for top-down view
    
    def _resample_commands_curriculum_linvel_x_v2(self, envs_idx): # update velocity range specifically for env that satisfies the threshold
        self.tracking_lin_vel_rew_mean = torch.mean(self.tracking_lin_vel_rew)
        # tracking_lin_vel_rew_high = self.tracking_lin_vel_rew > self.tracking_lin_vel_rew_threshold
        tracking_lin_vel_rew_high = self.tracking_lin_vel_rew_mean > self.tracking_lin_vel_rew_threshold_one

        if self.first_resample_done: # After first update
            self.resample_count += 1
            if (self.resample_count) > self.first_resample_count * self.range_update_count: # After spending larger time spent in the first range time
                self.tracking_lin_vel_rew_threshold_one = self.tracking_lin_vel_rew_threshold_one_ori # Get back to original threshold
            else: # Right after first update has been done
                self.tracking_lin_vel_rew_threshold_one = self.tracking_lin_vel_rew_max_one # So that I do not update soon
        else: # Before first update
            self.first_resample_count += 1

        # if tracking_lin_vel_rew_high.all():
        if tracking_lin_vel_rew_high:
            self.first_resample_done = True
            self.resample_count = 0
            self.resample_updated = True
            self.range_update_count += 1

            self.lin_vel_x_range_min = torch.clip(
                torch.tensor(self.lin_vel_x_range_min - self.command_cfg["increase_rate"]),
                self.command_cfg["lin_vel_x_range_goal"][0],
                0.0
            ).item()
            self.lin_vel_x_range_max = torch.clip(
                torch.tensor(self.lin_vel_x_range_max + self.command_cfg["increase_rate"]),
                0.0,
                self.command_cfg["lin_vel_x_range_goal"][1]
            ).item()
            self.updated_lin_vel_x_command_range = [self.lin_vel_x_range_min, self.lin_vel_x_range_max]



        self.commands[envs_idx, 0] = gs_rand_float(*self.updated_lin_vel_x_command_range, (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
    
    def _resample_commands_curriculum_linvel_x(self, envs_idx):

        # self.command_cfg["lin_vel_x_range_start"] # [-1.0, 1.0]
        # self.command_cfg["lin_vel_x_range_goal"] # [-2.0, 2.0]

        self.tracking_lin_vel_rew_mean = torch.mean(self.tracking_lin_vel_rew)
        # tracking_lin_vel_rew_high = self.tracking_lin_vel_rew > self.tracking_lin_vel_rew_threshold
        tracking_lin_vel_rew_high = self.tracking_lin_vel_rew_mean > self.tracking_lin_vel_rew_threshold_one

        if self.first_resample_done: # After first update
            self.resample_count += 1
            if (self.resample_count) > self.first_resample_count * self.range_update_count: # After spending larger time spent in the first range time
                self.tracking_lin_vel_rew_threshold_one = self.tracking_lin_vel_rew_threshold_one_ori # Get back to original threshold
            else: # Right after first update has been done
                self.tracking_lin_vel_rew_threshold_one = self.tracking_lin_vel_rew_max_one # So that I do not update soon
        else: # Before first update
            self.first_resample_count += 1

        # if tracking_lin_vel_rew_high.all():
        if tracking_lin_vel_rew_high:
            self.first_resample_done = True
            self.resample_count = 0
            self.resample_updated = True
            self.range_update_count += 1

            self.lin_vel_x_range_min = torch.clip(
                torch.tensor(self.lin_vel_x_range_min - self.command_cfg["increase_rate"]),
                self.command_cfg["lin_vel_x_range_goal"][0],
                0.0
            ).item()
            self.lin_vel_x_range_max = torch.clip(
                torch.tensor(self.lin_vel_x_range_max + self.command_cfg["increase_rate"]),
                0.0,
                self.command_cfg["lin_vel_x_range_goal"][1]
            ).item()
            self.updated_lin_vel_x_command_range = [self.lin_vel_x_range_min, self.lin_vel_x_range_max]



        self.commands[envs_idx, 0] = gs_rand_float(*self.updated_lin_vel_x_command_range, (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
    

    def _resample_commands_max(self, envs_idx):
        # Sample linear and angular velocities
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

        # Randomly multiply by -1 or 1 (50/50 chance)
        random_signs = torch.randint(0, 2, self.commands[envs_idx].shape, device=self.device) * 2 - 1
        self.commands[envs_idx] *= random_signs
    
    def _resample_commands_from_acc(self):
        self.a_count += 1
        self.base_lin_vel_x_cmd = self.commands[:, 0]
        # self.base_lin_vel_y_cmd = self.commands[:, 1] 
        # self.base_ang_vel_z_cmd = self.commands[:, 2]

        if self.a_count % 10 == 0:
            base_lin_vel_x_temp = self.base_lin_vel_x.clone()
            base_lin_vel_z_temp = self.base_lin_vel_z.clone()
            self.base_lin_vel_x_low_freq = base_lin_vel_x_temp
            self.base_lin_vel_z_low_freq = base_lin_vel_z_temp

        ax = (self.last_base_lin_vel_x - self.base_lin_vel_x_low_freq) / self.dt
        az = -9.8 + (self.last_base_lin_vel_z - self.base_lin_vel_z_low_freq) / self.dt
        ax *= self.ax_scale
        az *= self.az_scale
        # ax *= 10
        # az *= 5
        # Compute the desired pitch angle (theta) in radians
        desired_pitch = torch.atan2(ax, az)

        # print("ax: ", ax)
        # print("az: ", az)
        # print("Desired Pitch: ", desired_pitch)

        self.commands[:, 3]  = desired_pitch
        
        if PLOT:
            # Convert tensors to numbers (if using PyTorch)
            base_lin_vel_x = self.base_lin_vel_x_low_freq.item()
            base_lin_vel_z = self.base_lin_vel_z_low_freq.item()
            last_base_lin_vel_x = self.last_base_lin_vel_x.item()
            last_base_lin_vel_z = self.last_base_lin_vel_z.item()
            ax_val = ax.item()
            az_val = az.item()
            desired_pitch = desired_pitch.item()
            rot_y = self.rot_y.item()

            # print("current: ", base_lin_vel_x)
            # print("last: ", last_base_lin_vel_x)

            # Store values with a limit to prevent memory overload
            self.append_limited(self.base_lin_vel_x_list, base_lin_vel_x)
            self.append_limited(self.base_lin_vel_z_list, base_lin_vel_z)
            self.append_limited(self.last_base_lin_vel_x_list, last_base_lin_vel_x)
            self.append_limited(self.last_base_lin_vel_z_list, last_base_lin_vel_z)
            self.append_limited(self.ax_list, ax_val)
            self.append_limited(self.az_list, az_val)
            self.append_limited(self.desired_theta_list, desired_pitch)
            self.append_limited(self.current_pitch_list, rot_y)
            self.append_limited(self.time_steps, self.a_count)

            # # Only update the plot every 10 iterations for performance
            # if self.a_count % 10 == 0:
            # self.update_plot()
            self.update_plot_cla()

            if self.a_count == 10000:
                self.show_plot()

        # self.a_count += 1
        # if self.a_count % (1 / (self.dt)) == 0:
        # if self.a_count % (1 / (self.dt * 0.5)) == 0:
        # print("self.a_count: ", self.a_count)
        if self.a_count % 1 == 0:
            print("Last Base Lin Vel is updated")
            # breakpoint()
            base_lin_vel_x_temp = self.base_lin_vel_x_low_freq.clone()
            base_lin_vel_z_temp = self.base_lin_vel_z_low_freq.clone()
            self.last_base_lin_vel_x = base_lin_vel_x_temp
            self.last_base_lin_vel_z = base_lin_vel_z_temp
            
            # self.last_base_lin_vel_x = 0

    def append_limited(self, lst, value):
        """ Keep list size under self.max_points to avoid performance issues. """
        if len(lst) >= self.max_points:
            lst.pop(0)
        lst.append(value)

    def update_plot(self):
        """ Efficiently update existing plot data instead of redrawing everything. """
        self.line_base_lin_x.set_xdata(self.time_steps)
        self.line_base_lin_x.set_ydata(self.base_lin_vel_x_list)

        self.line_last_base_lin_x.set_xdata(self.time_steps)
        self.line_last_base_lin_x.set_ydata(self.last_base_lin_vel_x_list)

        # self.line_base_lin_z.set_xdata(self.time_steps)
        # self.line_base_lin_z.set_ydata(self.base_lin_vel_z_list)

        # self.line_last_base_lin_z.set_xdata(self.time_steps)
        # self.line_last_base_lin_z.set_ydata(self.last_base_lin_vel_z_list)

        # self.line_ax.set_xdata(self.time_steps)
        # self.line_ax.set_ydata(self.ax_list)

        # self.line_az.set_xdata(self.time_steps)
        # self.line_az.set_ydata(self.az_list)

        # self.line_desired_pitch.set_xdata(self.time_steps)
        # self.line_desired_pitch.set_ydata(self.desired_theta_list)

        # self.line_current_pitch.set_xdata(self.time_steps)
        # self.line_current_pitch.set_ydata(self.current_pitch_list)

        # Adjust axes limits dynamically
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        plt.draw()
        plt.pause(0.01)  # Refresh the plot in real-time

    def update_plot_cla(self):
        # self.axs[0].cla()
        # self.axs[0].plot(self.time_steps, self.base_lin_vel_x_list, label="Current Base Lin Vel X", color="b")
        # self.axs[0].set_ylabel("Velocity")
        # self.axs[0].legend()
        # self.axs[0].set_title("Current Base Linear X Velocities")

        # self.axs[1].cla()
        # self.axs[1].plot(self.time_steps, self.last_base_lin_vel_x_list, label="Last Base Lin Vel X", color="r")
        # self.axs[1].set_ylabel("Velocity")
        # self.axs[1].legend()
        # self.axs[1].set_title("Last Base Linear X Velocities")

        self.axs[0].cla()
        self.axs[0].plot(self.time_steps, self.base_lin_vel_x_list, label="Base Lin Vel X", color="b")
        self.axs[0].plot(self.time_steps, self.last_base_lin_vel_x_list, label="Last Base Lin Vel X", color="r")
        self.axs[0].set_ylabel("Velocity")
        self.axs[0].legend()
        self.axs[0].set_title("Base Linear X Velocities")

        self.axs[1].cla()
        self.axs[1].plot(self.time_steps, self.base_lin_vel_z_list, label="Base Lin Vel Z", color="b")
        self.axs[1].plot(self.time_steps, self.last_base_lin_vel_z_list, label="Last Base Lin Vel Z", color="r")
        self.axs[1].set_ylabel("Velocity")
        self.axs[1].legend()
        self.axs[1].set_title("Base Linear Z Velocities")

        self.axs[2].cla()
        self.axs[2].plot(self.time_steps, self.ax_list, label="Ax", color="g")
        self.axs[2].plot(self.time_steps, self.az_list, label="Az", color="m")
        self.axs[2].set_xlabel("Time Steps")
        self.axs[2].set_ylabel("Acceleration")
        self.axs[2].legend()
        self.axs[2].set_title("Accelerations")

        # self.axs[1].cla()
        # self.axs[1].plot(self.time_steps, self.ax_list, label="Ax", color="g")
        # self.axs[1].set_xlabel("Time Steps")
        # self.axs[1].set_ylabel("Acceleration")
        # self.axs[1].legend()
        # self.axs[1].set_title("Accelerations")

        # self.axs[3].cla()
        # self.axs[3].plot(self.time_steps, self.desired_theta_list, label="Desired", color="g")
        # self.axs[3].plot(self.time_steps, self.current_pitch_list, label="Current", color="m")
        # self.axs[3].set_xlabel("Time Steps")
        # self.axs[3].set_ylabel("Pitch")
        # self.axs[3].legend()
        # self.axs[3].set_title("Pitch")

        self.axs[3].cla()
        self.axs[3].plot(self.time_steps, self.desired_theta_list, label="Desired", color="g")
        self.axs[3].plot(self.time_steps, self.current_pitch_list, label="Current", color="m")
        self.axs[3].set_xlabel("Time Steps")
        self.axs[3].set_ylabel("Pitch")
        self.axs[3].legend()
        self.axs[3].set_title("Pitch")

        plt.pause(0.01)  # Refresh the plot in real-time

    def show_plot(self):
        plt.ioff()
        plt.show()

    
    def _resample_commands_pitch(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 3] = gs_rand_float(*self.command_cfg["pitch_ang_range"], (len(envs_idx),), self.device)
        # print("Sampled Linear Velocity x: ", self.commands[envs_idx, 0])
        # print("Sampled Pitch Angle Command: ", self.commands[envs_idx, 3])
       
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

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

    def give_runner_data(self):
        return self.lin_vel_x_range_min, self.lin_vel_x_range_max, self.tracking_lin_vel_rew_mean, self.tracking_lin_vel_rew_threshold_one, self.desired_pitch_mean, self.pitch_count, self.action_rate_scale

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
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
        # self.rot_y = self.base_euler[:, 1]
        self.rot_y = self.base_euler[:, 1].unsqueeze(-1) 
        # print("self.rot_y: ", self.rot_y.shape)

        # print("Pitch: ", self.rot_y)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)

        self.base_lin_vel_x = self.base_lin_vel[:, 0]
        self.base_lin_vel_z = self.base_lin_vel[:, 2]
        # print("Base Linear Velocity x: ", self.base_lin_vel_x)
        # print("Base Linear Velocity z: ", self.base_lin_vel_z)

        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.hip_pos[:] = self.robot.get_dofs_position(self.hip_dofs)
        self.hip_vel[:] = self.robot.get_dofs_velocity(self.hip_dofs)
        self.contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )        

        # print("Base height (better to close to 0.28): ", self.base_pos[:, 2])
        
        # print("episode length: ", self.episode_length_buf)
        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )

        self.post_physics_step_callback()
        if PITCH_CMD:
            # self._resample_commands_pitch(envs_idx)
            self._resample_commands_curriculum_linvel_x(envs_idx)
        elif PITCH_HIGHLEVEL:
            self._resample_commands_from_acc()
            # self.commands[:, 3] = -15
            print("Command: ", self.commands)
        else:
            self._resample_commands(envs_idx)
        # self._resample_commands_max(envs_idx)
        self._randomize_rigids(envs_idx)
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


        self.check_termination()
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.compute_rewards()

        self.compute_observations()

        if self.whole_view:
            self.update_camera_pose()
        else:
            self._render_headless()

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def compute_observations(self):
        sin_phase = torch.sin(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)
        cos_phase = torch.cos(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)

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
        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.rot_y * self.obs_scales["pitch_ang"], # 1
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                sin_phase, #4
                cos_phase, #4
            ],
            axis=-1,
        )
        # compute observations
        self.privileged_obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.rot_y * self.obs_scales["pitch_ang"], # 1
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                sin_phase, #4
                cos_phase #4
            ],
            axis=-1,
        )
        
        self.obs_buf = torch.clip(self.obs_buf, -self.clip_obs, self.clip_obs)
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -self.clip_obs, self.clip_obs)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        self.check_and_sanitize_observations()
        self.second_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.a_count += 1
        # if self.a_count % (1 / (self.dt)) == 0:
        # if self.a_count % (1 / (self.dt * 0.5)) == 0:
        if self.a_count % 1 == 0:
            base_lin_vel_x_temp = self.base_lin_vel_x_low_freq.clone()
            base_lin_vel_z_temp = self.base_lin_vel_z_low_freq.clone()
            self.last_base_lin_vel_x = base_lin_vel_x_temp
            self.last_base_lin_vel_z = base_lin_vel_z_temp

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


    def compute_rewards(self):
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            if name == "action_rate":
                if self.resample_updated:
                    self.action_rate_scale *= self.reward_cfg["action_rate_weight_decay_rate"]
                else:
                    self.reward_scales[name] = self.action_rate_scale
                    
            rew = reward_func() * self.reward_scales[name]

            if name == "tracking_lin_vel":
                self.tracking_lin_vel_rew = rew # torch.Size([4096])
            elif name == "tracking_ang_vel":
                self.tracking_ang_vel_rew = rew
            elif name == "lin_vel_z":
                self.lin_vel_z_rew = rew
            elif name == "action_rate":
                self.action_rate_rew = rew
            elif name == "similar_to_default":
                self.similar_to_default_rew = rew
            elif name == "contact":
                self.contact_rew = rew
            elif name == "constrained_slosh_free":
                self.constrained_slosh_free_rew = rew


            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.reward_cfg["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
        self.resample_updated = False

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf



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

        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        
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
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)

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

        if PITCH_CMD or PITCH_HIGHLEVEL:
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

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
        if PITCH_CMD:
            # self._resample_commands_pitch(envs_idx)
            self._resample_commands_curriculum_linvel_x(envs_idx)
        elif PITCH_HIGHLEVEL:
            self._resample_commands_from_acc()
        else:
            self._resample_commands(envs_idx)
        # self._resample_commands_max(envs_idx)
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
        return self.obs_buf, self.privileged_obs_buf


    def _render_headless(self):
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

    def start_recording(self, record_internal=True):
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


        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_base_mass(self, env_ids):

        min_mass, max_mass = self.env_cfg['added_mass_range']
        base_link_id = 1

        added_mass = gs.rand((len(env_ids), 1), dtype=float) \
                        * (max_mass - min_mass) + min_mass

        self.rigid_solver.set_links_mass_shift(added_mass, [base_link_id,], env_ids)

    def _randomize_com_displacement(self, env_ids):

        min_displacement, max_displacement = self.env_cfg['com_displacement_range']
        base_link_id = 1

        com_displacement = gs.rand((len(env_ids), 1, 3), dtype=float) \
                            * (max_displacement - min_displacement) + min_displacement
        # com_displacement[:, :, 0] -= 0.02

        self.rigid_solver.set_links_COM_shift(com_displacement, [base_link_id,], env_ids)

    def _randomize_motor_strength(self, env_ids):

        min_strength, max_strength = self.env_cfg['motor_strength_range']
        self.motor_strengths[env_ids, :] = gs.rand((len(env_ids), 1), dtype=float) \
                                           * (max_strength - min_strength) + min_strength

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

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    # def _reward_ang_vel_xy(self):
    #     # Penalize xy axes base angular velocity
    #     return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # def _reward_action_rate_2nd_order(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.second_last_actions - 2*self.last_actions + self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    # def _reward_collision(self):
    #     """
    #     Penalize collisions on selected bodies.
    #     Returns the per-env penalty value as a 1D tensor of shape (n_envs,).
    #     """
    #     undesired_forces = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
    #     collisions = (undesired_forces > 0.1).float()  # shape (n_envs, len(...))
    #     # print("collisions: ", collisions)
        
    #     # Sum over those links to get # of collisions per environment
    #     return collisions.sum(dim=1)

    # def _reward_contact_no_vel(self):
    #     # Penalize contact with no velocity
    #     contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    #     # print(contact)
    #     contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
    #     # print(contact_feet_vel)
    #     penalize = torch.square(contact_feet_vel[:, :, :3])
    #     return torch.sum(penalize, dim=(1,2))

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



    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)


    # def _reward_dof_pos_limits(self):
    #     # Penalize dof positions too close to the limit
    #     out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
    #     out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
    #     return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    # def _reward_dof_acc(self):
    #     # Penalize dof accelerations
    #     return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    # def _reward_torques(self):
    #     # Penalize torques
    #     return torch.sum(torch.square(self.torques), dim=1)

    # def _reward_dof_vel(self):
    #     # Penalize dof velocities
    #     return torch.sum(torch.square(self.dof_vel), dim=1)


    # def _reward_torque_limits(self):
    #     # penalize torques too close to the limit
    #     return torch.sum((torch.abs(self.torques) - self.torque_limits*self.soft_torque_limit).clip(min=0.), dim=1)

    # def _reward_termination(self):
    #     # Terminal reward / penalty
    #     return self.reset_buf & ~self.time_out_buf & ~self.out_of_bounds_buf

    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime

    # def _reward_feet_contact_forces(self):
    #     # penalize high contact forces
    #     return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.reward_cfg["max_contact_force"]).clip(min=0.), dim=1)
    


    # def _reward_slosh_free(self):
    #     ax = (self.last_base_lin_vel_x - self.base_lin_vel_x) / self.dt
    #     az = (self.last_base_lin_vel_z - self.base_lin_vel_z) / self.dt
    #     ax *= self.ax_scale
    #     az *= self.az_scale
    #     # Compute the desired pitch angle (theta) in radians
    #     desired_pitch = torch.atan2(ax, az)

    #     # Compute the squared error between desired and current pitch angles
    #     reward = torch.sum(torch.square(desired_pitch - self.rot_y))

    #     return reward

    # def _reward_tracking_pitch_ang(self):
    #     pitch_ang_error = torch.square(self.commands[:, 3] - self.rot_y)
    #     return torch.exp(-pitch_ang_error / self.reward_cfg["tracking_sigma"])
    

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
            return torch.sum(torch.square(desired_pitch - self.rot_y))