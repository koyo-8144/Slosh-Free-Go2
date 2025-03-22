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

import time




class WaterEnv:
    def __init__(self, show_viewer=True, device="cuda"):
        self.device = torch.device(device)
        
        self.dt = 1 / 50
        sim_dt = self.dt / 4
        sim_substeps = 1

        gs.init()

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=sim_dt,
                # dt=self.dt,
                substeps=sim_substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * 4),
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
            sph_options=gs.options.SPHOptions(
                lower_bound=(-0.5, -0.5, -2.0),
                upper_bound=(0.5, 0.5, 2.0),
                particle_size=0.01,
            ),
            show_viewer=False,
        )
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

        self.show_vis = True
        self.selected_robot = 0
        if show_viewer:
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


        self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.base_init_pos = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # self.robot  = self.scene.add_entity(
        #     gs.morphs.MJCF(
        #     file="xml/go2/go2.xml",
        #     pos=self.base_init_pos.cpu().numpy(),
        #     quat=self.base_init_quat.cpu().numpy(),
        #     ),
        # )

        self.liquid = self.scene.add_entity(
            # viscous liquid
            # material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
            material=gs.materials.SPH.Liquid(),
            morph=gs.morphs.Cylinder(
                pos=(0.0, 0.0, 0.0),  # Lowered initial position
                height=0.3,  # Reduced height to fit inside boundary
                radius=0.5,  # Adjusted radius for better containment
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.8, 1.0),
                vis_mode="particle",
                # vis_mode="recon",
            ),
        )


        # build
        self.scene.build()

        self.init_camera_params()

        self.base_pos = torch.zeros((3,), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((4,), device=self.device, dtype=gs.tc_float)

        self.update_count = 0

    def init_camera_params(self):
        self.whole_view = False

        # Initialize rotation parameters
        self.radius = 30.0  # Radius of circular path
        self.theta = 0.0  # Initial angle
        self.theta_increment = np.radians(2)  # Increment angle by 2 degrees

        # Fixed target (lookat) position
        self.target = np.array([0.0, 0.0, 0.5])  # Assume robot is at this position
        self.camera_height = 0.0  # Fixed z-axis position for top-down view
    
        
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
            x, y, z = self.base_pos[0].cpu().numpy(), self.base_pos[1].cpu().numpy(), self.base_pos[2].cpu().numpy()  # Convert the tensor to NumPy
            self.cam_0.set_pose(pos=(x+5.0, y, z+5.5), lookat=(x, y, z+0.5))
            self.cam_0.render(
                rgb=True,
            )


    def step(self):
        if self.whole_view:
            self.update_camera_pose()
        else:
            self._render_headless()

        # Advance simulation step
        self.scene.step()

        self.update_count += 1
        if self.update_count % 100 == 0:
            # Get current time or frame
            f = self.scene.sim.cur_substep_local

            # Generate dynamic offset (e.g., up and down using sin wave)
            t = self.scene.sim.cur_step_global * self.dt
            z_offset = 0.1 * math.sin(2 * math.pi * 0.25 * t)  # 0.25 Hz

            # Create new position tensor
            new_pos = gs.zeros((self.liquid.n_particles, 3), dtype=float)
            for i in range(self.liquid.n_particles):
                # Sample original particle positions from morph and offset them
                new_pos[i] = torch.tensor([0.0, 0.0, z_offset], device=new_pos.device, dtype=new_pos.dtype)


            # Set new position
            self.liquid.set_pos(f, new_pos)


if __name__ == "__main__":
    env = WaterEnv(show_viewer=True)  # or False for headless
    for _ in range(20000000):
        env.step()
        time.sleep(0.01)  # small delay (~100 FPS cap)