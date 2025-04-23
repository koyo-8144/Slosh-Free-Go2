import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import os
import pickle
import torch
import cv2
import numpy as np
from legged_env import LeggedEnv
from legged_sf_env import LeggedSfEnv
from rsl_rl.runners import OnPolicyRunner
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs
import re
import copy

PITCH_COMMAND = 0

SLOSH_FREE = 1
NO_ACC_SAMPLE = 0
NO_SLOSH_FREE = 0
LATEST = 0

class Go2EvaluationNode(Node):
    def __init__(self):
        super().__init__('go2_evaluation_node')
        self.device="cuda:0"

        # Declare and retrieve parameters
        self.declare_parameter('exp_name', 'go2_slosh_free_v3')
        # self.declare_parameter('exp_name', 'paper')
        self.declare_parameter('ckpt', 10000)

        self.exp_name = self.get_parameter('exp_name').value
        self.ckpt = self.get_parameter('ckpt').value

        # Publishers and subscribers
        self.teleop_sub = self.create_subscription(Twist, '/cmd_vel', self.teleop_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Initialize Genesis and environment
        gs.init(backend=gs.cuda)

        log_dir = f"/home/psxkf4/Genesis/logs/{self.exp_name}"
        # Get all subdirectories in the base log directory
        subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        # Sort subdirectories by their names (assuming they are timestamped in lexicographical order)
        most_recent_subdir = sorted(subdirs)[-1] if subdirs else None
        log_dir = os.path.join(log_dir, most_recent_subdir)

        # log_dir = "/home/psxkf4/Genesis/logs/paper/no_slosh_free_MLP"
        # log_dir = "/home/psxkf4/Genesis/logs/paper/slosh_free_acc_profile_sigma03_LSTM"
        # log_dir = "/home/psxkf4/Genesis/logs/paper/slosh_free_acc_profile_sigma03_MLP"
        # log_dir = "/home/psxkf4/Genesis/logs/paper/slosh_free_no_acc_profile_LSTM"
        # log_dir = "/home/psxkf4/Genesis/logs/paper/slosh_free_no_acc_profile_MLP"
        # log_dir = "/home/psxkf4/Genesis/logs/paper/slosh_free_no_acc_profile_MLP_pitch_obs"
        # log_dir = "/home/psxkf4/Genesis/logs/paper/slosh_free_acc_profile_sigma03_MLP_clip025_dkl0005"
        # log_dir = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250410_112423"
        # log_dir = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250417_014903"
        # log_dir ="/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250410_112423"

        if SLOSH_FREE:
            # log_dir = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250417_210914"
            # log_dir = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250421_205405"
            # log_dir = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250421_122056"
            # log_dir = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250422_131828"

            # log_dir = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250422_180004"
            # log_dir = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250423_044137"
            log_dir = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250423_130100"
        elif NO_ACC_SAMPLE:
            log_dir = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250423_022456"
        elif NO_SLOSH_FREE:
            # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250421_175410/model_10000.pt"
            resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250423_072540"
        elif LATEST:
            log_dir = log_dir

        
        folder_name = log_dir.split("/")[-1]

        env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, train_cfg, terrain_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
        # env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg = get_cfgs()
        # train_cfg = get_train_cfg("slr", "100000")
        reward_cfg["reward_scales"] = {}
        # print("reward_cfg: ", reward_cfg)
        # breakpoint()
        env_cfg["robot_mjcf"] = "xml/go2/go2.xml"

        # print("env_cfg: ", env_cfg)
        # print("obs_cfg: ", obs_cfg)
        # print("reward_cfg: ", reward_cfg)
        # print("command_cfg: ", command_cfg)
        # breakpoint()

        self.env = LeggedSfEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            noise_cfg=noise_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            terrain_cfg=terrain_cfg,
            folder_name=folder_name,
            show_viewer=True,
        )

        runner = OnPolicyRunner(self.env, train_cfg, log_dir, device="cuda:0")

        # List all files in the most recent subdirectory
        files = os.listdir(log_dir)

        # Regex to match filenames like 'model_100.pt' and extract the number
        model_files = [(f, int(re.search(r'model_(\d+)\.pt', f).group(1)))
                    for f in files if re.search(r'model_(\d+)\.pt', f)]
        model_file = max(model_files, key=lambda x: x[1])[0]

        resume_path = os.path.join(log_dir, model_file)

        # resume_path = "/home/psxkf4/Genesis/logs/paper/slosh_free_acc_profile_sigma03_LSTM/model_10000.pt"
        # resume_path = "/home/psxkf4/Genesis/logs/paper/slosh_free_acc_profile_sigma03_MLP/model_10000.pt"
        # resume_path = "/home/psxkf4/Genesis/logs/paper/slosh_free_no_acc_profile_LSTM/model_10000.pt"
        # resume_path = "/home/psxkf4/Genesis/logs/paper/slosh_free_no_acc_profile_MLP/model_8000.pt"
        # resume_path = "/home/psxkf4/Genesis/logs/paper/slosh_free_no_acc_profile_MLP_pitch_obs/model_6000.pt"
        # resume_path = "/home/psxkf4/Genesis/logs/paper/slosh_free_acc_profile_sigma03_MLP_clip025_dkl0005/model_12000.pt"

        # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250409_203321/model_10000.pt"
        # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250411_012456/model_10000.pt"
        # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250410_112423/model_10000.pt"
        # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250416_181016/model_1000.pt"
        # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250417_014903/model_8000.pt"
        # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250410_112423/model_10000.pt"
        # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250418_015930/model_6000.pt"

        if SLOSH_FREE:
            # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250417_210914/model_10000.pt"
            # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250421_205405/model_10000.pt"
            # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250421_122056/model_10000.pt"
            # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250422_131828/model_10000.pt"
            
            # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250422_180004/model_10000.pt"
            # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250423_044137/model_10000.pt"
            resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250423_130100/model_10000.pt"
        elif NO_ACC_SAMPLE:
            resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250423_022456/model_10000.pt"
        elif NO_SLOSH_FREE:
            # resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250421_175410/model_10000.pt"
            resume_path = "/home/psxkf4/Genesis/logs/go2_slosh_free_v3/20250423_072540/model_10000.pt"
        elif LATEST:
            resume_path = resume_path

        runner.load(resume_path)
        # resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        # runner.load(resume_path)
        self.policy = runner.get_inference_policy(device="cuda:0")
        # export policy as a jit module (used to run it from C++)
        EXPORT_POLICY = True
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

        if PITCH_COMMAND:
            self.teleop_commands = torch.zeros((1, 4), device=self.device) 
        else:
            self.teleop_commands = torch.zeros((1, 3), device=self.device)  # 3 for [x, y, z]

        self.obs, _ = self.env.reset()


    def teleop_callback(self, msg: Twist):
        """
        Update self.teleop_commands based on teleop input.
        """
        # Ensure self.teleop_commands is 2D: [num_envs, 3]
        # print("msg: ", msg) # msg:  geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=-0.6050000000000001, y=0.0, z=0.0), angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0))
        self.teleop_commands[:, 0] = msg.linear.x  # Forward/Backward
        self.teleop_commands[:, 1] = msg.linear.y  # Left/Right
        self.teleop_commands[:, 2] = msg.angular.z  # Rotational velocity


    def control_loop(self):
        # Update commands in the environment
        self.env.commands = self.teleop_commands  # Pass updated commands to the environment
        # self.env.commands = torch.zeros((1, 3), device=self.device)

        with torch.no_grad():
            self.actions = self.policy(self.obs)
            self.obs, _, self.rews, self.dones, infos = self.env.step(self.actions)
    


def main(args=None):
    rclpy.init(args=args)
    node = Go2EvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
# ROS 2 evaluation example:
ros2 run <package_name> go2_evaluation_node --ros-args -p exp_name:=go2-walking -p ckpt:=100
"""
