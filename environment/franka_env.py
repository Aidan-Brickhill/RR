"""Environment using Gymnasium API for Franka robot.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

This code was also implemented over the repository relay-policy-learning on GitHub (https://github.com/google-research/relay-policy-learning),
published in Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning, by
Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, Karol Hausman

Original Author of the code: Abhishek Gupta & Justin Fu

The modifications made involve separatin the Kitchen environment from the Franka environment and addint support for compatibility with
the Gymnasium and Multi-goal API's

This project is covered by the Apache 2.0 License.
"""

from os import path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from utils import (
    get_config_root_node,
    read_config_from_node,
)

from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames, robot_get_obs

MAX_CARTESIAN_DISPLACEMENT = 0.2
MAX_ROTATION_DISPLACEMENT = 0.5

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.8,
    "azimuth": -90.0,
    "elevation": -35.0,
    "lookat": np.array([0, 0, 0.9]),
}

class FrankaRobot(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        model_path="../assets/franka_assets/chain.xml",
        frame_skip=10,
        robot_noise_ratio: float = 0.01,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )

        self.robot_noise_ratio = robot_noise_ratio

        observation_space = (
            spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32),
        )

        super().__init__(
            xml_file_path,
            frame_skip,
            observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.init_qpos = self.data.qpos
        self.init_qvel = self.data.qvel

        self.act_mid = np.zeros(18)
        self.act_rng = np.ones(18) * 2
        config_path = path.join(
            path.dirname(__file__),
            "../assets/franka_assets/franka_config.xml",
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float64)
        self._read_specs_from_config(config_path)
        self.model_names = MujocoModelNames(self.model)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Denormalize the input action from [-1, 1] range to the each actuators control range
        action = self.act_mid + action * self.act_rng

        # enforce velocity limits
        ctrl_feasible = self._ctrl_velocity_limits(action)
        # enforce position limits
        ctrl_feasible = self._ctrl_position_limits(ctrl_feasible)

        self.do_simulation(action, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()

        return obs, 0.0, False, False, {}
    
    def _collsion_detection(self):

        bad_collisons = []
        good_collisons = []

        for i in range(self.data.ncon):

            contact = self.data.contact[i]
            geom1 = self.model_names.geom_id2name[contact.geom1]
            geom2 = self.model_names.geom_id2name[contact.geom2]


            # collsion between giver fingers
            if (geom1 == "giver_leftfinger_col" and geom2 == "giver_rightfinger_col") or (geom2 == "giver_leftfinger_col" and geom1 == "giver_rightfinger_col"):
                continue
            # collsion between reciever fingers
            if (geom1 == "reciever_rightfinger_col" and geom2 == "reciever_leftfinger_col") or (geom2 == "reciever_rightfinger_col" and geom1 == "reciever_leftfinger_col"):
                continue

            if (("giver_rightfinger_flat_col" in geom1 and "object_collision" == geom2) or ("giver_rightfinger_flat_col" in geom2 and "object_collision" == geom1)):
                if good_collisons.count("inside_giver_robot_rightfinger_object_col") == 0:
                    good_collisons.append("inside_giver_robot_rightfinger_object_col")
                continue  
            if (("giver_leftfinger_flat_col" in geom1 and "object_collision" == geom2) or ("giver_leftfinger_flat_col" in geom2 and "object_collision" == geom1)):
                if good_collisons.count("inside_giver_robot_leftfinger_object_col") == 0:
                    good_collisons.append("inside_giver_robot_leftfinger_object_col")
                continue  

            if (("reciever_rightfinger_flat_col" in geom1 and "object_collision" == geom2) or ("reciever_rightfinger_flat_col" in geom2 and "object_collision" == geom1)):
                if good_collisons.count("inside_reciever_robot_rightfinger_object_col") == 0:
                    good_collisons.append("inside_reciever_robot_rightfinger_object_col")
                continue  
            if (("reciever_leftfinger_flat_col" in geom1 and "object_collision" == geom2) or ("reciever_leftfinger_flat_col" in geom2 and "object_collision" == geom1)):
                if good_collisons.count("inside_reciever_robot_leftfinger_object_col") == 0:
                    good_collisons.append("inside_reciever_robot_leftfinger_object_col")
                continue  


            # collsion between box and table
            if (geom1 == "object_collision" and geom2 == "giver_table_top") or (geom2 == "object_collision" and geom1 == "giver_table_top"):
                bad_collisons.append("object_on_giver_table")
                continue

            # collsion between box and table
            if (geom1 == "object_collision" and geom2 == "reciever_table_top") or (geom2 == "object_collision" and geom1 == "reciever_table_top"):
                bad_collisons.append("object_on_reciever_table")
                continue

            # collsion between giver robot and table
            if ("giver" in geom1 and "table_top" in geom2) or ("giver" in geom2 and "table_top" in geom1):
               
                # collsion between giver robot finger and table
                if ("finger" in geom1 or "finger" in geom2):
                    bad_collisons.append("giver_robot_finger_table_collision")
                    continue
                
                # collsion between giver robot and table
                else:
                    bad_collisons.append("giver_robot_table_collision")
                    continue

            # collsion between reciever robot and table
            if ("reciever" in geom1 and "table_top" in geom2) or ("reciever" in geom2 and "table_top" in geom1):
                
                # collsion between giver robot finger and table
                if ("finger" in geom1 or "finger" in geom2):
                    bad_collisons.append("reciever_robot_finger_table_collision")
                    continue
                
                # collsion between giver robot and table
                else:
                    bad_collisons.append("reciever_robot_table_collision")
                    continue

            # collsion between giver robot and reciever robot
            if ("giver" in geom1 and "reciever" in geom2) or ("giver" in geom2 and "reciever" in geom1):
                bad_collisons.append("robot_collision")
                continue
                
            # collsion between giver robot and object
            if ("giver" in geom1 and geom2 == "object_collision") or ("giver" in geom2 and geom1 == "object_collision"):
               
                # collsion between giver robot finger and object
                if ("finger" in geom1 or "finger" in geom2):
                    good_collisons.append("giver_robot_finger_object_col")
                    continue

                # collsion between giver robot hand and object
                elif ("hand" in geom1 or "hand" in geom2):
                    bad_collisons.append("giver_robot_hand_object_col")
                    continue
                
                # collsion between giver robot link and object
                else:
                    bad_collisons.append("giver_robot_link_object_col")
                    continue
            
            # collsion between reciever robot and object
            if ("reciever" in geom1 and geom2 == "object_collision") or ("reciever" in geom2 and geom1 == "object_collision"):
               
                # collsion between reciever robot finger and object
                if ("finger" in geom1 or "finger" in geom2):
                    good_collisons.append("reciever_robot_finger_object_col")
                    continue

                # collsion between reciever robot hand and object
                elif ("hand" in geom1 or "hand" in geom2):
                    bad_collisons.append("reciever_robot_hand_object_col")
                    continue
                
                # collsion between reciever robot link and object
                else:
                    bad_collisons.append("reciever_robot_link_object_col")
                    continue

            col = geom1 + geom2
        
        return (good_collisons, bad_collisons)

    def _get_obs(self):
        # Gather simulated observation
        robot_qpos, robot_qvel = robot_get_obs(
            self.model, self.data, self.model_names.joint_names
        )

        # end_effector
        end_effector_giver_id = self.model_names.site_name2id["panda_giver_end_effector"]
        robot_giver_end_effector_pos = self.data.site_xpos[end_effector_giver_id].ravel()

        end_effector_receiver_id = self.model_names.site_name2id["panda_reciever_end_effector"]
        robot_reciever_end_effector_pos = self.data.site_xpos[end_effector_receiver_id].ravel()

        # Simulate observation noise
        robot_qpos += (
            self.robot_noise_ratio
            * self.robot_pos_noise_amp[:18]
            * self.np_random.uniform(low=-1.0, high=1.0, size=robot_qpos.shape)
        )
        robot_qvel += (
            self.robot_noise_ratio
            * self.robot_vel_noise_amp[:18]
            * self.np_random.uniform(low=-1.0, high=1.0, size=robot_qvel.shape)
        )

        self._last_robot_qpos = robot_qpos
        return np.concatenate((robot_qpos[:9].copy(), robot_giver_end_effector_pos, robot_qvel[:9].copy(), robot_qpos[9:].copy(), robot_reciever_end_effector_pos, robot_qvel[9:].copy()))

    def reset_model(self):
        qpos = self.init_qpos
        # use for handover env positions
        # qpos[:9] = [2.006107156172305, 
        #             0.10017181958914444, 
        #             -1.9900147359117764, 
        #             -2.0244419418897657, 
        #             0.5669482994122488, 
        #             0.20115351350196695, 
        #             -1.9991158485286418, 
        #             0.03854448190718312, 
        #             0.03303391539854269]
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        obs = self._get_obs()

        return obs

    def _ctrl_velocity_limits(self, ctrl_velocity: np.ndarray):
        """Enforce velocity limits and estimate joint position control input (to achieve the desired joint velocity).

        ALERT: This depends on previous observation. This is not ideal as it breaks MDP assumptions. This is the original
        implementation from the D4RL environment: https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/kitchen/adept_envs/franka/robot/franka_robot.py#L259.

        Args:
            ctrl_velocity (np.ndarray): environment action with space: Box(low=-1.0, high=1.0, shape=(9,))

        Returns:
            ctrl_position (np.ndarray): input joint position given to the MuJoCo simulation actuators.
        """
        ctrl_feasible_vel = np.clip(
            ctrl_velocity, self.robot_vel_bound[:18, 0], self.robot_vel_bound[:18, 1]
        )
        ctrl_feasible_position = self._last_robot_qpos + ctrl_feasible_vel * self.dt
        return ctrl_feasible_position

    def _ctrl_position_limits(self, ctrl_position: np.ndarray):
        """Enforce joint position limits.

        Args:
            ctrl_position (np.ndarray): unbounded joint position control input .

        Returns:
            ctrl_feasible_position (np.ndarray): clipped joint position control input.
        """
        ctrl_feasible_position = np.clip(
            ctrl_position, self.robot_pos_bound[:18, 0], self.robot_pos_bound[:18, 1]
        )
        return ctrl_feasible_position

    def _read_specs_from_config(self, robot_configs: str):
        """Read the specs of the Franka robot joints from the config xml file.
            - pos_bound: position limits of each joint.
            - vel_bound: velocity limits of each joint.
            - pos_noise_amp: scaling factor of the random noise applied in each observation of the robot joint positions.
            - vel_noise_amp: scaling factor of the random noise applied in each observation of the robot joint velocities.

        Args:
            robot_configs (str): path to 'franka_config.xml'
        """
        root, root_name = get_config_root_node(config_file_name=robot_configs)
        self.robot_name = root_name[0]
        self.robot_pos_bound = np.zeros([self.model.nv, 2], dtype=float)
        self.robot_vel_bound = np.zeros([self.model.nv, 2], dtype=float)
        self.robot_pos_noise_amp = np.zeros(self.model.nv, dtype=float)
        self.robot_vel_noise_amp = np.zeros(self.model.nv, dtype=float)

        for i in range(self.model.nv):
            self.robot_pos_bound[i] = read_config_from_node(
                root, "qpos" + str(i), "pos_bound", float
            )
            self.robot_vel_bound[i] = read_config_from_node(
                root, "qpos" + str(i), "vel_bound", float
            )
            self.robot_pos_noise_amp[i] = read_config_from_node(
                root, "qpos" + str(i), "pos_noise_amp", float
            )[0]
            self.robot_vel_noise_amp[i] = read_config_from_node(
                root, "qpos" + str(i), "vel_noise_amp", float
            )[0]
