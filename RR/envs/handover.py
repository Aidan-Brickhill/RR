from typing import Dict, Union

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}


class PusherEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "handover_scene.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        # reward_near_weight: float = 0.5,
        # reward_dist_weight: float = 1,
        # reward_control_weight: float = 0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            # reward_near_weight,
            # reward_dist_weight,
            # reward_control_weight,
            **kwargs,
        )
        # self._reward_near_weight = reward_near_weight
        # self._reward_dist_weight = reward_dist_weight
        # self._reward_control_weight = reward_control_weight

        # num robots =                      2
        # num joints =                      7
        # x,y,z position of joints          3
        # x,y,z orientation of joints       3
        # x,y,z velocity of joints          3
        # total                            126

        # num fingers =                     4
        # x,y,z position of fingers         3
        # total                             12

        # x,y,z position of object          3
        # x,y,z orientation of object       3
        # x,y,z velocity of object          3
        # total                             9

        observation_space = Box(low=-np.inf, high=np.inf, shape=(147,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, action):
        # vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        # vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        # reward_near = -np.linalg.norm(vec_1) * self._reward_near_weight
        # reward_dist = -np.linalg.norm(vec_2) * self._reward_dist_weight
        # reward_ctrl = -np.square(action).sum() * self._reward_control_weight

        # reward = reward_dist + reward_ctrl + reward_near

        # reward_info = {
        #     "reward_dist": reward_dist,
        #     "reward_ctrl": reward_ctrl,
        #     "reward_near": reward_near,
        # }

        return 0, [0]

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos,
                self.data.qvel,
                self.get_body_com("object"),
            ]
        )