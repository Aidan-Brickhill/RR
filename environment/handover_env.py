"""Environment using Gymnasium API and Multi-goal API for kitchen and Franka robot.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

This code was also implemented over the repository relay-policy-learning on GitHub (https://github.com/google-research/relay-policy-learning),
published in Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning, by
Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, Karol Hausman.

Original Author of the code: Abhishek Gupta & Justin Fu

The modifications made involve separatin the Kitchen environment from the Franka environment and addint support for compatibility with
the Gymnasium and Multi-goal API's.

This project is covered by the Apache 2.0 License.
"""

from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.core import GoalEnv
from franka_env import FrankaRobot

OBS_ELEMENT_INDICES = {
    "kettle": np.array([18, 19, 20, 21, 22, 23, 24]),
}
# todo - goal position and quaternion
# todo, set multiple goal:
# robot 1 reaches kettle
# robot 1 lifts kettle
# robot 2 reaches kettle
# robot 2 takes kettle
# robot 2 places kettle

OBS_ELEMENT_GOALS = {
    "kettle": np.array([-0.75, -0.4, 0.775, 1, 0.0, 0.0, 0]),
} 
BONUS_THRESH = 0.3
STABILITY_THRESH = 0.3
MAX_HEIGHT = 1.25
MIN_HEIGHT = 0.7
MIN_HANDOVER_HEIGHT = 0.85
w1 = 0.1
w2 = 0.1
w3 = 0.1 
w4 = 0.1
w5 = 0.1


class HandoverEnv(GoalEnv, EzPickle):
    """
    ## Description

    This environment is a used for a two robot arm handover.

    ## Goal

    The goal has a multitask configuration. The multiple tasks to be completed in an episode can be set by passing a list of tasks to the argument`tasks_to_complete`. For example, to open
    the microwave door and move the kettle create the environment as follows:

    The following is a table with all the possible tasks and their respective joint goal values:

    | Task             | Description                                                    | Joint Type | Goal                                     |
    | ---------------- | -------------------------------------------------------------- | ---------- | ---------------------------------------- |
    | "kettle"         | Move the kettle to the top left burner                         | free       | [ 0.75, -0.4, 0.775, 1, 0., 0., 0.] |

    ## Action Space

    The default joint actuators in the Franka MuJoCo model are position controlled. However, the action space of the environment are joint velocities clipped between -1 and 1 rad/s.
    The space is a `Box(-1.0, 1.0, (18,), float32)`. The desired joint position control input is estimated in each time step with the current joint position values and the desired velocity
    action:

    | Num | Action                                         | Action Min  | Action Max  | Joint | Unit  |
    | --- | ---------------------------------------------- | ----------- | ----------- | ----- | ----- |
    | 0   | `robot:panda0_joint1` angular velocity         | -1          | 1           | hinge | rad/s |
    | 1   | `robot:panda0_joint2` angular velocity         | -1          | 1           | hinge | rad/s |
    | 2   | `robot:panda0_joint3` angular velocity         | -1          | 1           | hinge | rad/s |
    | 3   | `robot:panda0_joint4` angular velocity         | -1          | 1           | hinge | rad/s |
    | 4   | `robot:panda0_joint5` angular velocity         | -1          | 1           | hinge | rad/s |
    | 5   | `robot:panda0_joint6` angular velocity         | -1          | 1           | hinge | rad/s |
    | 6   | `robot:panda0_joint7` angular velocity         | -1          | 1           | hinge | rad/s |
    | 7   | `robot:panda0_r_gripper_finger_joint` linear velocity | -1          | 1           | slide | m/s   |
    | 8   | `robot:panda0_l_gripper_finger_joint` linear velocity | -1          | 1           | slide | m/s   |
    | 9   | `robot:panda1_joint1` angular velocity         | -1          | 1           | hinge | rad/s |
    | 10  | `robot:panda1_joint2` angular velocity         | -1          | 1           | hinge | rad/s |
    | 11  | `robot:panda1_joint3` angular velocity         | -1          | 1           | hinge | rad/s |
    | 12  | `robot:panda1_joint4` angular velocity         | -1          | 1           | hinge | rad/s |
    | 13  | `robot:panda1_joint5` angular velocity         | -1          | 1           | hinge | rad/s |
    | 14  | `robot:panda1_joint6` angular velocity         | -1          | 1           | hinge | rad/s |
    | 15  | `robot:panda1_joint7` angular velocity         | -1          | 1           | hinge | rad/s |
    | 16  | `robot:panda1_r_gripper_finger_joint` linear velocity | -1          | 1           | slide | m/s   |
    | 17  | `robot:panda1_l_gripper_finger_joint` linear velocity | -1          | 1           | slide | m/s   |

    ## Observation Space

    The observation is a `goal-aware` observation space. The observation space contains the following keys:

    * `observation`: this is a `Box(-inf, inf, shape=(49,), dtype="float64")` space and it is formed by the robot's joint positions and velocities, as well as
        the pose and velocities of the kettle (object to be handed over). An additional uniform noise of range `[-1,1]` is added to the observations. The noise is also scaled by a factor
        of `robot_noise_ratio` and `object_noise_ratio` given in the environment arguments. The elements of the `observation` array are the following:


    | Num   | Observation                                           | Min      | Max      | Joint Name (in corresponding XML file)   | Joint Type | Unit                       |
    | ----- | ----------------------------------------------------- | -------- | -------- | ---------------------------------------- | ---------- | -------------------------- |
    | 0     | `robot:panda0_joint1` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint1                             | hinge      | angle (rad)                |
    | 1     | `robot:panda0_joint2` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint2                             | hinge      | angle (rad)                |
    | 2     | `robot:panda0_joint3` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint3                             | hinge      | angle (rad)                |
    | 3     | `robot:panda0_joint4` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint4                             | hinge      | angle (rad)                |
    | 4     | `robot:panda0_joint5` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint5                             | hinge      | angle (rad)                |
    | 5     | `robot:panda0_joint6` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint6                             | hinge      | angle (rad)                |
    | 6     | `robot:panda0_joint7` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint7                             | hinge      | angle (rad)                |
    | 7     | `robot:panda0_r_gripper_finger_joint` slide joint translation value   | -Inf     | Inf      | robot:panda0_r_gripper_finger_joint                      | slide      | position (m)               |
    | 8     | `robot:panda0_l_gripper_finger_joint` slide joint translation value   | -Inf     | Inf      | robot:panda0_l_gripper_finger_joint                      | slide      | position (m)               |
    | 9     | `robot:panda0_joint1` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint1                             | hinge      | angular velocity (rad/s)   |
    | 10    | `robot:panda0_joint2` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint2                             | hinge      | angular velocity (rad/s)   |
    | 11    | `robot:panda0_joint3` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint3                             | hinge      | angular velocity (rad/s)   |
    | 12    | `robot:panda0_joint4` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint4                             | hinge      | angular velocity (rad/s)   |
    | 13    | `robot:panda0_joint5` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint5                             | hinge      | angular velocity (rad/s)   |
    | 14    | `robot:panda0_joint6` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint6                             | hinge      | angular velocity (rad/s)   |
    | 15    | `robot:panda0_joint7` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint7                             | hinge      | angle (rad)                |
    | 16    | `robot:panda0_r_gripper_finger_joint` slide joint linear velocity     | -Inf     | Inf      | robot:panda0_r_gripper_finger_joint                      | slide      | linear velocity (m/s)      |
    | 17    | `robot:panda0_l_gripper_finger_joint` slide joint linear velocity     | -Inf     | Inf      | robot:panda0_l_gripper_finger_joint                      | slide      | linear velocity (m/s)      |
    | 18    | `robot:panda1_joint1` hinge joint angle value                | -Inf     | Inf      | robot:panda1_joint1                             | hinge      | angle (rad)                |
    | 19    | `robot:panda1_joint2` hinge joint angle value                | -Inf     | Inf      | robot:panda1_joint2                             | hinge      | angle (rad)                |
    | 20    | `robot:panda1_joint3` hinge joint angle value                | -Inf     | Inf      | robot:panda1_joint3                             | hinge      | angle (rad)                |
    | 21    | `robot:panda1_joint4` hinge joint angle value                | -Inf     | Inf      | robot:panda1_joint4                             | hinge      | angle (rad)                |
    | 22    | `robot:panda1_joint5` hinge joint angle value                | -Inf     | Inf      | robot:panda1_joint5                             | hinge      | angle (rad)                |
    | 23    | `robot:panda1_joint6` hinge joint angle value                | -Inf     | Inf      | robot:panda1_joint6                             | hinge      | angle (rad)                |
    | 24    | `robot:panda1_joint7` hinge joint angle value                | -Inf     | Inf      | robot:panda1_joint7                             | hinge      | angle (rad)                |
    | 25    | `robot:panda1_r_gripper_finger_joint` slide joint translation value   | -Inf     | Inf      | robot:panda1_r_gripper_finger_joint                      | slide      | position (m)               |
    | 26    | `robot:panda1_l_gripper_finger_joint` slide joint translation value   | -Inf     | Inf      | robot:panda1_l_gripper_finger_joint                      | slide      | position (m)               |
    | 27    | `robot:panda1_joint1` hinge joint angular velocity           | -Inf     | Inf      | robot:panda1_joint1                             | hinge      | angular velocity (rad/s)   |
    | 28    | `robot:panda1_joint2` hinge joint angular velocity           | -Inf     | Inf      | robot:panda1_joint2                             | hinge      | angular velocity (rad/s)   |
    | 29    | `robot:panda1_joint3` hinge joint angular velocity           | -Inf     | Inf      | robot:panda1_joint3                             | hinge      | angular velocity (rad/s)   |
    | 30    | `robot:panda1_joint4` hinge joint angular velocity           | -Inf     | Inf      | robot:panda1_joint4                             | hinge      | angular velocity (rad/s)   |
    | 31    | `robot:panda1_joint5` hinge joint angular velocity           | -Inf     | Inf      | robot:panda1_joint5                             | hinge      | angular velocity (rad/s)   |
    | 32    | `robot:panda1_joint6` hinge joint angular velocity           | -Inf     | Inf      | robot:panda1_joint6                             | hinge      | angular velocity (rad/s)   |
    | 33    | `robot:panda1_joint7` hinge joint angular velocity           | -Inf     | Inf      | robot:panda1_joint7                             | hinge      | angle (rad)                |
    | 34    | `robot:panda1_r_gripper_finger_joint` slide joint linear velocity     | -Inf     | Inf      | robot:panda1_r_gripper_finger_joint                      | slide      | linear velocity (m/s)      |
    | 35    | `robot:panda1_l_gripper_finger_joint` slide joint linear velocity     | -Inf     | Inf      | robot:panda1_l_gripper_finger_joint                      | slide      | linear velocity (m/s)      |    
    | 36    | Kettle's x coordinate                                 | -Inf     | Inf      | kettle                                   | free       | position (m)               |
    | 37    | Kettle's y coordinate                                 | -Inf     | Inf      | kettle                                   | free       | position (m)               |
    | 38    | Kettle's z coordinate                                 | -Inf     | Inf      | kettle                                   | free       | position (m)               |
    | 39    | Kettle's x quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 40    | Kettle's y quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 41    | Kettle's z quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 42    | Kettle's w quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 43    | Kettle's x linear velocity                            | -Inf     | Inf      | kettle                                   | free       | linear velocity (m/s)      |
    | 44    | Kettle's y linear velocity                            | -Inf     | Inf      | kettle                                   | free       | linear velocity (m/s)      |
    | 45    | Kettle's z linear velocity                            | -Inf     | Inf      | kettle                                   | free       | linear velocity (m/s)      |
    | 46    | Kettle's x axis angular rotation                      | -Inf     | Inf      | kettle                                   | free       | angular velocity(rad/s)    |
    | 47    | Kettle's y axis angular rotation                      | -Inf     | Inf      | kettle                                   | free       | angular velocity(rad/s)    |
    | 48    | Kettle's z axis angular rotation                      | -Inf     | Inf      | kettle                                   | free       | angular velocity(rad/s)    |

    * `desired_goal`: this key represents the final goal to be achieved. The value is another `Dict` space with keys the tasks to be completed in the episode and values the joint
    goal configuration of each joint in the task as specified in the `Goal` section.

    * `achieved_goal`: this key represents the current state of the tasks. The value is another `Dict` space with keys the tasks to be completed in the episode and values the
    current joint configuration of each joint in the task.

    ## Info

    The environment also returns an `info` dictionary in each Gymnasium step. The keys are:

    - `tasks_to_complete` (list[str]): list of tasks that haven't yet been completed in the current episode.
    - `step_task_completions` (list[str]): list of tasks completed in the step taken.
    - `episode_task_completions` (list[str]): list of tasks completed during the episode uptil the current step.

    ## Rewards

    The environment's reward is `sparse`. The reward in each Gymnasium step is equal to the number of task completed in the given step. If no task is completed the returned reward will be zero.
    The tasks are considered completed when their joint configuration is within a norm threshold of `0.3` with respect to the goal configuration specified in the `Goal` section.

    ## Starting State

    The simulation starts with all of the joint position actuators of the Franka robot set to zero. The kettle will be placed at the edge of the table.

    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 280 timesteps.
    The episode is `terminated` when all the tasks have been completed unless the `terminate_on_tasks_completed` argument is set to `False`.

    ## Arguments

    The following arguments can be passed when initializing the environment with `gymnasium.make` kwargs:

    | Parameter                      | Type            | Default                                     | Description                                                                                                                                                               |
    | -------------------------------| --------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- |
    | `tasks_to_complete`            | **list[str]**   | All possible goal tasks. Go to Goal section | The goal tasks to reach in each episode                                             |
    | `terminate_on_tasks_completed` | **bool**        | `True`                                      | Terminate episode if no more tasks to complete (episodic multitask)                 |
    | `remove_task_when_completed`   | **bool**        | `True`                                      | Remove the completed tasks from the info dictionary returned after each step        |
    | `object_noise_ratio`           | **float**       | `0.0005`                                    | Scaling factor applied to the uniform noise added to the kitchen object observations|
    | `robot_noise_ratio`            | **float**       | `0.01`                                      | Scaling factor applied to the uniform noise added to the robot joint observations   |
    | `max_episode_steps`            | **integer**     | `280`                                       | Maximum number of steps per episode                                                 |

    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 12,
    }

    def __init__(
        self,
        tasks_to_complete: "list[str]" = list(OBS_ELEMENT_GOALS.keys()),
        terminate_on_tasks_completed: bool = True,
        remove_task_when_completed: bool = True,
        object_noise_ratio: float = 0.0005,
        **kwargs,
    ):
        self.robot_env = FrankaRobot(
            model_path="../assets/handover_assets/handover_env_model.xml",
            **kwargs,
        )

        self.model = self.robot_env.model
        self.data = self.robot_env.data
        self.render_mode = self.robot_env.render_mode

        self.terminate_on_tasks_completed = terminate_on_tasks_completed
        self.remove_task_when_completed = remove_task_when_completed

        self.goal = {}
        self.tasks_to_complete = set(tasks_to_complete)
        # Validate list of tasks to complete
        for task in tasks_to_complete:
            if task not in OBS_ELEMENT_GOALS.keys():
                raise ValueError(
                    f"The task {task} cannot be found the the list of possible goals: {OBS_ELEMENT_GOALS.keys()}"
                )
            else:
                self.goal[task] = OBS_ELEMENT_GOALS[task]

        self.step_task_completions = (
            []
        )  # Tasks completed in the current environment step
        self.episode_task_completions = (
            []
        )  # Tasks completed that have been completed in the current episode
        self.object_noise_ratio = (
            object_noise_ratio  # stochastic noise added to the object observations
        )

        robot_obs = self.robot_env._get_obs()
        obs = self._get_obs(robot_obs)

        assert (
            int(np.round(1.0 / self.robot_env.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.robot_env.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.action_space = self.robot_env.action_space
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Dict(
                    {
                        task: spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=goal.shape,
                            dtype="float64",
                        )
                        for task, goal in obs["achieved_goal"].items()
                    }
                ),
                achieved_goal=spaces.Dict(
                    {
                        task: spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=goal.shape,
                            dtype="float64",
                        )
                        for task, goal in obs["achieved_goal"].items()
                    }
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

        EzPickle.__init__(
            self,
            tasks_to_complete,
            terminate_on_tasks_completed,
            remove_task_when_completed,
            object_noise_ratio,
            **kwargs,
        )

    def compute_reward(
        self,
        achieved_goal: "dict[str, np.ndarray]",
        desired_goal: "dict[str, np.ndarray]",
        info: "dict[str, Any]",
    ):
        # self.step_task_completions.clear()
        # for task in self.tasks_to_complete:
        #     distance = np.linalg.norm(achieved_goal[task] - desired_goal[task])
        #     complete = distance < BONUS_THRESH
        #     if complete:
        #         self.step_task_completions.append(task)

        # return float(len(self.step_task_completions))

        # Initialize reward components
        r_distance = 0
        r_stability = 0
        r_height = 0
        r_collision = 0
        r_drop = 0

        # Distance reward
        distance = np.linalg.norm(achieved_goal["kettle"][:3] - desired_goal["kettle"][:3])
        if distance < BONUS_THRESH:
            r_distance = 1.0
        else:
            r_distance = -1.0

        # Stability reward
        stability = np.abs(achieved_goal["kettle"][3:6] - desired_goal["kettle"][3:6]).sum()
        if stability < STABILITY_THRESH:
            r_stability = 1.0
        else:
            r_stability = -1.0

        # Height reward
        height = achieved_goal["kettle"][2]
        if MIN_HANDOVER_HEIGHT < height < MAX_HEIGHT:
            r_height = 1.0
        else:
            r_height = -1.0

        # Collision penalty todo
        # if self._check_collision():
        #     r_collision = 10.0

        # Drop penalty
        if height < MIN_HEIGHT:
            r_drop = 10.0


        # Compute total reward
        return (
            w1 * r_distance
            + w2 * r_stability
            + w3 * r_height
            - w4 * r_collision
            - w5 * r_drop
        )

        # # Check if the kettle is within the desired goal threshold
        # for task in self.tasks_to_complete:
        #     distance = np.linalg.norm(achieved_goal[task] - desired_goal[task])
        #     complete = distance < BONUS_THRESH
        #     if complete:
        #         self.step_task_completions.append(task)

        # return reward

    def _get_obs(self, robot_obs):
        obj_qpos = self.data.qpos[18:].copy()
        obj_qvel = self.data.qvel[18:].copy()

        # Simulate observation noise
        obj_qpos += (
            self.object_noise_ratio
            * self.robot_env.robot_pos_noise_amp[17:]
            * self.robot_env.np_random.uniform(low=-1.0, high=1.0, size=obj_qpos.shape)
        )
        obj_qvel += (
            self.object_noise_ratio
            * self.robot_env.robot_vel_noise_amp[18:]
            * self.robot_env.np_random.uniform(low=-1.0, high=1.0, size=obj_qvel.shape)
        )

        achieved_goal = {
            task: self.data.qpos[OBS_ELEMENT_INDICES[task]] for task in self.goal.keys()
        }

        obs = {
            "observation": np.concatenate((robot_obs, obj_qpos, obj_qvel)),
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal,
        }

        return obs

    def step(self, action):
        robot_obs, _, terminated, truncated, info = self.robot_env.step(action)
        obs = self._get_obs(robot_obs)

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        if self.remove_task_when_completed:
            # When the task is accomplished remove from the list of tasks to be completed
            [
                self.tasks_to_complete.remove(element)
                for element in self.step_task_completions
            ]

        info = {"tasks_to_complete": list(self.tasks_to_complete)}
        info["step_task_completions"] = self.step_task_completions.copy()

        for task in self.step_task_completions:
            if task not in self.episode_task_completions:
                self.episode_task_completions.append(task)
        info["episode_task_completions"] = self.episode_task_completions
        if self.terminate_on_tasks_completed:
            # terminate if there are no more tasks to complete
            terminated = len(self.episode_task_completions) == len(self.goal.keys())

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.episode_task_completions.clear()
        robot_obs, _ = self.robot_env.reset(seed=seed)
        obs = self._get_obs(robot_obs)
        self.tasks_to_complete = set(self.goal.keys())
        info = {
            "tasks_to_complete": list(self.tasks_to_complete),
            "episode_task_completions": [],
            "step_task_completions": [],
        }

        return obs, info

    def render(self):
        return self.robot_env.render()

    def close(self):
        self.robot_env.close()

