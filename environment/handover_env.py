from typing import Any, Optional
import gymnasium as gym

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.core import GoalEnv
from franka_env import FrankaRobot


OBS_ELEMENT_INDICES = {
    "panda_giver_fetch": np.array([9, 10, 11]),
    "panda_giver_lift": np.array([9, 10, 11, 36, 37, 38]),
    "panda_reciever_wait": np.array([30, 31, 32]),
    "panda_reciever_fetch": np.array([30, 31, 32]),
    "panda_reciever_place": np.array([30, 31, 32, 36, 37, 38]),
}

OBS_ELEMENT_GOALS = {
    "panda_giver_fetch": np.array([-0.648, 0.4, 0.95]),
    "panda_giver_lift": np.array([0, 0, 0.85, 0, 0, 0.85]),
    "panda_reciever_wait": np.array([0.47, 0.01, 1.84]),
    "panda_reciever_fetch": np.array([0, 0, 0.8]),
    "panda_reciever_place": np.array([-0.75, -0.4, 0.8, -0.75, -0.4, 0.775]),
} 

PANDA_GIVER_FETCH_THRESH = 0.1
PANDA_GIVER_LIFT_THRESH = 0.1
OBJECT_LIFT_THRESH = 0.3

PANDA_RECIEVER_WAIT_THRESH= 0.1
PANDA_RECIEVER_FETCH_THRESH = 0.1
PANDA_RECIEVER_PLACE_THRESH = 0.1
OBJECT_PLACE_THRESH = 0.1

MAX_HEIGHT = 1.25
MIN_HEIGHT = 0.7
MIN_HANDOVER_HEIGHT = 0.85

STABILITY_THRESH = 0.3
END_EFFECTOR_DISTANCE_THRESH = 0.3



class HandoverEnv(gym.Env, EzPickle):
    """
    ## Description

    This environment has been created in order to learn a safe handover between two robot arms. It has been heavily inspired by the  
    Frank Kitchen Environment. Their code (https://github.com/Farama-Foundation/Gymnasium-Robotics) was used as a base in order to implement this environment.

    ## Goal

    The goal has a multitask configuration. The multiple tasks to be completed in an episode can be set by passing a list of tasks to the argument`tasks_to_complete`. For example, to open
    the microwave door and move the kettle create the environment as follows:

    The following is a table with all the possible tasks and their respective joint goal values:

    | Task             | Description                                                    | Joint Type | Goal                                     |
    | ---------------- | -------------------------------------------------------------- | ---------- | ---------------------------------------- |
    | "kettle"         | Move the kettle from one table to the next                     | free       | [ 0.75, -0.4, 0.775, 1, 0., 0., 0.] |

    ## Action Space

    The default joint actuators in the Franka MuJoCo model are position controlled. However, the action space of the environment are joint velocities clipped between -1 and 1 rad/s.
    The space is a `Box(-1.0, 1.0, (18,), float32)`. The desired joint position control input is estimated in each time step with the current joint position values and the desired velocity
    action:

    | Num | Action                                         | Action Min  | Action Max  | Joint | Unit  |
    | --- | ---------------------------------------------- | ----------- | ----------- | ----- | ----- |
    | 0   | `robot:panda_giver_joint1` angular velocity         | -1          | 1           | hinge | rad/s |
    | 1   | `robot:panda_giver_joint2` angular velocity         | -1          | 1           | hinge | rad/s |
    | 2   | `robot:panda_giver_joint3` angular velocity         | -1          | 1           | hinge | rad/s |
    | 3   | `robot:panda_giver_joint4` angular velocity         | -1          | 1           | hinge | rad/s |
    | 4   | `robot:panda_giver_joint5` angular velocity         | -1          | 1           | hinge | rad/s |
    | 5   | `robot:panda_giver_joint6` angular velocity         | -1          | 1           | hinge | rad/s |
    | 6   | `robot:panda_giver_joint7` angular velocity         | -1          | 1           | hinge | rad/s |
    | 7   | `robot:panda_giver_r_gripper_finger_joint` linear velocity | -1          | 1           | slide | m/s   |
    | 8   | `robot:panda_giver_l_gripper_finger_joint` linear velocity | -1          | 1           | slide | m/s   |
    | 9   | `robot:panda_reciever_joint1` angular velocity         | -1          | 1           | hinge | rad/s |
    | 10  | `robot:panda_reciever_joint2` angular velocity         | -1          | 1           | hinge | rad/s |
    | 11  | `robot:panda_reciever_joint3` angular velocity         | -1          | 1           | hinge | rad/s |
    | 12  | `robot:panda_reciever_joint4` angular velocity         | -1          | 1           | hinge | rad/s |
    | 13  | `robot:panda_reciever_joint5` angular velocity         | -1          | 1           | hinge | rad/s |
    | 14  | `robot:panda_reciever_joint6` angular velocity         | -1          | 1           | hinge | rad/s |
    | 15  | `robot:panda_reciever_joint7` angular velocity         | -1          | 1           | hinge | rad/s |
    | 16  | `robot:panda_reciever_r_gripper_finger_joint` linear velocity | -1          | 1           | slide | m/s   |
    | 17  | `robot:panda_reciever_l_gripper_finger_joint` linear velocity | -1          | 1           | slide | m/s   |

    ## Observation Space

    The observation is a `goal-aware` observation space. The observation space contains the following keys:

    * `observation`: this is a `Box(-inf, inf, shape=(49,), dtype="float64")` space and it is formed by the robot's joint positions and velocities, as well as
        the pose and velocities of the kettle (object to be handed over). An additional uniform noise of range `[-1,1]` is added to the observations. The noise is also scaled by a factor
        of `robot_noise_ratio` and `object_noise_ratio` given in the environment arguments. The elements of the `observation` array are the following:


    | Num   | Observation                                           | Min      | Max      | Joint Name (in corresponding XML file)   | Joint Type | Unit                       |
    | ----- | ----------------------------------------------------- | -------- | -------- | ---------------------------------------- | ---------- | -------------------------- |
    | 0     | `robot:panda_giver_joint1` hinge joint angle value                | -Inf     | Inf      | robot:panda_giver_joint1                             | hinge      | angle (rad)                |
    | 1     | `robot:panda_giver_joint2` hinge joint angle value                | -Inf     | Inf      | robot:panda_giver_joint2                             | hinge      | angle (rad)                |
    | 2     | `robot:panda_giver_joint3` hinge joint angle value                | -Inf     | Inf      | robot:panda_giver_joint3                             | hinge      | angle (rad)                |
    | 3     | `robot:panda_giver_joint4` hinge joint angle value                | -Inf     | Inf      | robot:panda_giver_joint4                             | hinge      | angle (rad)                |
    | 4     | `robot:panda_giver_joint5` hinge joint angle value                | -Inf     | Inf      | robot:panda_giver_joint5                             | hinge      | angle (rad)                |
    | 5     | `robot:panda_giver_joint6` hinge joint angle value                | -Inf     | Inf      | robot:panda_giver_joint6                             | hinge      | angle (rad)                |
    | 6     | `robot:panda_giver_joint7` hinge joint angle value                | -Inf     | Inf      | robot:panda_giver_joint7                             | hinge      | angle (rad)                |
    | 7     | `robot:panda_giver_r_gripper_finger_joint` slide joint translation value   | -Inf     | Inf      | robot:panda_giver_r_gripper_finger_joint                      | slide      | position (m)               |
    | 8     | `robot:panda_giver_l_gripper_finger_joint` slide joint translation value   | -Inf     | Inf      | robot:panda_giver_l_gripper_finger_joint                      | slide      | position (m)               |
    9
    10
    11
    +3+ x,y,z end effector pos robot 0
    
    | 12     | `robot:panda_giver_joint1` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_giver_joint1                             | hinge      | angular velocity (rad/s)   |
    | 13    | `robot:panda_giver_joint2` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_giver_joint2                             | hinge      | angular velocity (rad/s)   |
    | 14    | `robot:panda_giver_joint3` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_giver_joint3                             | hinge      | angular velocity (rad/s)   |
    | 15    | `robot:panda_giver_joint4` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_giver_joint4                             | hinge      | angular velocity (rad/s)   |
    | 16    | `robot:panda_giver_joint5` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_giver_joint5                             | hinge      | angular velocity (rad/s)   |
    | 17    | `robot:panda_giver_joint6` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_giver_joint6                             | hinge      | angular velocity (rad/s)   |
    | 18    | `robot:panda_giver_joint7` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_giver_joint7                             | hinge      | angle (rad)                |
    | 19    | `robot:panda_giver_r_gripper_finger_joint` slide joint linear velocity     | -Inf     | Inf      | robot:panda_giver_r_gripper_finger_joint                      | slide      | linear velocity (m/s)      |
    | 20    | `robot:panda_giver_l_gripper_finger_joint` slide joint linear velocity     | -Inf     | Inf      | robot:panda_giver_l_gripper_finger_joint                      | slide      | linear velocity (m/s)      |    
    
    | 21    | `robot:panda_reciever_joint1` hinge joint angle value                | -Inf     | Inf      | robot:panda_reciever_joint1                             | hinge      | angle (rad)                |
    | 22    | `robot:panda_reciever_joint2` hinge joint angle value                | -Inf     | Inf      | robot:panda_reciever_joint2                             | hinge      | angle (rad)                |
    | 23    | `robot:panda_reciever_joint3` hinge joint angle value                | -Inf     | Inf      | robot:panda_reciever_joint3                             | hinge      | angle (rad)                |
    | 24    | `robot:panda_reciever_joint4` hinge joint angle value                | -Inf     | Inf      | robot:panda_reciever_joint4                             | hinge      | angle (rad)                |
    | 25    | `robot:panda_reciever_joint5` hinge joint angle value                | -Inf     | Inf      | robot:panda_reciever_joint5                             | hinge      | angle (rad)                |
    | 26    | `robot:panda_reciever_joint6` hinge joint angle value                | -Inf     | Inf      | robot:panda_reciever_joint6                             | hinge      | angle (rad)                |
    | 27    | `robot:panda_reciever_joint7` hinge joint angle value                | -Inf     | Inf      | robot:panda_reciever_joint7                             | hinge      | angle (rad)                |
    | 28    | `robot:panda_reciever_r_gripper_finger_joint` slide joint translation value   | -Inf     | Inf      | robot:panda_reciever_r_gripper_finger_joint                      | slide      | position (m)               |
    | 29    | `robot:panda_reciever_l_gripper_finger_joint` slide joint translation value   | -Inf     | Inf      | robot:panda_reciever_l_gripper_finger_joint                      | slide      | position (m)               |
    30
    31
    32
    +3+ x,y,z end effector pos robot 1
    | 33    | `robot:panda_reciever_joint1` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_reciever_joint1                             | hinge      | angular velocity (rad/s)   |
    | 34    | `robot:panda_reciever_joint2` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_reciever_joint2                             | hinge      | angular velocity (rad/s)   |
    | 35    | `robot:panda_reciever_joint3` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_reciever_joint3                             | hinge      | angular velocity (rad/s)   |
    | 36    | `robot:panda_reciever_joint4` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_reciever_joint4                             | hinge      | angular velocity (rad/s)   |
    | 37    | `robot:panda_reciever_joint5` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_reciever_joint5                             | hinge      | angular velocity (rad/s)   |
    | 38    | `robot:panda_reciever_joint6` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_reciever_joint6                             | hinge      | angular velocity (rad/s)   |
    | 39    | `robot:panda_reciever_joint7` hinge joint angular velocity           | -Inf     | Inf      | robot:panda_reciever_joint7                             | hinge      | angle (rad)                |
    | 40    | `robot:panda_reciever_r_gripper_finger_joint` slide joint linear velocity     | -Inf     | Inf      | robot:panda_reciever_r_gripper_finger_joint                      | slide      | linear velocity (m/s)      |
    | 41    | `robot:panda_reciever_l_gripper_finger_joint` slide joint linear velocity     | -Inf     | Inf      | robot:panda_reciever_l_gripper_finger_joint                      | slide      | linear velocity (m/s)      |    
    
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
        max_episode_steps: int = 80,
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
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=obs.shape, dtype="float64"
        )
        
        self.episode_step = 0
        self.max_episode_steps = max_episode_steps

        self.prev_step_robot_qpos = np.array(18)
        self.prev_step_robot_qvel = np.array(18)


        EzPickle.__init__(
            self,
            tasks_to_complete,
            terminate_on_tasks_completed,
            remove_task_when_completed,
            object_noise_ratio,
            **kwargs,
        )

    def calculate_reward(
        self,
        desired_goal: "dict[str, np.ndarray]",
        achieved_goal: "dict[str, np.ndarray]",
        robot_obs,
    ):        
        # weight factors
        distance_penalty_factor = 0.5
        velocity_penalty_factor = 0.05
        position_penalty_factor = 0.05
        stop_penalty_factor = 0.2

        # gets the previous qpos and qvels of the robot
        giver_prev_pos = self.prev_step_robot_qpos[:9]
        reciever_prev_pos = self.prev_step_robot_qpos[9:]
        giver_prev_vel = self.prev_step_robot_qvel[:9]
        reciever_prev_vel = self.prev_step_robot_qvel[9:]

        # gets the current qpos and qvels of the robot
        giver_current_pos = robot_obs[:9]
        reciever_current_pos = robot_obs[21:30]
        giver_current_vel = robot_obs[12:21]
        reciever_current_vel = robot_obs[33:42]

        combined_reward = 0

        # distance of giver
        distance_giver = np.linalg.norm(achieved_goal["panda_giver_fetch"] - desired_goal["panda_giver_fetch"])

        # reward determination
        if distance_giver < PANDA_GIVER_FETCH_THRESH:
            self.step_task_completions.append("panda_giver_fetch")
            combined_reward += distance_penalty_factor / distance_giver
            
            # Penalize any movement after reaching the goal
            giver_velocity = np.sum(np.abs(giver_current_vel))
            combined_reward -= stop_penalty_factor * giver_velocity
        else:
            combined_reward -= distance_penalty_factor / distance_giver

        # distance of reciever
        distance_reciever = np.linalg.norm(achieved_goal["panda_reciever_wait"] - desired_goal["panda_reciever_wait"])
       
        # reward determination
        if distance_reciever < PANDA_RECIEVER_WAIT_THRESH:
            self.step_task_completions.append("panda_reciever_wait")
            combined_reward += distance_penalty_factor / distance_reciever

            # Penalize any movement after reaching the goal
            reciever_velocity = np.sum(np.abs(reciever_current_vel))
            combined_reward -= stop_penalty_factor * reciever_velocity
        else:
            combined_reward -= distance_penalty_factor / distance_reciever

        # if the fetch has been completed for the first time, big reward
        if "panda_giver_fetch" in self.step_task_completions and "panda_giver_fetch" not in self.episode_task_completions:
            self.episode_task_completions.append("panda_giver_fetch")
            combined_reward += 100
        
        # if the wait has been completed after the fetch has been done, big reward
        if "panda_reciever_wait" in self.step_task_completions and "panda_giver_fetch" in self.episode_task_completions:
            self.episode_task_completions.append("panda_reciever_wait")
            combined_reward += 1000

        # penalize velocity changes
        giver_velocity_diff = np.sum(np.abs(giver_current_vel - giver_prev_vel))
        reciever_velocity_diff = np.sum(np.abs(reciever_current_vel - reciever_prev_vel))
        velocity_penalty = velocity_penalty_factor * (giver_velocity_diff + reciever_velocity_diff)
        combined_reward -= velocity_penalty

        # penalize position changes
        reciever_position_diff = np.sum(np.abs(reciever_current_pos - reciever_prev_pos))
        giver_position_diff = np.sum(np.abs(giver_current_pos - giver_prev_pos))
        combined_reward = position_penalty_factor * (giver_position_diff + reciever_position_diff)
        combined_reward -= velocity_penalty

        return combined_reward
    
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

        observations = np.concatenate((robot_obs, obj_qpos, obj_qvel))
        self.achieved_goal = {
            task: observations[OBS_ELEMENT_INDICES[task]] for task in self.goal.keys()
        }

        return observations

    def step(self, action):
        robot_obs, _, terminated, truncated, info = self.robot_env.step(action)
        obs = self._get_obs(robot_obs)

        self.episode_step += 1

        reward = self.calculate_reward(self.goal, self.achieved_goal, robot_obs)
        self.prev_step_robot_qpos = np.concatenate((robot_obs[:9], robot_obs[21:30]))
        self.prev_step_robot_qvel = np.concatenate((robot_obs[12:21], robot_obs[33:42]))
       
        # When the task is accomplished remove from the list of tasks to be completed
        if self.remove_task_when_completed:
            for element in self.episode_task_completions:
                if element in self.tasks_to_complete:
                    self.tasks_to_complete.remove(element)

        info = {"tasks_to_complete": list(self.tasks_to_complete)}
        info["step_task_completions"] = self.step_task_completions.copy()

        for task in self.step_task_completions:
            if task not in self.episode_task_completions:
                self.episode_task_completions.append(task)
        info["episode_task_completions"] = self.episode_task_completions
        if self.terminate_on_tasks_completed:
            # terminate if there are no more tasks to complete
            terminated = len(self.episode_task_completions) == len(self.goal.keys())
            # terminate if there are no more tasks to complete
        
        if self.episode_step >= self.max_episode_steps:
            terminated = True

        self.step_task_completions = []
       
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        # super().reset(seed=seed, **kwargs)
        self.episode_step = 0
        self.episode_task_completions.clear()
        robot_obs, _ = self.robot_env.reset(seed=seed)
        self.prev_step_robot_qpos = np.concatenate((robot_obs[:9], robot_obs[21:30]))
        self.prev_step_robot_qvel = np.concatenate((robot_obs[12:21], robot_obs[33:42]))

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

