from typing import Any, Optional
import gymnasium as gym

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from franka_env import FrankaRobot


OBS_ELEMENT_INDICES = {
    "panda_giver_grasp": np.array([9, 10, 11]),
    "panda_reciever_to_giver": np.array([30, 31, 32]),
    "object_move_lift":  np.array([44]),

    "object_move_handover":  np.array([42, 43, 44]),

    "panda_reciever_grasp": np.array([30, 31, 32]),
    "object_move_place": np.array([42, 43, 44]),
    "panda_giver_retreat": np.array([9, 10, 11]),
}

OBS_ELEMENT_GOALS = {
    "panda_giver_grasp": np.array([0,0,0]),
    "panda_reciever_to_giver": np.array([0.65, 0.15, 1.87]),
    "object_move_lift":  np.array([0.95]),

    "object_move_handover":  np.array([0, 0, 1.3]),

    "panda_reciever_grasp": np.array([0, 0, 1.3]),
    "object_move_place": np.array([0.5, -0.3, 0.785]),
    "panda_giver_retreat": np.array([-0.65, -0.15, 1.87]),
} 

# TODO: order of 0.5 20, then 0 20, then 0.25 10

PENALTY_FACTOR = 0

OBJECT_PLACE_THRESHOLD = 0.1
OBJECT_HANDOVER_REGION_THRESHOLD = 0.15

MAX_OBJECT_HEIGHT = 1.8
MIN_OBJECT_HEIGHT = 0.7

MIN_END_EFFECTOR_HEIGHT = 0.8

class HandoverEnv(gym.Env, EzPickle):
    """
    ## Description

    This environment has been created in order to learn a safe handover between two robot arms. It has been heavily inspired by the  
    Frank Kitchen Environment. Their code (https://github.com/Farama-Foundation/Gymnasium-Robotics) was used as a base in order to implement this environment.

    ## Goal

    The goal has a multitask configuration. The multiple tasks to be completed in an episode can be set by passing a list of tasks to the argument`tasks_to_complete`. For example, to open
    the microwave door and move the object create the environment as follows:

    The following is a table with all the possible tasks and their respective joint goal values:

    | Task             | Description                                                    | Joint Type | Goal                                     |
    | ---------------- | -------------------------------------------------------------- | ---------- | ---------------------------------------- |
    | "object"         | Move the object from one table to the next                     | free       | [ 0.75, -0.4, 0.775, 1, 0., 0., 0.] |

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
        the pose and velocities of the object (object to be handed over). An additional uniform noise of range `[-1,1]` is added to the observations. The noise is also scaled by a factor
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
    
    | 42    | object's x coordinate                                 | -Inf     | Inf      | object                                   | free       | position (m)               |
    | 43    | object's y coordinate                                 | -Inf     | Inf      | object                                   | free       | position (m)               |
    | 44    | object's z coordinate                                 | -Inf     | Inf      | object                                   | free       | position (m)               |
    | 45    | object's x quaternion rotation                        | -Inf     | Inf      | object                                   | free       | -                          |
    | 46    | object's y quaternion rotation                        | -Inf     | Inf      | object                                   | free       | -                          |
    | 47    | object's z quaternion rotation                        | -Inf     | Inf      | object                                   | free       | -                          |
    | 48    | object's w quaternion rotation                        | -Inf     | Inf      | object                                   | free       | -                          |
    | 49    | object's x linear velocity                            | -Inf     | Inf      | object                                   | free       | linear velocity (m/s)      |
    | 50    | object's y linear velocity                            | -Inf     | Inf      | object                                   | free       | linear velocity (m/s)      |
    | 51    | object's z linear velocity                            | -Inf     | Inf      | object                                   | free       | linear velocity (m/s)      |
    | 52    | object's x axis angular rotation                      | -Inf     | Inf      | object                                   | free       | angular velocity(rad/s)    |
    | 53    | object's y axis angular rotation                      | -Inf     | Inf      | object                                   | free       | angular velocity(rad/s)    |
    | 54    | object's z axis angular rotation                      | -Inf     | Inf      | object                                   | free       | angular velocity(rad/s)    |

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

    The simulation starts with all of the joint position actuators of the Franka robot set to zero. The object will be placed at the edge of the table.

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
        "render_fps": 50,
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
        
        # self.episode_task_completions = (
        #     [] # pickup
        # ) # Tasks completed that have been completed in the current episode
        self.episode_task_completions = (
            ["object_move_lift", "panda_giver_grasp"] # handover
        )
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
        self.object_max_height = 0.76
        self.episode_violations = 0
        self.robot_to_robot_violations = 0
        self.robot_to_table_violations = 0
        self.robot_to_object_violations = 0
        self.object_dropped_violations = 0

        EzPickle.__init__(
            self,
            tasks_to_complete,
            terminate_on_tasks_completed,
            remove_task_when_completed,
            object_noise_ratio,
            **kwargs,
        )

    def giver_robot_grasp_object_reward(
        self,
        achieved_goal,
        desired_goal,
        good_collisons,
        combined_reward,
    ):

        # reward positive height of object changes
        if (achieved_goal["object_move_lift"][0] > self.object_max_height):
            self.object_max_height = achieved_goal["object_move_lift"][0]

        # calculate distance between giver robot end effector and objects current position
        distance_giver_end_effector_to_object = np.linalg.norm(achieved_goal["panda_giver_grasp"] - achieved_goal["object_move_place"])
        # provide relative reward based on the distance between them (closer, bigger reward and furhter, smaller reward)
        combined_reward += 0.25 * (1-np.tanh(distance_giver_end_effector_to_object))
        
        # reward the giver robot touching the object with the inside of its fingers (within its grip)
        if good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 and good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
            combined_reward += 250
            # add the task for the robot fetching the object to completed tasks once the object has been grasped by the giver robot
            self.episode_task_completions.append("panda_giver_grasp")  
        elif good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 or good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
            combined_reward += 3
        # reward the giver robot touching the object with its fingers
        if good_collisons.count("giver_robot_finger_object_col") == 2:
            combined_reward += 2
        elif good_collisons.count("giver_robot_finger_object_col") == 1:
            combined_reward += 1
        
        # get the distance between the end effector and the goal positon
        distance_reciever_from_start_pos = np.linalg.norm(achieved_goal["panda_reciever_to_giver"] - desired_goal["panda_reciever_to_giver"])
        # provide relative reward based on the distance
        combined_reward += 0.25 * (1-np.tanh(distance_reciever_from_start_pos))

        return combined_reward

    def giver_robot_lift_object(
        self,
        achieved_goal,
        desired_goal,
        good_collisons,
        combined_reward,
    ):
        
        # reward the giver robot touching the object with the inside of its fingers (within its grip)
        if good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 and good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
            combined_reward += 5
        elif good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 or good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
            combined_reward += 3
        # reward the giver robot touching the object with its fingers
        if good_collisons.count("giver_robot_finger_object_col") == 2:
            combined_reward += 2
        elif good_collisons.count("giver_robot_finger_object_col") == 1:
            combined_reward += 1
        
        # reward positive height of object changes
        if (achieved_goal["object_move_lift"][0] > self.object_max_height + 0.001):
            self.object_max_height = achieved_goal["object_move_lift"][0]
            combined_reward += 20

        # if the objects current position is within the handover region
        if  achieved_goal["object_move_lift"] >=  desired_goal["object_move_lift"]:
            # add the task for the objecting being in the handover position completed tasks
            self.episode_task_completions.append("object_move_lift")
            combined_reward += 500
        
        # get the distance between the end effector and the goal positon
        distance_reciever_from_start_pos = np.linalg.norm(achieved_goal["panda_reciever_to_giver"] - desired_goal["panda_reciever_to_giver"])
        # provide relative reward based on the distance
        combined_reward += 0.25 * (1-np.tanh(distance_reciever_from_start_pos))

        return combined_reward

    def giver_robot_move_object_and_reciever_end_effector_together_reward(
        self,
        achieved_goal,
        desired_goal,
        good_collisons,
        combined_reward,
    ):
        
        # reward the giver robot touching the object with the inside of its fingers (within its grip)
        if good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 and good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
            combined_reward += 5
        elif good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 or good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
            combined_reward += 3
        # reward the giver robot touching the object with its fingers
        if good_collisons.count("giver_robot_finger_object_col") == 2:
            combined_reward += 2
        elif good_collisons.count("giver_robot_finger_object_col") == 1:
            combined_reward += 1
            
        # calculate distance between object and desired handover
        distance_object_to_handover_region = np.linalg.norm(achieved_goal["object_move_handover"] - desired_goal["object_move_handover"])
        # provide relative reward based on the distance
        combined_reward += 5 * (1-np.tanh(distance_object_to_handover_region))

        # calculate distance between end effectors of reciever and object poositioin
        distance_reciever_end_effector_to_handover_region = np.linalg.norm(achieved_goal["panda_reciever_grasp"] - desired_goal["object_move_handover"])
        # provide relative reward based on the distance
        combined_reward += 5 * (1-np.tanh(distance_reciever_end_effector_to_handover_region))

        robots_in_handover_region = True
        if distance_object_to_handover_region < OBJECT_HANDOVER_REGION_THRESHOLD:
            combined_reward += 10
        else:
            combined_reward -= 5 * distance_object_to_handover_region
            robots_in_handover_region = False
        
        if distance_reciever_end_effector_to_handover_region < OBJECT_HANDOVER_REGION_THRESHOLD:
            combined_reward += 10
        else:
            combined_reward -= 5 * distance_reciever_end_effector_to_handover_region
            robots_in_handover_region = False

        # if the  and reciver end effector current position is within the handover region 
        if robots_in_handover_region:
            # add the task for the objecting being in the handover position completed tasks
            self.episode_task_completions.append("panda_reciever_to_giver")
            combined_reward += 1000

        return combined_reward

    def robot_object_handover_reward(
        self,
        achieved_goal,
        desired_goal,
        good_collisons,
        combined_reward,
    ):
        
        # reward the giver robot touching the object with the inside of its fingers (within its grip)
        if good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 and good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
            combined_reward += 5
        elif good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 or good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
            combined_reward += 3
        # reward the giver robot touching the object with its fingers
        if good_collisons.count("giver_robot_finger_object_col") == 2:
            combined_reward += 2
        elif good_collisons.count("giver_robot_finger_object_col") == 1:
            combined_reward += 1
        
        # calculate distance between end effectors of both robots
        distance_reciever_end_effector_to_object = np.linalg.norm(achieved_goal["panda_reciever_grasp"] - achieved_goal["object_move_handover"])
        # provide relative reward based on the distance
        combined_reward += 20 * (1-np.tanh(distance_reciever_end_effector_to_object)) 

        # calculate distance between end effectors of both robots
        distance_giver_to_reciever_end_effectors = np.linalg.norm(achieved_goal["panda_reciever_grasp"] - achieved_goal["panda_giver_grasp"])
        # provide relative reward based on the distance
        combined_reward += 10 * (1-np.tanh(distance_giver_to_reciever_end_effectors))

        distance_object_to_handover_region = np.linalg.norm(achieved_goal["object_move_handover"] - desired_goal["object_move_handover"])
        # penalize for robots moving away from eachother
        if distance_object_to_handover_region > OBJECT_HANDOVER_REGION_THRESHOLD:
            combined_reward -= 7.5 * (1-np.tanh(distance_object_to_handover_region))

        # reward the reciever robot touching the object with the inside of its fingers (within its grip)
        if good_collisons.count("inside_reciever_robot_rightfinger_object_col") == 1 and good_collisons.count("inside_reciever_robot_leftfinger_object_col") == 1:
            combined_reward += 2000
            # add the task for the robot fetching the object to completed tasks once the object has been grasped by the reciever robot
            self.episode_task_completions.append("panda_reciever_grasp") 
        elif good_collisons.count("inside_reciever_robot_rightfinger_object_col") == 1 or good_collisons.count("inside_reciever_robot_leftfinger_object_col") == 1:
            combined_reward += 45
        # reward the reciever robot touching the object with its fingers
        if good_collisons.count("reciever_robot_finger_object_col") == 2:
            combined_reward += 30
        elif good_collisons.count("reciever_robot_finger_object_col") == 1:
            combined_reward += 15
        
        return combined_reward

    def giver_robot_retreat_reciever_robot_place_object_reward(
        self,
        achieved_goal,
        desired_goal,
        good_collisons,
        combined_reward,
    ):
        
        # penalize the giver robot touching the object with the inside of its fingers (within its grip)
        if good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 and good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
            combined_reward -= 50
        elif good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 or good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
            combined_reward -= 40
        # penalize the giver robot touching the object with its fingers
        if good_collisons.count("giver_robot_finger_object_col") == 2:
            combined_reward -= 30
        elif good_collisons.count("giver_robot_finger_object_col") == 1:
            combined_reward -= 20

        # reward the reciever robot touching the object with the inside of its fingers (within its grip)
        if good_collisons.count("inside_reciever_robot_rightfinger_object_col") == 1 and good_collisons.count("inside_reciever_robot_leftfinger_object_col") == 1:
            combined_reward += 80
        elif good_collisons.count("inside_reciever_robot_rightfinger_object_col") == 1 or good_collisons.count("inside_reciever_robot_leftfinger_object_col") == 1:
            combined_reward += 60
        # reward the reciever robot touching the object with its fingers
        if good_collisons.count("reciever_robot_finger_object_col") == 2:
            combined_reward += 40
        elif good_collisons.count("reciever_robot_finger_object_col") == 1:
            combined_reward += 20

        # get the distance between the object and the goal positon (place)
        distance_place_object = np.linalg.norm(achieved_goal["object_move_place"] - desired_goal["object_move_place"])
        # provide relative reward based on the distance
        combined_reward += 50 * (1-np.tanh(distance_place_object))

        # calculate distance between the giver current pos and desired pos (retreat)
        distance_giver_to_start_pos = np.linalg.norm(achieved_goal["panda_giver_retreat"] - desired_goal["panda_giver_retreat"])
        # provide relative reward based on the distance
        combined_reward += 30 * (1-np.tanh(distance_giver_to_start_pos)) 
        
        # if the object is in the goal position  (place)
        if distance_place_object <= OBJECT_PLACE_THRESHOLD:
            # finish the episode
            
            if "panda_giver_grasp" not in self.episode_task_completions:
                self.episode_task_completions.append("panda_giver_grasp")
            if "panda_giver_retreat" not in self.episode_task_completions:
                self.episode_task_completions.append("panda_giver_retreat")

            if "object_move_handover" not in self.episode_task_completions:
                self.episode_task_completions.append("object_move_handover")
            if "object_move_lift" not in self.episode_task_completions:
                self.episode_task_completions.append("object_move_lift")
            if "object_move_place" not in self.episode_task_completions:
                self.episode_task_completions.append("object_move_place")
            
            if "panda_reciever_to_giver" not in self.episode_task_completions:
                self.episode_task_completions.append("panda_reciever_to_giver")
            if "panda_reciever_grasp" not in self.episode_task_completions:
                self.episode_task_completions.append("panda_reciever_grasp")
                
            # provide a very positive reward
            combined_reward += 10000

        return combined_reward

    def calculate_reward(
        self,
        desired_goal: "dict[str, np.ndarray]",
        achieved_goal: "dict[str, np.ndarray]",
        robot_obs,
        collsions,
    ):      
        
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

        # gets the collisons from franka
        good_collisons = collsions[0]
        bad_collisons = collsions[1]  

        combined_reward = 0

        # penalty for the object touching the giver table
        combined_reward -= 0.25 * bad_collisons.count("object_on_giver_table")
            
        # penalty for giver robot hitting table
        combined_reward -= bad_collisons.count("giver_robot_table_collision")
        combined_reward -= 0.35 * bad_collisons.count("giver_robot_finger_table_collision")

        # penalty for reciever robot hitting table
        combined_reward -= bad_collisons.count("reciever_robot_table_collision")
        combined_reward -= 0.35 * bad_collisons.count("reciever_robot_finger_table_collision")

        # penalty for giver robot not using fingers in pickup task
        combined_reward -= 1.5 * bad_collisons.count("reciever_robot_hand_object_col")
        combined_reward -= 1.5 * bad_collisons.count("reciever_robot_link_object_col")

        # penalty for reciever robot not using fingers in handover task
        combined_reward -= 1.5 * bad_collisons.count("giver_robot_hand_object_col")
        combined_reward -= 1.5 * bad_collisons.count("giver_robot_link_object_col")

        # penalty for robots colliding
        combined_reward -= 2 * bad_collisons.count("robot_collision")

        # penalize changes in velocity to help with smoother movements
        giver_velocity_diff = np.sum(np.abs(giver_current_vel - giver_prev_vel))
        combined_reward -= 0.005 * giver_velocity_diff 

        # penalize changes in position to help with smoother movements
        giver_position_diff = np.sum(np.abs(giver_current_pos - giver_prev_pos))
        combined_reward -= 0.005 * giver_position_diff

        reciever_wait_quantifier = 0.005
        if "object_move_lift" not in self.episode_task_completions: 
            reciever_wait_quantifier = 0.05
            
        # penalize changes in velocity to help with smoother movements
        reviever_velocity_diff = np.sum(np.abs(reciever_current_vel - reciever_prev_vel))
        combined_reward -= reciever_wait_quantifier * reviever_velocity_diff 

        # penalize changes in position to help with smoother movements
        reviever_position_diff = np.sum(np.abs(reciever_current_pos - reciever_prev_pos))
        combined_reward -= reciever_wait_quantifier * reviever_position_diff

        combined_reward = PENALTY_FACTOR * combined_reward

        # fetch and grasp  reward
        if "panda_giver_grasp" not in self.episode_task_completions:
            combined_reward += self.giver_robot_grasp_object_reward(achieved_goal, desired_goal, good_collisons, combined_reward)
        
         # giver robot move object lift reward
        elif "object_move_lift" not in self.episode_task_completions:
            combined_reward += self.giver_robot_lift_object(achieved_goal, desired_goal, good_collisons, combined_reward)

        # giver robot move object to handover region reward
        elif "panda_reciever_to_giver" not in self.episode_task_completions:
            combined_reward += self.giver_robot_move_object_and_reciever_end_effector_together_reward(achieved_goal, desired_goal, good_collisons, combined_reward)

        # move robot reciever and giver robot wai to handover region reward
        elif "panda_reciever_grasp" not in self.episode_task_completions:
            combined_reward += self.robot_object_handover_reward(achieved_goal, desired_goal, good_collisons, combined_reward)

        # giver robot retreat to start pos and reciever robot place object reward
        else:
            combined_reward += self.giver_robot_retreat_reciever_robot_place_object_reward(achieved_goal, desired_goal, good_collisons, combined_reward)                    
       
        # calculate distance between reciever robot end effector and objects current position
        distance_reciever_end_effector_to_object = np.linalg.norm(achieved_goal["panda_reciever_grasp"] - achieved_goal["object_move_place"])

        end_episode = False
        # if the object is too high/low (fallen of table)
        if (achieved_goal["object_move_lift"] < MIN_OBJECT_HEIGHT or achieved_goal["object_move_lift"] > MAX_OBJECT_HEIGHT):
            end_episode = True
            combined_reward -= PENALTY_FACTOR * 100

        # if the handover is complete and has been dropped
        elif ("panda_reciever_grasp" in self.episode_task_completions and distance_reciever_end_effector_to_object > 0.5):
            end_episode = True
            combined_reward -= PENALTY_FACTOR * 100

        # if the handover is complete and has been dropped on givers tables
        elif ("object_move_lift" in self.episode_task_completions and (bad_collisons.count("object_on_giver_table") > 0 or bad_collisons.count("object_on_reciever_table") > 0)):
            end_episode = True
            combined_reward -= PENALTY_FACTOR * 100

        if (end_episode):

            if "panda_giver_grasp" not in self.episode_task_completions:
                self.episode_task_completions.append("panda_giver_grasp")
            if "panda_giver_retreat" not in self.episode_task_completions:
                self.episode_task_completions.append("panda_giver_retreat")

            if "object_move_handover" not in self.episode_task_completions:
                self.episode_task_completions.append("object_move_handover")
            if "object_move_lift" not in self.episode_task_completions:
                self.episode_task_completions.append("object_move_lift")
            if "object_move_place" not in self.episode_task_completions:
                self.episode_task_completions.append("object_move_place")
            
            if "panda_reciever_to_giver" not in self.episode_task_completions:
                self.episode_task_completions.append("panda_reciever_to_giver")
            if "panda_reciever_grasp" not in self.episode_task_completions:
                self.episode_task_completions.append("panda_reciever_grasp")

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

    def record_safety_violation(self,collsions):

        bad_collisons = collsions[1]
    
        # Object being dropped
        if ("object_move_lift" in self.episode_task_completions and (bad_collisons.count("object_on_giver_table") > 0 or bad_collisons.count("object_on_reciever_table") > 0)):
            self.episode_violations += 1
            self.object_dropped_violations += 1

        # Giver robot touching table
        if bad_collisons.count("giver_robot_table_collision") > 0:
            self.episode_violations += 1
            self.robot_to_table_violations += 1
        if bad_collisons.count("giver_robot_finger_table_collision") > 0:
            self.episode_violations += 1
            self.robot_to_table_violations += 1

        # Receiver robot touching table
        if bad_collisons.count("reciever_robot_table_collision") > 0:
            self.episode_violations += 1
            self.robot_to_table_violations += 1
        if bad_collisons.count("reciever_robot_finger_table_collision") > 0:
            self.episode_violations += 1
            self.robot_to_table_violations += 1

        # Giver robot not using fingers during handover
        if bad_collisons.count("giver_robot_hand_object_col") > 0:
            self.episode_violations += 1
            self.robot_to_object_violations += 1
        if bad_collisons.count("giver_robot_link_object_col") > 0:
            self.episode_violations += 1
            self.robot_to_object_violations += 1

        # Receiver robot not using fingers during handover
        if bad_collisons.count("reciever_robot_hand_object_col") > 0:
            self.episode_violations += 1
            self.robot_to_object_violations += 1
        if bad_collisons.count("reciever_robot_link_object_col") > 0:
            self.episode_violations += 1
            self.robot_to_object_violations += 1

        # Robots colliding
        if bad_collisons.count("robot_collision") > 0:
            self.episode_violations += 1
            self.robot_to_robot_violations += 1

    def step(self, action):
        
        robot_obs, _, terminated, truncated, info = self.robot_env.step(action)
        obs = self._get_obs(robot_obs)   
        collsions = self.robot_env._collsion_detection()

        self.episode_step += 1

        self.record_safety_violation(collsions)

        reward = self.calculate_reward(self.goal, self.achieved_goal, robot_obs, collsions)

        self.prev_step_robot_qpos = np.concatenate((robot_obs[:9], robot_obs[21:30]))
        self.prev_step_robot_qvel = np.concatenate((robot_obs[12:21], robot_obs[33:42]))
       
        # When the task is accomplished remove from the list of tasks to be completed
        if self.remove_task_when_completed:
            for element in self.episode_task_completions:
                if element in self.tasks_to_complete:
                    self.tasks_to_complete.remove(element)

        info = {"tasks_to_complete": list(self.tasks_to_complete)}
        info["step_task_completions"] = self.step_task_completions.copy()
        info["episode_violations"] = self.episode_violations
        info["robot_to_robot_violations"] = self.robot_to_robot_violations
        info["robot_to_table_violations"] = self.robot_to_table_violations
        info["robot_to_object_violations"] = self.robot_to_object_violations
        info["object_dropped_violations"] = self.object_dropped_violations

        # To detrmine success rate of tasks
        info["object_handed_over"] = False
        if "panda_reciever_grasp" in self.episode_task_completions:
            
            distance_giver_end_effector_to_object = np.linalg.norm(self.achieved_goal["panda_giver_grasp"] - self.achieved_goal["object_move_handover"])

            # penalize the giver robot touching the object with the inside of its fingers (within its grip)
            if distance_giver_end_effector_to_object > 0.082 and collsions[0].count("inside_giver_robot_rightfinger_object_col") == 0 and collsions[0].count("inside_giver_robot_leftfinger_object_col") == 0 and collsions[0].count("inside_reciever_robot_rightfinger_object_col") == 1 and collsions[0].count("inside_reciever_robot_leftfinger_object_col") == 1:
                info["object_handed_over"] = True           


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
        # pickup
        # self.episode_task_completions.clear()
        # handover
        self.episode_task_completions = (
            ["object_move_lift", "panda_giver_grasp"]
        )
        robot_obs, _ = self.robot_env.reset(seed=seed)

        self.prev_step_robot_qpos = np.concatenate((robot_obs[:9], robot_obs[21:30]))
        self.prev_step_robot_qvel = np.concatenate((robot_obs[12:21], robot_obs[33:42]))
        self.object_max_height = 0.76

        obs = self._get_obs(robot_obs)

        self.tasks_to_complete = set(self.goal.keys())
        info = {
            "tasks_to_complete": list(self.tasks_to_complete),
            "episode_task_completions": [],
            "step_task_completions": [],
            "episode_violations": self.episode_violations,
            "robot_to_robot_violations": self.robot_to_robot_violations,
            "robot_to_table_violations": self.robot_to_table_violations,
            "robot_to_object_violations": self.robot_to_object_violations,
            "object_dropped_violations": self.object_dropped_violations,
            "object_handed_over" : False,
        }

        self.episode_violations = 0
        self.robot_to_robot_violations = 0
        self.robot_to_table_violations = 0
        self.robot_to_object_violations = 0
        self.object_dropped_violations = 0

        return obs, info

    def render(self):
        return self.robot_env.render()

    def close(self):
        self.robot_env.close()