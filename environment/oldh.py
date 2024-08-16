#  def handover_reward(
#         self,
#         achieved_goal,
#         desired_goal,
#         good_collisons,
#         combined_reward,
#         reciever_current_vel,
#         giver_position_diff,
#     ):
        
#         # if the reciever hasnt grasped the object
#         if self.reciever_grasped == False:
#             # reward the reciver robot touching the object with its fingers
#             if good_collisons.count("giver_robot_finger_object_col") == 1:
#                 combined_reward += 1
#             if good_collisons.count("giver_robot_finger_object_col") == 2:
#                 combined_reward += 2
#             # reward the reciver robot touching the object with its fingers (inside its grip)
#             if good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 and good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
#                 combined_reward += 5
#             elif good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 or good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
#                 combined_reward += 3

#             min_OBJECT_HEIGHT_P2 = 1
            
        
#         # if the revieverr has grasped the object
#         else:
#             # penalize the giver robot touching the object with its fingers
#             if good_collisons.count("giver_robot_finger_object_col") == 1:
#                 combined_reward -= 10
#             if good_collisons.count("giver_robot_finger_object_col") == 2:
#                 combined_reward -= 20
#             # penalize the giver robot touching the object with its fingers (inside its grip)
#             if good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 and good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
#                 combined_reward -= 50
#             elif good_collisons.count("inside_giver_robot_rightfinger_object_col") == 1 or good_collisons.count("inside_giver_robot_leftfinger_object_col") == 1:
#                 combined_reward -= 30

#             # calculate distance between the giver current pos and desired pos (retreat)
#             distance_giver_retreat = np.linalg.norm(achieved_goal["panda_giver_retreat"] - desired_goal["panda_giver_retreat"])
#             # provide relative reward based on the distance
#             combined_reward += 30 * (1-np.tanh(distance_giver_retreat)) 

#             min_OBJECT_HEIGHT_P2 = 0.7


#         # get the distance between the object position and the goal positon
#         distance_object = np.linalg.norm(achieved_goal["object_move_p2"] - desired_goal["object_move_p2"])      

#         # if the end effector and object has made it to the goal
#         if distance_object <= object_move_p2_THRESH:
            
#             # if the end effector enters the goal postion, the reciever is done waiting
#             if "panda_reciever_wait" not in self.episode_task_completions:
#                 # allow the reciebver robot to move
#                 self.episode_task_completions.append("panda_reciever_wait")
#                 combined_reward += 200

#         # if the end effector and object has made it to the goal 
#         if "panda_reciever_wait" in self.episode_task_completions:
            
#             # provide constant reward for being in handover zone
#             if "panda_reciever_fetch" not in self.episode_task_completions:
                
#                 # penalize for the object leaving the handover zone
#                 if distance_object > object_move_p2_THRESH:
#                     combined_reward -= 5*(1-np.tanh(distance_object))
#                     combined_reward -= 5*giver_position_diff
                
#                 # provide constant reward for being in handover zone
#                 else:
#                     combined_reward += 15
                    
#                 # calculate distance between current pos and desired pos
#                 distance_reciever = np.linalg.norm(achieved_goal["panda_reciever_fetch"] - desired_goal["panda_reciever_fetch"])
#                 # provide relative reward based on the distance
#                 combined_reward += 20 * (1-np.tanh(distance_reciever)) 

#                 # if the end effector enters the goal postion
#                 if distance_reciever <= PANDA_RECIEVER_FETCH_THRESH:
#                     # allow the reciebver robot to move
#                     self.episode_task_completions.append("panda_reciever_fetch") 
#                     combined_reward += 400

#         else:

#             # provide relative reward based on the distance
#             combined_reward += 5*(1-np.tanh(distance_object))

#             # get the distance between the end effector and the goal positon
#             distance_reciever = np.linalg.norm(achieved_goal["panda_reciever_wait"] - desired_goal["panda_reciever_wait"])

#             # provide relative reward based on the distance
#             combined_reward += 0.125 * (1-np.tanh(distance_reciever))

#             # punish velocity from the waiter
#             combined_reward -= 0.05 * np.sum(np.abs(reciever_current_vel))
        
#         # if the end effector hasnt been put in the goal position 
#         if  "panda_reciever_fetch" in self.episode_task_completions:
            
#             # calculate distance between current recievr an object postions and try miminize the distance
#             distance_reciever_object = np.linalg.norm(achieved_goal["panda_reciever_fetch"] - achieved_goal["object_move_p2"])
#             # provide relative reward based on the distance
#             combined_reward += 20 * (1-np.tanh(distance_reciever_object)) 

#             # calculate distance between current recievr and giver postions and try miminize the distance
#             distance_giver_to_reciever = np.linalg.norm(achieved_goal["panda_giver_retreat"] - achieved_goal["panda_reciever_fetch"])
#             # provide relative reward based on the distance
#             combined_reward += 10 * (1-np.tanh(distance_giver_to_reciever)) 

#             # reward the robot touching the object with its fingers
#             if good_collisons.count("reciever_robot_finger_object_col") == 1:
#                 combined_reward += 8
#             if good_collisons.count("reciever_robot_finger_object_col") == 2:
#                 combined_reward += 16
#             # reward the robot touching the object with its fingers (inside its grip)
#             if good_collisons.count("inside_reciever_robot_rightfinger_object_col") == 1 and good_collisons.count("inside_reciever_robot_leftfinger_object_col") == 1:
#                 combined_reward += 40
#                 if self.reciever_grasped == False:
#                     combined_reward += 600
#                     min_OBJECT_HEIGHT_P2 = 0.7
#                     self.reciever_grasped = True                 
#             elif good_collisons.count("inside_reciever_robot_rightfinger_object_col") == 1 or good_collisons.count("inside_reciever_robot_leftfinger_object_col") == 1:
#                 combined_reward += 24

#             # if the object has been grasped
#             if self.reciever_grasped:

#                 # get the distance between the object and the goal positon (place)
#                 distance_place_object = np.linalg.norm(achieved_goal["panda_reciever_place"] - desired_goal["panda_reciever_place"])
#                 # provide relative reward based on the distance
#                 combined_reward += 50 * (1-np.tanh(distance_place_object))
                
#                 # if the object is in the goal position  (place)
#                 if distance_place_object <= PANDA_RECIEVER_PLACE_THRESH:
#                     # finish the episode
#                     if "object_move_p2" not in self.episode_task_completions:
#                         self.episode_task_completions.append("object_move_p2")
#                     if "panda_reciever_fetch" not in self.episode_task_completions:
#                         self.episode_task_completions.append("panda_reciever_fetch")
#                     if "panda_reciever_place" not in self.episode_task_completions:
#                         self.episode_task_completions.append("panda_reciever_place")
#                     if "panda_giver_retreat" not in self.episode_task_completions:
#                         self.episode_task_completions.append("panda_giver_retreat")
#                     if "panda_reciever_wait" not in self.episode_task_completions:
#                         self.episode_task_completions.append("panda_reciever_wait")
#                     # provide a reward
#                     combined_reward +=  10000
            
#             else:

#                 # penalize for the object leaving the handover zone
#                 if distance_object > object_move_p2_THRESH:
#                     combined_reward -= 5*(1-np.tanh(distance_object))
#                     combined_reward -= 5*giver_position_diff


#         # if the object is too high/low 
#         if achieved_goal["object_move_p2"][2] < min_OBJECT_HEIGHT_P2 or achieved_goal["object_move_p2"][2] > MAX_OBJECT_HEIGHT:
#             # finish the episode
#             if "object_move_p2" not in self.episode_task_completions:
#                 self.episode_task_completions.append("object_move_p2")
#             if "panda_reciever_fetch" not in self.episode_task_completions:
#                 self.episode_task_completions.append("panda_reciever_fetch")
#             if "panda_reciever_place" not in self.episode_task_completions:
#                 self.episode_task_completions.append("panda_reciever_place")
#             if "panda_giver_retreat" not in self.episode_task_completions:
#                 self.episode_task_completions.append("panda_giver_retreat")
#             if "panda_reciever_wait" not in self.episode_task_completions:
#                 self.episode_task_completions.append("panda_reciever_wait")
#             # provide a very negative reward (cancle out the completed reward)
#             combined_reward -= 1000 

#         return combined_reward