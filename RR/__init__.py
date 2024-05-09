from gymnasium.envs.registration import register

register(
     id="RR/GridWorld-v0",
     entry_point="RR.envs:handover",
     max_episode_steps=300,
)