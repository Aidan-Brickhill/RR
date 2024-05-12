from gymnasium.envs.registration import register

register(
     id="RR/handover-v0",
     entry_point="RR.envs.handover:handover",
     max_episode_steps=300,
)
