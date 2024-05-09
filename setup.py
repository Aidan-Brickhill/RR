from gymnasium.envs.registration import register

register(
     id="RR/GridWorld-v0",
     entry_point="RR.envs:handover",
     max_episode_steps=300,
)

from setuptools import setup

setup(
    name="handover",
    version="0.0.1",
    install_requires=[""],
)