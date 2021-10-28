import gym
from actor_critic import Actor
import numpy as np

if __name__ == "__main__":
    agent = Actor(alpha=0.001, beta=0.00001)
