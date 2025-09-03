import os
import sys
import cv2
import time
import pickle
import random
import numpy as np
import gymnasium 
import vizdoom.gymnasium_wrapper
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecMonitor
import pandas as pd

# Observation preprocessing wrapper
class ObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, shape, frame_skip):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.env.frame_skip = frame_skip

        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)
        if observation.shape[-1] != 3:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        return observation

# Grayscale conversion wrapper
class GrayscaleObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
        )

    def observation(self, observation):
        if observation.shape[-1] != 3:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, axis=-1)

# DRL Agent class specialized for Vizdoom
class VizdoomDRLAgent:
    def __init__(self, learning_alg, train_mode=True, seed=None, n_envs=8, frame_skip=4):
        self.environment_id = "VizdoomTakeCover-v0"
        self.learning_alg = learning_alg
        self.train_mode = train_mode
        self.seed = seed if seed else random.randint(0, 1000)
        self.policy_filename = f"{learning_alg}-{self.environment_id}-seed{self.seed}.policy.pkl"
        self.n_envs = n_envs if train_mode else 1
        self.frame_skip = frame_skip
        self.image_shape = (84, 84)
        self.training_timesteps = 10000
        self.num_test_episodes = 20
        self.l_rate = 0.00083
        self.gamma = 0.995
        self.n_steps = 512
        self.policy_rendering = True
        self.rendering_delay = 0.05
        self.log_dir = './vizdoom_logs'
        self.model = None
        self.policy = "CnnPolicy"
        self.environment = None
        self.train_time = 0
        self.test_time = 0

        self._validate_environment()
        self._create_log_directory()

    def _validate_environment(self):
        if "VizdoomTakeCover-v0" not in gymnasium.envs.registry:
            print("ERROR: VizdoomTakeCover-v0 environment not available!")
            sys.exit(1)

    def _create_log_directory(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def wrap_env(self, env):
        env = ObservationWrapper(env, shape=self.image_shape, frame_skip=self.frame_skip)
        env = GrayscaleObservationWrapper(env)
        if self.train_mode:
            env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01)
        return env

    def create_environment(self):
        self.environment = make_vec_env(
            self.environment_id,
            n_envs=self.n_envs,
            seed=self.seed,
            monitor_dir=self.log_dir,
            wrapper_class=self.wrap_env
        )
        self.environment = VecFrameStack(self.environment, n_stack=4)
        self.environment = VecTransposeImage(self.environment)

    def create_model(self):
        if self.learning_alg == "DQN":
            self.model = DQN(self.policy, self.environment,
                           seed=self.seed,
                           learning_rate=self.l_rate,
                           gamma=self.gamma,
                           buffer_size=10000,
                           batch_size=64,
                           exploration_fraction=0.9,
                           verbose=1)
        elif self.learning_alg == "A2C":
            self.model = A2C(self.policy, self.environment,
                           seed=self.seed,
                           learning_rate=self.l_rate,
                           gamma=self.gamma,
                           verbose=1)
        elif self.learning_alg == "PPO":
            self.model = PPO(self.policy, self.environment,
                           seed=self.seed,
                           learning_rate=self.l_rate,
                           gamma=self.gamma,
                           verbose=1)
        else:
            print(f"Unsupported algorithm: {self.learning_alg}")
            sys.exit(1)

    def train_or_load_model(self):
        if self.train_mode:
            start_time = time.time()
            self.model.learn(total_timesteps=self.training_timesteps)
            self.train_time = time.time() - start_time
            print(f"Saving policy to {self.policy_filename}")
            pickle.dump(self.model.policy, open(self.policy_filename, 'wb'))
        else:
            print(f"Loading policy from {self.policy_filename}")
            with open(self.policy_filename, "rb") as f:
                self.model.policy = pickle.load(f)

    def evaluate_policy(self):
        start_time = time.time()
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.model.get_env(),
            n_eval_episodes=self.num_test_episodes
        )
        self.test_time = time.time() - start_time
        print(f"Evaluation Results - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        return mean_reward, std_reward

    def render_policy(self):
        env = self.model.get_env()
        obs = env.reset()
        total_reward = 0
        
        for episode in range(1, self.num_test_episodes + 1):
            episode_reward = 0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                if self.policy_rendering:
                    env.render("human")
                    time.sleep(self.rendering_delay)
                if done:
                    break
            total_reward += episode_reward
            print(f"Episode {episode}: Reward = {episode_reward[0]:.2f}")
            obs = env.reset()
        
        print(f"\nAverage Reward: {total_reward / self.num_test_episodes:.2f}")

    def save_metrics(self, mean_reward):
        metrics = {
            'Environment': [self.environment_id],
            'Algorithm': [self.learning_alg],
            'Seed': [self.seed],
            'Training Time (s)': [self.train_time],
            'Testing Time (s)': [self.test_time],
            'Mean Reward': [mean_reward],
            'Timesteps': [self.training_timesteps]
        }
        df = pd.DataFrame(metrics)
        df.to_excel(f'{self.log_dir}/training_metrics.xlsx', index=False)

    def run(self):
        self.create_environment()
        self.create_model()
        self.train_or_load_model()
        mean_reward, _ = self.evaluate_policy()
        self.render_policy()
        self.save_metrics(mean_reward)

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python vizdoom_drl.py (train|test) (DQN|A2C|PPO) [seed]")
        sys.exit(1)
    
    train_mode = sys.argv[1].lower() == 'train'
    algorithm = sys.argv[2].upper()
    seed = int(sys.argv[3]) if len(sys.argv) == 4 else None

    if algorithm not in ["DQN", "A2C", "PPO"]:
        print("Invalid algorithm. Choose from: DQN, A2C, PPO")
        sys.exit(1)

    agent = VizdoomDRLAgent(
        learning_alg=algorithm,
        train_mode=train_mode,
        seed=seed
    )
    agent.run()