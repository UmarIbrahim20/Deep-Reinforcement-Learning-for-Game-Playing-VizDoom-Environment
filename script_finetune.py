import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
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

class VizdoomDRLAgent:
    def __init__(self, learning_alg, trial=None, train_mode=True, seed=None, 
                 n_envs=8, frame_skip=4, tuning_timesteps=100000):
        self.environment_id = "VizdoomTakeCover-v0"
        self.learning_alg = learning_alg
        self.trial = trial
        self.train_mode = train_mode
        self.seed = seed if seed else random.randint(0, 1000)
        self.policy_filename = f"{learning_alg}-{self.environment_id}-seed{self.seed}.policy.pkl"
        self.n_envs = n_envs if train_mode else 1
        self.frame_skip = frame_skip
        self.image_shape = (84, 84)
        self.training_timesteps = tuning_timesteps
        self.num_test_episodes = 20
        self.policy_rendering = True
        self.rendering_delay = 0.05
        self.log_dir = './metric_folder'
        self.model = None
        self.policy = "CnnPolicy"
        self.environment = None
        self.train_time = 0
        self.test_time = 0

        self._validate_environment()
        self._create_log_directory()
        self.create_environment()

    def _validate_environment(self):
        if "VizdoomTakeCover-v0" not in gymnasium.envs.registry:
            print("ERROR: VizdoomTakeCover-v0 environment not available!")
            sys.exit(1)

    def _create_log_directory(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def wrap_env(self, env):
        env = ObservationWrapper(env, shape=self.image_shape, frame_skip=self.frame_skip)
        env = GrayscaleObservationWrapper(env)
        if self.train_mode and self.learning_alg != "DQN":
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

    def suggest_hyperparameters(self):
        params = {
            'learning_rate': self.trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'gamma': self.trial.suggest_float('gamma', 0.9, 0.9999),
        }

        if self.learning_alg == "PPO":
            params.update({
                'n_steps': self.trial.suggest_categorical('n_steps', [128, 256, 512]),
                'batch_size': self.trial.suggest_categorical('batch_size', [64, 128, 256]),
                'gae_lambda': self.trial.suggest_float('gae_lambda', 0.8, 0.99),
                'clip_range': self.trial.suggest_float('clip_range', 0.1, 0.4),
                'ent_coef': self.trial.suggest_float('ent_coef', 0.0, 0.1),
            })
        elif self.learning_alg == "A2C":
            params.update({
                'n_steps': self.trial.suggest_categorical('n_steps', [5, 16, 32]),
                'vf_coef': self.trial.suggest_float('vf_coef', 0.25, 0.75),
                'max_grad_norm': self.trial.suggest_float('max_grad_norm', 0.3, 5.0),
            })
        elif self.learning_alg == "DQN":
            params.update({
                'batch_size': self.trial.suggest_categorical('batch_size', [32, 64, 128]),
                'buffer_size': self.trial.suggest_categorical('buffer_size', [10000, 20000]),
                'exploration_final_eps': self.trial.suggest_float('exploration_final_eps', 0.01, 0.1),
                'train_freq': self.trial.suggest_categorical('train_freq', [4, 8]),
                'learning_starts': self.trial.suggest_categorical('learning_starts', [1000, 5000]),
            })

        params['net_arch'] = self.trial.suggest_categorical('net_arch', ['small', 'medium', 'large'])
        return params

    def create_model(self):
        if self.trial:
            params = self.suggest_hyperparameters()
        else:
            params = self._get_default_params()

        policy_kwargs = self._get_network_architecture(params)
        self._build_model(params, policy_kwargs)

    def _get_default_params(self):
        base_params = {
            'learning_rate': 0.0001 if self.learning_alg == "DQN" else 0.00083,
            'gamma': 0.995,
        }
        
        if self.learning_alg == "DQN":
            base_params.update({
                'batch_size': 64,
                'buffer_size': 10000,
                'train_freq': 4,
                'learning_starts': 1000
            })
        elif self.learning_alg == "A2C":
            base_params.update({'n_steps': 5})
        elif self.learning_alg == "PPO":
            base_params.update({'n_steps': 256, 'batch_size': 128})
            
        return base_params

    def _get_network_architecture(self, params):
        arch_mapping = {
            'small': [256],
            'medium': [512, 256],
            'large': [1024, 512, 256]
        } if self.learning_alg == "DQN" else {
            'small': [64, 64],
            'medium': [128, 128],
            'large': [256, 256]
        }
        return {'net_arch': arch_mapping[params.get('net_arch', 'medium')]}

    def _build_model(self, params, policy_kwargs):
        alg_map = {
            "DQN": lambda: DQN(
                self.policy, self.environment,
                policy_kwargs=policy_kwargs,
                **{k: v for k, v in params.items() if k in [
                    'learning_rate', 'gamma', 'batch_size', 'buffer_size',
                    'exploration_final_eps', 'train_freq', 'learning_starts'
                ]}
            ),
            "A2C": lambda: A2C(
                self.policy, self.environment,
                policy_kwargs=policy_kwargs,
                **{k: v for k, v in params.items() if k in [
                    'learning_rate', 'gamma', 'n_steps', 'vf_coef', 'max_grad_norm'
                ]}
            ),
            "PPO": lambda: PPO(
                self.policy, self.environment,
                policy_kwargs=policy_kwargs,
                **{k: v for k, v in params.items() if k in [
                    'learning_rate', 'gamma', 'n_steps', 'batch_size',
                    'gae_lambda', 'clip_range', 'ent_coef'
                ]}
            )
        }
        self.model = alg_map[self.learning_alg]()

    def train(self):
        start_time = time.time()
        self.model.learn(total_timesteps=self.training_timesteps)
        self.train_time = time.time() - start_time

    def evaluate(self):
        start_time = time.time()
        mean_reward, std_reward = evaluate_policy(
            self.model, self.model.get_env(),
            n_eval_episodes=self.num_test_episodes
        )
        self.test_time = time.time() - start_time
        return mean_reward, std_reward
    
    def save_metrics(self, mean_reward, std_reward):
        metrics = {
            'Environment': [self.environment_id],
            'Algorithm': [self.learning_alg],
            'Seed': [self.seed],
            'Training Time (s)': [self.train_time],
            'Testing Time (s)': [self.test_time],
            'Mean Reward': [mean_reward],
            'Std Reward': [std_reward],
            'Timesteps': [self.training_timesteps]
        }
        df = pd.DataFrame(metrics)
        df.to_excel(f'{self.log_dir}/training_metrics.xlsx', index=False)
        print(f"Metrics saved to {self.log_dir}/training_metrics.xlsx")

    def render_policy(self):
        env = gymnasium.make(self.environment_id, render_mode="human")
        env = self.wrap_env(env)
        obs = env.reset()
        
        for episode in range(3):  # Render 3 episodes max
            total_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                env.render()
                time.sleep(self.rendering_delay)
            print(f"Rendered Episode {episode+1} Reward: {total_reward:.1f}")
            obs = env.reset()
        env.close()

def objective(trial):
    try:
        agent = VizdoomDRLAgent(
            learning_alg=algorithm,
            trial=trial,
            train_mode=True,
            tuning_timesteps=10000
        )
        agent.create_model()
        agent.train()
        mean_reward, _ = agent.evaluate()
        
        # Early pruning for DQN/A2C
        if algorithm in ["DQN", "A2C"] and mean_reward < 1.0:
            raise TrialPruned()
        if algorithm == "PPO" and mean_reward < 5.0:
            raise TrialPruned()
            
        return mean_reward
        
    except Exception as e:
        trial.set_user_attr("error", str(e))
        raise TrialPruned()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vizdoom_tuning.py (tune|train|test) (DQN|A2C|PPO) [seed]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    algorithm = sys.argv[2].upper()
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if mode == "tune":
        storage_url = f"sqlite:///vizdoom_study_{algorithm}.db"
        
        try:
            study = optuna.load_study(
                study_name=f"vizdoom_study_{algorithm}",
                storage=storage_url,
                sampler=TPESampler(n_startup_trials=10),
                pruner=MedianPruner()
            )
            print(f"Resuming study with {len(study.trials)} trials")
        except:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(n_startup_trials=10),
                pruner=MedianPruner(),
                storage=storage_url,
                study_name=f"vizdoom_study_{algorithm}"
            )
            print("Created new study")

        study.optimize(
            lambda trial: objective(trial),
            n_trials=50,
            show_progress_bar=True,
            catch=(RuntimeError, TrialPruned)
        )

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            print("No successful trials! Check hyperparameter ranges.")
            sys.exit(1)
            
        best_trial = study.best_trial
        print(f"\nBest trial (Reward: {best_trial.value:.2f})")
        for k, v in best_trial.params.items():
            print(f"- {k}: {v}")
            
        with open(f"best_params_{algorithm}.pkl", "wb") as f:
            pickle.dump(best_trial.params, f)

    elif mode in ["train", "test"]:
        agent = VizdoomDRLAgent(
            learning_alg=algorithm,
            train_mode=(mode == "train"),
            seed=seed,
            tuning_timesteps=100000 if mode == "train" else 0
        )
        agent.create_model()
        
        if mode == "train":
            agent.train()
            pickle.dump(agent.model.policy, open(agent.policy_filename, 'wb'))
            print(f"Saved policy to {agent.policy_filename}")
        
        mean_reward, std_reward = agent.evaluate()
        agent.save_metrics(mean_reward, std_reward)
        
        if agent.policy_rendering and mode == "test":
            agent.render_policy()

    else:
        print("Invalid mode. Choose: tune, train, test")
        sys.exit(1)