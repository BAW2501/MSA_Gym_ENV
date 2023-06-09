import numpy as np
from MultipleSequenceAlignmentEnv import MultipleSequenceAlignmentEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = MultipleSequenceAlignmentEnv()
obs = env.reset()

# Instantiate the agent
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
#Train the agent and display a progress bar
model.learn(total_timesteps=int(1e4), progress_bar=True)

vec_env = model.get_env()
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
print(mean_reward)

model.save(f"Models/ppo_recurrent_msa{mean_reward:.2f}")
# del model # remove to demonstrate saving and loading

# model = RecurrentPPO.load("ppo_recurrent")

obs = vec_env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    episode_starts = dones
    vec_env.render("human")