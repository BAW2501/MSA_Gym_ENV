from MultipleSequenceAlignmentEnv import MultipleSequenceAlignmentEnv

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = MultipleSequenceAlignmentEnv(['MCRIAGGRGTLLPLLAALLQA',
                                    'MSFPCKFVASFLLIFNVSSKGA',
                                    'MPGKMVVILGASNILWIMF'])
obs = env.reset()

# Instantiate the agent
model = A2C("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()