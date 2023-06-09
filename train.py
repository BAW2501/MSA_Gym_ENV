from MultipleSequenceAlignmentEnv import MultipleSequenceAlignmentEnv

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


# Create environment
env = MultipleSequenceAlignmentEnv()
print('env created')
# It will check your custom environment and output additional warnings if needed
check_env(env)
print('env checked')
obs = env.reset()

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)
#Train the agent and display a progress bar
model.learn(total_timesteps=int(1e4), progress_bar=True)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
model.save(f"Models/ppo_msa{mean_reward:.2f}")

# # del model # remove to demonstrate saving and loading
#model = PPO.load(f"Models/ppo_msa{mean_reward:.2f}")
# model = PPO.load("Models/ppo_msa52.00")
obs,_ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done,_, info = env.step(action)
    env.render()
    print(action, reward,'\n')

