from dqn import *
from tqdm import tqdm
from source.modules.environment.grid_world import *
import matplotlib.pyplot as plt
from di_estimator import *

num_episodes = 10000
save_freq = 1000
print_freq = num_episodes // 10
training_mode = False
render = False
evaluate_di = True
total_rewards_per_episode = []
train_average_qs = []
episode_losses = []
epsilons = []
load_path = "lambda1.pth"
save_path = f"model"

env = GridWorld(obstacle_density=0.0,lbda=1, debug=False, max_steps=100)
agent = DQN(observation_size=3, action_size=4, hidden_size=32, epsilon=1, gamma=0.9, N_steps=1, learning_rate=0.0003, train_freq=1, replay_capacity=100000)
di_estimator = DIEstimator(num_actions=4,k=5)
action_seq = []

try:
    if not training_mode:
        agent.load_agent(load_path)

    for episode in tqdm(range(num_episodes), desc="Training Progress", unit="episode"):
        done = False
        truncated = False
        episode_reward = 0
        obs, _ = env.reset()
        step = 0 
        not training_mode and render and print(f"Episode: {episode}")
        not training_mode and render and env.render()
        while not done and not truncated and (step < env.max_steps):
            #print(obs)
            step += 1
            action = agent.sample_action(obs, training_mode=training_mode, action_mask=None)
            next_obs, reward, done, truncated, _ = env.step(action)
            #print(action, next_obs, reward, done)
            if training_mode:
                agent.store_transition(obs, action, reward, next_obs, done or truncated)
                results = agent.train(batch_size=64)
                if results is not None:
                        batch_loss, avg_q, = results
                        train_average_qs.append(avg_q)
                        episode_losses.append(batch_loss)
            if evaluate_di:
                action_seq.append(action)
            episode_reward += reward
            obs = next_obs

            not training_mode and render and env.render()
        total_rewards_per_episode.append(episode_reward)
        epsilons.append(agent.epsilon)
        if evaluate_di and episode_reward >= 1.0:
            di_estimator.update(actions=action_seq)

        action_seq = []

        if (episode + 1) % (num_episodes/1000) == 0:
            agent.update_epsilon(decay_rate=0.997)
            
        
        if training_mode and (episode + 1) % print_freq == 0:
            print(f"Average Rewards: {np.mean(total_rewards_per_episode[-print_freq:])}")
            print(f"Average Qs: {np.mean(train_average_qs[-print_freq:])}")
            print(f"Average Loss: {np.mean(episode_losses[-print_freq:])}")

except KeyboardInterrupt:
    print("Training Interrupted...")
training_mode and agent.save_agent(f"{episode+1}_{save_path}.pth")
        
    
# Rewards over episodes
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
window = 10
if len(total_rewards_per_episode) >= window:
    plt.plot(np.convolve(total_rewards_per_episode, 
                        np.ones(window)/window, 
                        mode="valid"), 
            label=f"Rolling Avg ({window})")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode")
plt.legend()

# Loss over training steps
plt.subplot(2, 2, 2)
plt.plot(episode_losses, label="Loss", alpha=0.7)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.legend()

# Q-values over training steps
plt.subplot(2, 2, 3)
plt.plot(train_average_qs, label="Average Q-value", color="orange")
plt.xlabel("Training Steps")
plt.ylabel("Q-value")
plt.title("Average Q-value")
plt.legend()


# Epsilon decay
plt.subplot(2, 2, 4)
plt.plot(epsilons, label="Epsilon", color="green")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay")
plt.legend()


plt.tight_layout()
plt.show()

print(f"Final Average Reward: {np.average(total_rewards_per_episode)}")
print("Entropy:", di_estimator.get_entropy())
print("Average actions:", di_estimator.get_average_actions())
    