import chess
import numpy as np
from chess_environment import ChessEnvironment
from rl_agent import RLAgent
from utils import save_q_values, load_q_values, plot_rewards

# Parameters
episodes = 100000
initial_exploration_rate = 1.0
exploration_decay = 0.00002
min_exploration_rate = 0.01
learning_rate = 0.8
discount_factor = 0.99
save_file = 'q_values.json'

# Chess endgame scenario (Figure 1)
initial_position = {
    'K': chess.C3,
    'Q': chess.G3,
    'k': chess.C5
}

# Initialize the environment and the agent
env = ChessEnvironment(initial_position)
agent = RLAgent(learning_rate, initial_exploration_rate, discount_factor)

# Load saved Q-values if available
try:
    agent.q_values = load_q_values(save_file)
except FileNotFoundError:
    pass

reward_history = []

# Training loop
for episode in range(episodes):
    state = env.reset()
    legal_moves = env.get_legal_moves()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state, legal_moves)
        next_state, reward, done, next_legal_moves = env.step(action)
        agent.learn(state, action, reward, next_state, next_legal_moves)

        state = next_state
        legal_moves = next_legal_moves
        total_reward += reward

    reward_history.append(total_reward)

    # Update exploration rate
    agent.exploration_rate = max(min_exploration_rate, agent.exploration_rate - exploration_decay)

    # Print progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes}, Exploration Rate: {agent.exploration_rate:.4f}")

# Save Q-values
save_q_values(agent.q_values, save_file)

# Plot rewards
plot_rewards(reward_history)
