import chess
import numpy as np
from chess_environment import ChessEnvironment
from rl_agent import RLAgent, opponent_agent
from utils import save_q_values, load_q_values, plot_rewards, plot_win_rate

# Parameters
episodes = 50000
initial_exploration_rate = 1.0
exploration_decay = 0.0001

min_exploration_rate = 0.01
learning_rate = 0.05
discount_factor = 0.95
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
opponent = opponent_agent()
# Load saved Q-values if available
try:
    agent.q_values = load_q_values(save_file)
except FileNotFoundError:
    pass

reward_history = []
win_history = []

# Training loop
for episode in range(episodes):
    state = env.reset()
    legal_moves = env.get_legal_moves()
    done = False
 
    total_reward = 0
    total_winn = 0
    active = 2
    while not done:
        if(active%2 == 0):
            # our agent chooses an action
            action = agent.choose_action(state, legal_moves)
            next_state, reward, done, next_legal_moves = env.step_train(action)
            ##just checking if its winning, average reward graph might be misleading
            if(done):
                if(env.board.is_checkmate):
                    total_winn+=1
            agent.learn(state, action, reward, next_state, next_legal_moves)
            total_reward += reward

         #the opponent chooses an random action
        else:
            oppponent_action = opponent_agent.action_choice(legal_moves)
            next_state, done, next_legal_moves= env.step_opponent(oppponent_action)
            state = next_state
            legal_moves = next_legal_moves
        active +=1
        state = next_state
        legal_moves = next_legal_moves

    reward_history.append(total_reward)
    win_history.append(total_winn)

    # Update exploration rate
    agent.exploration_rate = max(min_exploration_rate, agent.exploration_rate - exploration_decay)

    # Print progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes}, Exploration Rate: {agent.exploration_rate:.4f}")

# Save Q-values
save_q_values(agent.q_values, save_file)

# Plot rewards
plot_rewards(reward_history)
plot_win_rate(win_history)
