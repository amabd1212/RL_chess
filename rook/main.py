import chess
import numpy as np
from chess_environment import ChessEnvironment
from rl_agent import RLAgent
from utils import save_q_values, load_q_values, plot_rewards, save_data, plot_data

# Parameters
episodes = 2000
initial_exploration_rate = 1
exploration_decay = 0.05
# min_exploration_rate = 0.1
learning_rate = 0.1
discount_factor = 0.99
save_file = 'q_values8.json'

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

win = 0
win_total = 0
reward_history = []
win_rate = []
moves = []
win_rate1000 = []
move_count1000 = []
reward_history1000 = []

# Training loop
for episode in range(episodes):
    state = env.reset()
    legal_moves = env.get_legal_moves()
    done = False
    turn = "white"
    total_reward = 0
    win = 0
    # print("new game")
    move_count = 0
    while not done:
        move_count +=1
        if turn  == "white":
            action = agent.choose_action(state, legal_moves)
        else:
            action = env.choose_random_move(legal_moves)
        next_state, reward, done, next_legal_moves = env.step(action)
        if reward == 10000:
            win = 1 
            win_total +=1
        if turn == "white":
            agent.learn(state, action, reward, next_state, next_legal_moves)
            turn = "black"
        else:
            turn = "white"
        if total_reward != 10000 or total_reward != -10000:
            total_reward += reward
        state = next_state
        legal_moves = next_legal_moves
    total_reward = (total_reward/move_count) + reward
    win_rate.append(win)
    moves.append(move_count)
    reward_history.append(total_reward)
    
    if (episode + 1) % 1000 == 0:
        win_rate1000.append(np.mean(win_rate[-1000:]))
        move_count1000.append(np.mean(moves[-1000:]))
        reward_history1000.append(np.mean(reward_history[-1000:]))

    # Update exploration rate 
    if initial_exploration_rate != 0:
        if (episode + 1) % int(0.8 * float(episodes)/(initial_exploration_rate/exploration_decay)) == 0:
            agent.exploration_rate = max(0, agent.exploration_rate - exploration_decay)
    # Print progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes}, Exploration Rate: {agent.exploration_rate:.4f}")
        print("Total",win_total)
        win_total =0
    

# Save Q-values
save_q_values(agent.q_values, save_file)
save_data(win_rate1000, 'data/win_rate100.json')
save_data(move_count1000, 'move_count100.json')
save_data(reward_history1000, 'reward_history100.json')

# Plot rewards
plot_data(win_rate1000,'Win Rate', 'Win Rate over 1000 Episodes', 'win_rate_every_100.png')
plot_data(move_count1000,'Average Move Count', 'Average Move Count over 1000 Episodes', 'move_count_avg_every_100.png')
plot_data(reward_history1000,'Average Reward', 'Average Reward over 1000 Episodes', 'win_rate_every_100.png')

