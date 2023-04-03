import chess
import numpy as np
from chess_environment import ChessEnvironment
from utils import save_q_values, load_q_values, plot_rewards, save_data, plot_data

# Parameters
episodes = 50000

# Chess endgame scenario (Figure 1)
initial_position = {
    'K': chess.C3,
    'Q': chess.G3,
    'k': chess.C5
}

# Initialize the environment and the agent
env = ChessEnvironment(initial_position)

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
    move_count = 0
    while not done:
        move_count +=1
        action = env.choose_random_move(legal_moves)
        next_state, reward, done, next_legal_moves = env.step(action)
        if reward == 10000:
            win = 1 
            win_total +=1
        if turn == "white":
            if total_reward != 10000 or total_reward != -10000:
                total_reward += reward
                turn = "black"
        else:
            turn = "white"

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

    # Print progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes}")
        print("Total",win_total)
        win_total =0
    

# Save Q-values
save_data(win_rate1000, 'win_rate_random1000.json')
save_data(move_count1000, 'move_count_random1000.json')
save_data(reward_history1000, 'reward_history_random1000.json')

# Plot rewards
plot_data(win_rate1000,'Win Rate', 'Win Rate over 1000 Episodes', 'win_rate_random1000.png')
plot_data(move_count1000,'Win Rate', 'Average Move Count over 1000 Episodes', 'move_count_avg_random1000.png')
plot_data(reward_history1000,'Average Reward', 'Average Reward over 1000 Episodes', 'reward_random1000.png')

