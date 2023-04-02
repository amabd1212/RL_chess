import json
import matplotlib.pyplot as plt
import chess

def save_q_values(q_values, file_path):
    q_values_json = {}
    for key, value in q_values.items():
        state, action = key
        action_str = action.uci()
        q_values_json[f"{state}_{action_str}"] = value

    with open(file_path, 'w') as outfile:
        json.dump(q_values_json, outfile)

def load_q_values(file_path):
    with open(file_path, 'r') as infile:
        q_values_json = json.load(infile)

    q_values = {}
    for key, value in q_values_json.items():
        state, action_str = key.split('_')
        action = chess.Move.from_uci(action_str)
        q_values[(state, action)] = value

    return q_values

def plot_rewards(reward_history, window_size=100):
    moving_average_rewards = [sum(reward_history[i:i+window_size])/window_size for i in range(len(reward_history) - window_size + 1)]

    plt.plot(moving_average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward over {window_size} Episodes")
    plt.show()

def plot_moves(reward_history, window_size=100):
    moving_average_rewards = [sum(reward_history[i:i+window_size])/window_size for i in range(len(reward_history) - window_size + 1)]

    plt.plot(moving_average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Number of Moves")
    plt.title(f"Average Number of Moves over {window_size} Episodes")
    plt.show()

def plot_win_rate(reward_history, window_size=100):
    moving_average_rewards = [sum(reward_history[i:i+window_size])/window_size for i in range(len(reward_history) - window_size + 1)]
    
    plt.plot(moving_average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Win Rate")
    plt.title(f"Average Win Rate over {window_size} Episodes")
    plt.show()
