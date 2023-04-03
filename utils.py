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

def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def plot_data(data,yaxis, title, filename):
    plt.figure()
    plt.plot(data)
    plt.xlabel('Epochs (x1000)')
    plt.ylabel(yaxis)
    plt.title(title)
    # plt.savefig(filename)
    plt.show()

def plot_rewards(reward_history, window_size=1000):
    moving_average_rewards = [sum(reward_history[i:i+window_size])/window_size for i in range(len(reward_history) - window_size + 1)]

    plt.plot(moving_average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward over {window_size} Episodes")
    plt.show()
