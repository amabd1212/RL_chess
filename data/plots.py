import matplotlib.pyplot as plt
import json

data_folder = 'C://Users/amir0/OneDrive/Documents/RL/RL_chess/data'
# Load data from JSON files
with open(f'{data_folder}/final_move_count_Q_1000.json', 'r') as f:
    move_count_Q = json.load(f)
with open(f'{data_folder}/move_count_random1000.json', 'r') as f:
    move_count_random = json.load(f)
with open(f'{data_folder}/final_move_count_Sarsa_1000.json', 'r') as f:
    move_count_sarsa = json.load(f)

with open(f'{data_folder}/final_no_move_count_Q_1000.json', 'r') as f:
    no_move_count_Q = json.load(f)
with open(f'{data_folder}/final_no_move_count_Sarsa_1000.json', 'r') as f:
    no_move_count_sarsa = json.load(f)

with open(f'{data_folder}/final_win_rate_Q_1000.json', 'r') as f:
    win_rate_Q = json.load(f)
with open(f'{data_folder}/win_rate_random1000.json', 'r') as f:
    win_rate_random = json.load(f)
with open(f'{data_folder}/final_win_rate_sarsa_1000.json', 'r') as f:
    win_rate_sarsa = json.load(f)

with open(f'{data_folder}/final_no_win_rate_Q_1000.json', 'r') as f:
    no_win_rate_Q = json.load(f)
with open(f'{data_folder}/final_no_win_rate_sarsa_1000.json', 'r') as f:
    no_win_rate_sarsa = json.load(f)


    
# move_count_S5 = [52.526, 52.503, 53.355, 55.345, 53.518, 55.577, 56.593, 55.105, 59.298, 56.514, 57.606, 59.743, 60.258, 61.676, 62.109, 60.605, 63.622, 63.529, 64.283, 65.719, 68.879, 66.624, 69.576, 69.747, 68.412, 68.643, 70.211, 65.032, 70.314, 65.663, 68.214, 65.58, 61.819, 62.16, 59.811, 55.95, 55.148, 51.441, 53.836, 52.25, 51.258, 47.872, 43.602, 45.871, 39.512, 36.89, 35.309, 36.307, 32.171, 31.766, 29.812, 29.45, 28.61, 25.558, 25.689, 26.201, 22.891, 23.327, 23.403, 23.474, 22.728, 22.709, 23.362, 23.41, 23.141, 23.104, 23.053, 23.588, 22.907, 23.443]
# win_rate_S5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.001, 0.0, 0.0, 0.001, 0.001, 0.003, 0.002, 0.006, 0.013, 0.017, 0.017, 0.02, 0.032, 0.081, 0.077, 0.095, 0.16, 0.138, 0.208, 0.253, 0.27, 0.335, 0.374, 0.416, 0.476, 0.487, 0.491, 0.598, 0.599, 0.638, 0.729, 0.709, 0.757, 0.808, 0.786, 0.854, 0.866, 0.874, 0.93, 0.936, 0.935, 0.969, 0.969, 0.976, 0.959, 0.974, 0.975, 0.97, 0.976, 0.976, 0.976, 0.969, 0.971, 0.976, 0.978]

# move_count_S8 = [53.277, 51.817, 55.434, 55.293, 54.56, 54.817, 55.828, 55.42, 57.653, 56.569, 59.212, 57.781, 59.827, 60.092, 62.277, 62.001, 62.457, 62.739, 63.64, 63.741, 61.281, 60.642, 62.349, 59.374, 61.311, 60.17, 60.145, 58.868, 57.656, 58.912, 55.602, 53.784, 50.593, 50.513, 48.632, 48.423, 44.379, 43.111, 44.289, 43.394, 43.917, 39.989, 35.007, 35.674, 37.042, 33.901, 33.423, 32.44, 30.235, 30.954, 28.692, 28.187, 27.375, 25.598, 24.735, 24.725, 22.78, 22.776, 22.472, 22.42, 22.702, 22.754, 22.212, 22.788, 22.64, 22.56, 22.612, 22.346, 22.796, 22.472]
# win_rate_S8 = [0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.002, 0.001, 0.011, 0.014, 0.023, 0.032, 0.054, 0.068, 0.06, 0.113, 0.118, 0.117, 0.176, 0.188, 0.204, 0.255, 0.289, 0.33, 0.363, 0.416, 0.454, 0.485, 0.527, 0.571, 0.581, 0.599, 0.721, 0.702, 0.697, 0.776, 0.72, 0.749, 0.817, 0.831, 0.854, 0.9, 0.902, 0.926, 0.959, 0.955, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# Create x-axis values (Episode)

x = list(range(1, len(move_count_Q) + 1))

# Plot move counts
plt.figure()
plt.plot(x, move_count_Q, label='Q-Learning | ε = 1.00 | decay = 0.05')
plt.plot(x, move_count_random, label='Random')
plt.plot(x, move_count_sarsa, label='SARSA | ε = 1.00 | decay = 0.05')
plt.plot(x, no_move_count_Q, label='Q-Learning | ε = 0.05 | decay = 0')
plt.plot(x, no_move_count_sarsa, label='SARSA | ε = 0.05 | decay = 0')
plt.xlabel('Episode')
plt.ylabel('Move Count')
plt.title('Average Move Count Over 1000 Episodes')
plt.legend()
plt.savefig('move_count_plot.png')

# Plot win rates
plt.figure()
plt.plot(x, win_rate_Q, label='Q-Learning | ε = 1.00 | decay = 0.05')
plt.plot(x, win_rate_random, label='Random')
plt.plot(x, win_rate_sarsa, label='SARSA | ε = 1.00 | decay = 0.05')
plt.plot(x, no_win_rate_Q, label='Q-Learning | ε = 0.05 | decay = 0')
plt.plot(x, no_win_rate_sarsa, label='SARSA | ε = 0.05 | decay = 0')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Win Rate Over 1000 Episodes')
plt.legend()
plt.savefig('win_rate_plot.png')

plt.show()
