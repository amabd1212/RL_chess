import matplotlib.pyplot as plt
import json

data_folder = 'C://Users/amir0/OneDrive/Documents/RL/RL_chess/data'
# Load data from JSON files
move_count_Q1 = [53.283, 52.89, 55.743, 55.664, 53.833, 53.108, 58.146, 57.847, 57.053, 58.679, 59.071, 59.093, 58.742, 61.771, 62.38, 63.529, 63.233, 69.474, 64.892, 66.481, 69.224, 67.039, 69.95, 71.827, 74.353, 75.109, 74.638, 75.362, 77.355, 81.378, 77.96, 80.53, 82.501, 84.034, 86.904, 83.947, 85.094, 85.668, 85.123, 87.456, 88.17, 85.865, 88.597, 89.412, 86.662, 87.16, 77.268, 77.467, 68.443, 63.896, 58.244, 52.326, 52.129, 42.451, 39.739, 40.352, 26.462, 26.586, 25.683, 25.842, 25.829, 24.313, 25.893, 25.758, 24.339, 24.727, 24.181, 25.068, 24.28, 24.525]
move_count_Q2 = [54.483, 53.523, 52.733, 52.666, 54.202, 56.909, 55.905, 56.919, 55.41, 61.585, 57.167, 58.585, 61.029, 61.841, 63.571, 61.732, 64.268, 66.019, 66.545, 67.354, 70.913, 67.309, 72.034, 70.606, 71.197, 75.537, 77.532, 75.633, 75.435, 77.16, 77.397, 79.771, 78.151, 79.263, 83.561, 74.568, 78.084, 74.235, 75.767, 68.971, 68.019, 69.799, 60.183, 60.983, 57.642, 55.175, 48.719, 45.626, 43.036, 42.478, 37.841, 35.008, 34.252, 30.392, 29.645, 28.756, 23.985, 23.609, 23.815, 23.969, 24.549, 23.618, 24.068, 23.511, 23.241, 23.904, 23.774, 23.668, 23.563, 23.687] # Add your move_count_Q data here
move_count_Q3 = [53.631, 53.663, 52.764, 53.393, 52.019, 54.105, 57.743, 56.4, 57.444, 56.678, 58.46, 59.206, 59.893, 61.42, 61.264, 63.279, 62.251, 64.978, 67.308, 65.67, 67.802, 67.626, 70.629, 69.893, 70.314, 70.11, 72.149, 71.422, 69.373, 72.619, 72.805, 67.252, 64.895, 68.512, 61.121, 63.329, 61.056, 56.775, 55.327, 51.565, 50.251, 53.983, 46.347, 45.189, 42.427, 37.628, 38.753, 36.11, 32.266, 31.225, 29.566, 28.019, 27.903, 25.132, 23.257, 24.114, 20.408, 20.406, 20.294, 20.404, 20.67, 20.434, 20.41, 20.278, 20.378, 20.38, 20.444, 20.366, 20.518, 20.208] # Add your move_count_Q data here
move_count_Q4 = [53.631, 53.663, 52.764, 53.393, 52.019, 54.105, 57.743, 56.4, 57.444, 56.678, 58.46, 59.206, 59.893, 61.42, 61.264, 63.279, 62.251, 64.978, 67.308, 65.67, 67.802, 67.626, 70.629, 69.893, 70.314, 70.11, 72.149, 71.422, 69.373, 72.619, 72.805, 67.252, 64.895, 68.512, 61.121, 63.329, 61.056, 56.775, 55.327, 51.565, 50.251, 53.983, 46.347, 45.189, 42.427, 37.628, 38.753, 36.11, 32.266, 31.225, 29.566, 28.019, 27.903, 25.132, 23.257, 24.114, 20.408, 20.406, 20.294, 20.404, 20.67, 20.434, 20.41, 20.278, 20.378, 20.38, 20.444, 20.366, 20.518, 20.208] # Add your move_count_Q data here
move_count_Q5 = [54.207, 52.108, 52.197, 54.119, 55.01, 56.63, 56.64, 56.703, 58.918, 56.946, 58.897, 60.86, 59.876, 58.652, 61.684, 63.969, 60.275, 66.39, 64.924, 68.457, 65.312, 68.917, 70.633, 69.091, 69.047, 67.38, 70.706, 69.816, 68.393, 64.712, 67.229, 59.612, 62.624, 60.534, 56.896, 57.433, 51.982, 54.161, 49.739, 48.061, 44.338, 44.28, 39.562, 39.179, 37.36, 35.085, 36.763, 34.501, 31.745, 32.135, 29.267, 28.228, 26.981, 23.925, 23.231, 23.069, 20.602, 20.774, 20.512, 20.47, 20.625, 20.514, 20.328, 20.736, 20.77, 20.594, 20.618, 20.66, 20.392, 20.54] # Add your move_count_Q data here
move_count_Q6 = [53.176, 54.637, 52.963, 55.21, 52.637, 55.8, 53.043, 59.823, 55.103, 55.909, 59.124, 55.68, 59.523, 59.587, 57.749, 60.026, 61.088, 62.251, 62.502, 58.526, 65.068, 61.658, 62.074, 62.86, 64.104, 64.538, 66.831, 64.212, 66.201, 66.99, 62.669, 65.123, 66.222, 64.59, 64.759, 63.5, 63.382, 61.143, 59.393, 57.0, 57.603, 56.464, 56.355, 54.678, 53.277, 51.905, 47.304, 46.881, 43.497, 42.83, 42.788, 41.354, 41.933, 38.222, 38.528, 38.456, 35.31, 34.504, 32.901, 31.919, 31.382, 29.954, 29.424, 27.061, 27.039, 26.255, 25.508, 24.15, 23.667, 24.766]
move_count_Q7 = [54.813, 55.221, 54.416, 53.653, 55.954, 55.178, 54.198, 54.141, 57.107, 55.571, 57.85, 58.485, 58.283, 58.1, 61.296, 62.448, 58.959, 58.524, 61.921, 60.081, 58.607, 64.284, 63.381, 63.639, 61.171, 63.709, 63.196, 61.808, 63.685, 62.945, 59.893, 62.021, 60.071, 59.66, 60.61, 58.35, 60.337, 57.318, 59.446, 54.82, 52.923, 55.227, 52.03, 52.729, 53.399, 49.355, 47.222, 44.761, 46.464, 44.74, 43.752, 44.1, 43.536, 42.935, 41.059, 39.647, 36.272, 34.174, 32.564, 32.356, 33.014, 32.022, 31.511, 28.013, 28.836, 27.178, 24.938, 23.198, 22.836, 23.378]
move_count_Q8 = [52.348, 53.621, 53.012, 55.683, 54.335, 57.968, 57.865, 54.438, 55.485, 57.588, 60.152, 60.19, 58.086, 59.688, 61.058, 58.561, 59.858, 61.334, 60.315, 61.476, 61.829, 60.846, 63.037, 60.808, 59.931, 60.758, 58.734, 57.649, 58.505, 55.692, 57.698, 52.777, 52.788, 51.771, 51.85, 49.312, 45.765, 44.974, 44.856, 45.176, 41.229, 43.844, 37.802, 36.031, 36.059, 32.483, 33.685, 31.704, 30.357, 27.841, 27.47, 27.265, 26.411, 24.856, 23.569, 24.657, 21.908, 22.04, 21.768, 22.078, 22.2, 22.294, 21.868, 22.044, 21.858, 21.946, 22.232, 21.798, 22.196, 22.198]
move_count_Q9 = [53.875, 52.139, 54.452, 54.172, 53.274, 52.624, 56.633, 56.294, 53.727, 58.296, 54.431, 56.635, 59.575, 57.754, 59.413, 58.069, 59.735, 61.648, 61.294, 58.522, 58.512, 61.215, 58.439, 63.316, 62.185, 58.746, 59.572, 58.497, 58.274, 59.498, 58.681, 57.302, 58.798, 58.266, 54.296, 54.337, 56.582, 53.507, 52.756, 48.995, 50.316, 47.081, 46.695, 49.603, 45.411, 44.132, 43.025, 41.868, 46.258, 40.138, 36.455, 38.465, 37.637, 36.313, 35.666, 35.714, 31.834, 31.863, 31.908, 30.283, 28.313, 28.432, 28.95, 25.922, 26.813, 26.535, 26.612, 24.457, 24.756, 25.798]
move_count_Q89 = [54.955, 53.939, 54.125, 53.406, 53.756, 54.466, 54.093, 56.814, 55.77, 56.01, 55.15, 58.248, 58.287, 58.143, 58.283, 60.269, 56.806, 63.06, 61.007, 60.513, 60.792, 63.148, 61.483, 60.53, 60.771, 61.234, 59.449, 61.419, 61.905, 60.398, 60.676, 58.577, 57.187, 57.754, 57.1, 54.88, 56.961, 57.295, 52.946, 51.287, 51.791, 52.335, 48.504, 49.674, 47.694, 50.299, 50.174, 45.187, 44.176, 39.88, 39.841, 37.481, 36.091, 35.655, 36.046, 37.07, 33.62, 32.743, 34.378, 34.079, 32.743, 28.554, 28.0, 26.73, 26.616, 26.714, 25.524, 24.431, 23.821, 24.221]

win_rate_Q1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002, 0.001, 0.002, 0.002, 0.002, 0.002, 0.001, 0.005, 0.003, 0.002, 0.002, 0.001, 0.006, 0.007, 0.009, 0.009, 0.006, 0.011, 0.016, 0.019, 0.014, 0.018, 0.037, 0.029, 0.036, 0.04, 0.057, 0.065, 0.07, 0.073, 0.111, 0.147, 0.236, 0.28, 0.431, 0.484, 0.577, 0.657, 0.672, 0.781, 0.833, 0.832, 0.967, 0.968, 0.969, 0.976, 0.971, 0.981, 0.972, 0.975, 0.981, 0.978, 0.985, 0.98, 0.984, 0.982]
win_rate_Q2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.005, 0.008, 0.003, 0.01, 0.013, 0.02, 0.038, 0.059, 0.083, 0.122, 0.137, 0.228, 0.246, 0.303, 0.412, 0.431, 0.468, 0.571, 0.604, 0.654, 0.7, 0.719, 0.765, 0.796, 0.809, 0.881, 0.903, 0.916, 0.985, 0.985, 0.986, 0.989, 0.986, 0.989, 0.989, 0.988, 0.99, 0.986, 0.988, 0.987, 0.992, 0.992]
win_rate_Q3 = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001, 0.001, 0.0, 0.0, 0.002, 0.0, 0.003, 0.004, 0.001, 0.006, 0.006, 0.009, 0.011, 0.005, 0.006, 0.021, 0.036, 0.071, 0.129, 0.17, 0.238, 0.259, 0.304, 0.393, 0.432, 0.446, 0.522, 0.596, 0.59, 0.677, 0.687, 0.726, 0.775, 0.788, 0.817, 0.889, 0.862, 0.926, 0.936, 0.932, 0.986, 0.974, 0.977, 0.984, 0.981, 0.98, 0.983, 0.98, 0.987, 0.985, 0.974, 0.971, 0.974, 0.977]
win_rate_Q4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.001, 0.001, 0.001, 0.0, 0.002, 0.0, 0.001, 0.002, 0.003, 0.007, 0.004, 0.011, 0.01, 0.012, 0.029, 0.046, 0.058, 0.104, 0.12, 0.184, 0.224, 0.23, 0.296, 0.327, 0.361, 0.451, 0.434, 0.439, 0.561, 0.586, 0.644, 0.716, 0.708, 0.756, 0.812, 0.817, 0.872, 0.891, 0.89, 0.923, 0.944, 0.931, 0.975, 0.981, 0.978, 0.982, 0.982, 0.978, 0.98, 0.981, 0.976, 0.974, 0.984, 0.981, 0.98, 0.973]
win_rate_Q5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002, 0.002, 0.0, 0.001, 0.007, 0.006, 0.013, 0.019, 0.022, 0.054, 0.085, 0.114, 0.179, 0.188, 0.256, 0.269, 0.292, 0.376, 0.368, 0.457, 0.504, 0.55, 0.544, 0.658, 0.642, 0.683, 0.74, 0.737, 0.779, 0.816, 0.824, 0.851, 0.878, 0.897, 0.925, 0.932, 0.93, 0.984, 0.98, 0.976, 0.98, 0.983, 0.985, 0.983, 0.978, 0.976, 0.984, 0.981, 0.983, 0.978, 0.974]
win_rate_Q6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002, 0.0, 0.001, 0.001, 0.0, 0.0, 0.001, 0.001, 0.003, 0.002, 0.003, 0.005, 0.001, 0.005, 0.008, 0.012, 0.009, 0.017, 0.03, 0.033, 0.04, 0.077, 0.068, 0.09, 0.125, 0.136, 0.152, 0.211, 0.232, 0.244, 0.241, 0.317, 0.323, 0.347, 0.424, 0.468, 0.45, 0.488, 0.583, 0.596, 0.575, 0.555, 0.649, 0.669, 0.669, 0.756, 0.753, 0.772, 0.801, 0.85, 0.837, 0.833, 0.91, 0.92, 0.912, 0.931, 0.958, 0.955, 0.951]
win_rate_Q7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.002, 0.008, 0.022, 0.031, 0.025, 0.039, 0.054, 0.058, 0.071, 0.111, 0.108, 0.104, 0.165, 0.182, 0.181, 0.211, 0.239, 0.288, 0.255, 0.369, 0.385, 0.328, 0.398, 0.49, 0.451, 0.455, 0.567, 0.567, 0.585, 0.574, 0.61, 0.638, 0.653, 0.734, 0.756, 0.759, 0.795, 0.811, 0.8, 0.811, 0.878, 0.886, 0.893, 0.921, 0.944, 0.934, 0.948]
win_rate_Q8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.001, 0.0, 0.001, 0.0, 0.005, 0.001, 0.003, 0.01, 0.022, 0.018, 0.038, 0.036, 0.063, 0.067, 0.084, 0.128, 0.11, 0.186, 0.217, 0.197, 0.259, 0.286, 0.308, 0.394, 0.38, 0.44, 0.473, 0.478, 0.535, 0.566, 0.545, 0.673, 0.709, 0.7, 0.758, 0.807, 0.805, 0.859, 0.829, 0.888, 0.914, 0.907, 0.94, 0.952, 0.949, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
win_rate_Q9 = [0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.001, 0.001, 0.005, 0.004, 0.003, 0.019, 0.014, 0.012, 0.024, 0.029, 0.041, 0.049, 0.067, 0.074, 0.088, 0.123, 0.125, 0.143, 0.137, 0.192, 0.204, 0.224, 0.273, 0.312, 0.289, 0.319, 0.374, 0.389, 0.416, 0.483, 0.494, 0.5, 0.47, 0.628, 0.654, 0.581, 0.643, 0.698, 0.681, 0.697, 0.775, 0.757, 0.738, 0.794, 0.83, 0.831, 0.841, 0.894, 0.884, 0.891, 0.924, 0.93, 0.942, 0.929]
win_rate_Q89 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.001, 0.006, 0.001, 0.004, 0.002, 0.007, 0.015, 0.009, 0.024, 0.045, 0.044, 0.043, 0.091, 0.083, 0.099, 0.095, 0.121, 0.114, 0.124, 0.184, 0.148, 0.174, 0.234, 0.304, 0.334, 0.298, 0.379, 0.379, 0.371, 0.471, 0.442, 0.52, 0.467, 0.572, 0.61, 0.632, 0.657, 0.682, 0.7, 0.713, 0.767, 0.75, 0.742, 0.776, 0.758, 0.814, 0.819, 0.886, 0.885, 0.878, 0.91, 0.93, 0.946, 0.95]

# Create x-axis values (Episode)
x = list(range(1, len(move_count_Q1) + 1))

# Plot move counts
plt.figure()
plt.plot(x, move_count_Q1, label='α = 0.1')
plt.plot(x, move_count_Q2, label='α = 0.2')
plt.plot(x, move_count_Q3, label='α = 0.3')
plt.plot(x, move_count_Q4, label='α = 0.4')
plt.plot(x, move_count_Q5, label='α = 0.5')
plt.plot(x, move_count_Q6, label='α = 0.6')
plt.plot(x, move_count_Q7, label='α = 0.7')
plt.plot(x, move_count_Q8, label='α = 0.8')
plt.plot(x, move_count_Q89, label='α = 0.8 and γ = 0.9')
plt.plot(x, move_count_Q9, label='α = 0.9')
plt.xlabel('Episode')
plt.ylabel('Move Count')
plt.title('Average Move Count Over 1000 Episodes')
plt.legend()
plt.savefig('move_count_plot.png')

# Plot win rates
plt.figure()
plt.plot(x, win_rate_Q1, label='α = 0.1')
plt.plot(x, win_rate_Q2, label='α = 0.2')
plt.plot(x, win_rate_Q3, label='α = 0.3')
plt.plot(x, win_rate_Q4, label='α = 0.4')
plt.plot(x, win_rate_Q5, label='α = 0.5')
plt.plot(x, win_rate_Q6, label='α = 0.6')
plt.plot(x, win_rate_Q7, label='α = 0.7')
plt.plot(x, win_rate_Q8, label='α = 0.8')
plt.plot(x, win_rate_Q89, label='α = 0.8 and γ = 0.9')
plt.plot(x, win_rate_Q9, label='α = 0.9')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Win Rate Over 1000 Episodes')
plt.legend()
plt.savefig('win_rate_plot.png')

plt.show()
