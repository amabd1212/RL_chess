import chess
from chess_environment import ChessEnvironment
from rl_agent import RLAgent
from utils import load_q_values

# Parameters
num_games = 1000
save_file = 'q_values.json'

# Chess endgame scenario (Figure 1)
initial_position = {
    'K': chess.C3,
    'Q': chess.G3,
    'k': chess.C5
}

# Initialize the environment and the agent
env = ChessEnvironment(initial_position)
agent = RLAgent(learning_rate=0.8, exploration_rate=0.0, discount_factor=0.99)

# Load the trained Q-values
agent.q_values = load_q_values(save_file)

wins = 0
draws = 0
losses = 0

# Testing loop
for game in range(num_games):
    state = env.reset()
    legal_moves = env.get_legal_moves()
    done = False

    while not done:
        action = agent.choose_action(state, legal_moves)
        next_state, reward, done, next_legal_moves = env.step(action)

        state = next_state
        legal_moves = next_legal_moves

    # Record the game result
    if reward == 10.0:  # Win
        wins += 1
    elif reward == -10.0:  # Draw
        draws += 1
    else:  # Loss
        losses += 1

# Print the testing results
print(f"Results over {num_games} games:")
print(f"Wins: {wins}")
print(f"Draws: {draws}")
print(f"Losses: {losses}")
