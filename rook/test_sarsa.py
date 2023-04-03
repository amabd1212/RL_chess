import chess
from chess_environment import ChessEnvironment
from sarsa import SarsaAgent
from utils import load_q_values
import random

# Parameters
num_games = 5
save_file = 'sarsa_q_values.json'

# Chess endgame scenario (Figure 1)
initial_position = {
    'K': chess.D2,
    'R': chess.G3,
    'k': chess.C5
}

# Initialize the environment and the agent
env = ChessEnvironment(initial_position)
agent = SarsaAgent(learning_rate=0.1, exploration_rate=0.0, discount_factor=0.99)

# Load the trained Q-values
agent.q_values = load_q_values(save_file)

wins = 0
draws = 0
losses = 0

def choose_opponent_move(legal_moves):
    return random.choice(legal_moves)

# Testing loop
# Testing loop
for game in range(num_games):
    state = env.reset()
    legal_moves = env.get_legal_moves()
    done = False
    turn = "white"

    while not done:
        if turn == "white":
            action = agent.choose_action(state, legal_moves)
        else:  # turn == "black"
            action = choose_opponent_move(legal_moves)

        next_state, reward, done, next_legal_moves = env.step(action)
        print(state, action)
        state = next_state
        legal_moves = next_legal_moves
        turn = "black" if turn == "white" else "white"
    print("new game")


    # Record the game result
    if reward == 10000:  # Win
        wins += 1
    elif reward == -10000:  # Draw
        draws += 1
    else:  # Loss
        losses += 1

# Print the testing results
print(f"Results over {num_games} games:")
print(f"Wins: {wins}")
print(f"Draws: {draws}")
print(f"Losses: {losses}")