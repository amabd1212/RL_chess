import chess
from chess_environment import ChessEnvironment
from rl_agent import RLAgent, opponent_agent
from utils import load_q_values

# Parameters
num_games = 10000
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
opponent = opponent_agent()

# Load the trained Q-values
agent.q_values = load_q_values(save_file)

wins = 0
draws = 0
losses = 0

initial_exploration_rate = 1.0
exploration_decay = 0.003
min_exploration_rate = 0.01

# Testing loop
for game in range(num_games):
    print(f"Game {game + 1}/{num_games}")
    state = env.reset()
    legal_moves = env.get_legal_moves()
    done = False
    nrmovesInEpisode = 1
    #print("First board:")
    #print(env.board)
    active = 2
    while not done:
        if(active %2 == 0):
            #our agent chooses an action
            action = agent.choose_action(state, legal_moves)
            agent.exploration_rate = max(min_exploration_rate, agent.exploration_rate - exploration_decay)
            next_state, done, next_legal_moves= env.step(action)
            nrmovesInEpisode +=1
            #print("board after our action:")
            #print(env.board)
            #print("done after our step:",done)
            if (done):
                env.get_results(done, 0)
               # print("terminal after our move")
                break
        else:
            #the opponent chooses an random action
            oppponent_action = opponent_agent.action_choice(next_legal_moves)
            next_state, done, next_legal_moves= env.step_opponent(oppponent_action)
         #   print("board after opponents action:")
          #  print(env.board)
           # print("done after step_opponent:",done)
            if (done):
                env.get_results(done, 1)
            #    print("terminal after their move")
                break

        state = next_state
        legal_moves = next_legal_moves


    # Record the game result
    if done == 1:  # Win
        wins += 1
    elif done == 0:  # Draw
        draws += 1
    elif done == -1:  # Loss
        losses += 1

# Print the testing results
print(f"Results over {num_games} games:")
print(f"Wins: {wins}")
print(f"Draws: {draws}")
print(f"Losses: {losses}")