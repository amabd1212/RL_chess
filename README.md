# Reinforcement Learning Chess Endgame
Created by: Amir Abdul Aziz(s4019555)  and Mohamed Hassan(s4327276)


This project demonstrates the application of Q-learning and ε-greedy exploration to train an agent to play chess endgame scenarios. The agent is trained on a specific endgame scenario and learns an optimal policy for playing the game.

## Dependencies

- Python 3.6 or later
- `python-chess` library

You can install the required package using the following command:

pip install python-chess

Files
- chess_environment.py: Contains the ChessEnvironment class for simulating chess endgame scenarios.
- rl_agent.py: Contains the RLAgent class implementing Q-learning and ε-greedy exploration.
- utils.py: Contains utility functions for saving/loading Q-values and plotting the agent's learning progress.
- main.py: The main training script that integrates all components and trains the agent.
- test.py: A script to test the agent's performance after training.

Usage
1. Run the main.py script to train the agent on the chess endgame scenario:
    python main.py
2. Run the test.py script to test the agent on the chess endgame scenario:
    python test.py