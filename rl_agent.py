import numpy as np

class RLAgent:
    def __init__(self, learning_rate, exploration_rate, discount_factor):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.discount_factor = discount_factor
        self.q_values = {}  # Initialize an empty Q-values dictionary

    def choose_action(self, state, legal_moves):
        if np.random.rand() < self.exploration_rate:  # Exploration
            return np.random.choice(legal_moves)
        else:  # Exploitation
            return self._greedy_action(state, legal_moves)

    def learn(self, state, action, reward, next_state, next_legal_moves):
        current_q_value = self._get_q_value(state, action)
        max_next_q_value = self._max_q_value(next_state, next_legal_moves)
        target_q_value = reward + self.discount_factor * max_next_q_value
        self._update_q_value(state, action, current_q_value, target_q_value)

    def _greedy_action(self, state, legal_moves):
        max_q_value = float('-inf')
        best_move = None
        for move in legal_moves:
            q_value = self._get_q_value(state, move)
            if q_value > max_q_value and q_value != 0:
                max_q_value = q_value
                best_move = move
        if best_move == None:
            return np.random.choice(legal_moves)
        # print(state,best_move,max_q_value)
        return best_move

    def _get_q_value(self, state, action):
        return self.q_values.get((state, action), 0)

    def _max_q_value(self, state, legal_moves):
        max_q_value = float('-inf')
        if len(legal_moves) != 0 :
            for move in legal_moves:
                q_value = self._get_q_value(state, move)
                if q_value > max_q_value:
                    max_q_value = q_value
        else:
            max_q_value = 0
        return max_q_value

    def _update_q_value(self, state, action, current_q_value, target_q_value):
        error = target_q_value - current_q_value
        self.q_values[(state, action)] = current_q_value + self.learning_rate * error
