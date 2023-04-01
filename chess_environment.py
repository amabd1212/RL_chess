import chess
import numpy as np

class ChessEnvironment:
    def __init__(self, initial_position):
        self.board = chess.Board()
        self.initial_position = initial_position

    def reset(self):
        self.board.clear()
        for piece, square in self.initial_position.items():
            self.board.set_piece_at(square, chess.Piece.from_symbol(piece))
        return self.get_state()

    def step_train(self, action):
        self.board.push(action)
        next_state = self.get_state()
        reward = self.compute_reward()
        done = self.is_terminal()
        legal_moves = list(self.board.legal_moves)
        return next_state, reward, done, legal_moves
    
    def step(self, action):
        self.board.push(action)
        next_state = self.get_state()
        done = self.is_terminal()
        legal_moves = list(self.board.legal_moves)
        return next_state, done, legal_moves

    def step_opponent(self, action):
        self.board.push(action)
        next_state = self.get_state()
        done = self.is_terminal()
        legal_moves = list(self.board.legal_moves)
        return next_state, done, legal_moves
    
    def get_state(self):
        # You can customize the state representation depending on your needs
        return self.board.fen()
    
    def get_results(self, done, agentNr):
        #our agent
        if(agentNr == 0):
            if(self.board.is_checkmate):
                done = 1
                return done
            elif(self.board.is_variant_draw() or self.board.is_stalemate or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition()):
                done = 0
                return done
        #the other agnet
        elif(agentNr == 1):
            if(self.board.is_checkmate):
                done = -1
                return done
            elif(self.board.is_variant_draw() or self.board.is_stalemate or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition()):
                done = 0
                return done
        

    def compute_reward(self):
        if self.board.is_checkmate():
            return 20.0  # Win
        elif self.board.is_variant_draw() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition() or self.board.is_variant_loss():
            return -10  # Draw
        elif self.board.is_check():
            return -1  # Check
        else:
            return -0.01  # Other moves

    def is_terminal(self):
        return self.board.is_game_over()
    def get_legal_moves(self):
        return list(self.board.legal_moves)
    
    def print_board(self):
        return print(self.board)
    
   