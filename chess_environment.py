import chess
import random
import numpy as np

class ChessEnvironment:
    def __init__(self, initial_position):
        self.board = chess.Board()
        self.initial_position = initial_position

    def distance(self, square1, square2):
        x1, y1 = chess.square_file(square1), chess.square_rank(square1)
        x2, y2 = chess.square_file(square2), chess.square_rank(square2)
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def reset(self):
        self.board.clear()
        for piece, square in self.initial_position.items():
            self.board.set_piece_at(square, chess.Piece.from_symbol(piece))
        return self.get_state()

    def step(self, action):
        self.board.push(action)
        next_state = self.get_state()
        reward = self.compute_reward()
        done = self.is_terminal()
        legal_moves = list(self.board.legal_moves)
        return next_state, reward, done, legal_moves

    def get_state(self):
        fen_parts = self.board.fen().split(" ")
        fen_without_clocks = " ".join(fen_parts[:4])
        # You can customize the state representation depending on your needs
        return fen_without_clocks

    def compute_reward(self):
        if self.board.is_checkmate():
            # print("Win",self.get_state())
            return 10.0  # Win
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            # print("Stale",self.get_state())
            return -1  # Draw
        elif self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            # print("75",self.get_state())
            return -1  # Draw
        elif self.board.is_variant_draw():
            # print("Draw",self.get_state())
            return -1  # Draw
        else:
            # Calculate distances to the goal positions
            white_king_distance = self.distance(self.board.king(chess.WHITE), chess.C6)
            black_king_distance = self.distance(self.board.king(chess.BLACK), chess.C8)

            white_queen_positions = list(self.board.pieces(chess.QUEEN, chess.WHITE))
            # white_rook_positions = list(self.board.pieces(chess.ROOK, chess.WHITE))
            if not white_queen_positions:
                return -2  # Large negative reward for losing the queen

            if white_queen_positions:
                white_queen_distance = self.distance(white_queen_positions[0], chess.C7)
                return -0.01 * (white_king_distance + white_queen_distance + black_king_distance)
            # elif white_rook_positions:
            #     white_rook_distance = self.distance(white_rook_positions[0], chess.H8)
            #     return -0.01 * (white_king_distance + white_rook_distance + black_king_distance)
            else:
                return 0



    def choose_random_move(self,legal_moves):
        return random.choice(legal_moves)

    def is_terminal(self):
        return self.board.is_game_over()

    def get_legal_moves(self):
        return list(self.board.legal_moves)
