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
        state= self.get_state()
        self.board.push(action)
        next_state = self.get_state()
        reward = self.compute_reward(state,action)
        done = self.is_terminal()
        legal_moves = list(self.board.legal_moves)
        return next_state, reward, done, legal_moves

    def get_state(self):
        fen_parts = self.board.fen().split(" ")
        fen_without_clocks = " ".join(fen_parts[:4])
        # You can customize the state representation depending on your needs
        return fen_without_clocks
    
    def king_area(self, king_square, enemy_color):
        area = 0
        for move in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
            if not self.board.is_attacked_by(enemy_color, move) and not self.board.piece_at(move):
                area += 1
        return area

    def compute_reward(self,state,action):

        white_queen_positions = list(self.board.pieces(chess.QUEEN, chess.WHITE))
        white_king = self.board.king(chess.WHITE)
        black_king = self.board.king(chess.BLACK)

        white_king_target = chess.C6
        black_king_target = chess.C8
        white_queen_target = chess.C8

        if white_queen_positions:
            white_queen = white_queen_positions[0]
        else:
            white_queen = 0
        
        if black_king == black_king_target and white_king == white_king_target and white_queen == white_queen_target:
            return 10000  # Win
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_variant_draw() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return -10000  # Lose
        else:
            


            # Encourage the white pieces to control squares around the black king
            area = self.king_area(black_king, chess.WHITE)

            # Distance of the white king from its target square
            white_king_distance = self.distance(white_king, white_king_target)

            # Distance of the black king from its target square
            black_king_distance = self.distance(black_king, black_king_target)

            # Rank difference of black and white kings from their respective target ranks
            black_king_rank = chess.square_rank(black_king_target) - chess.square_rank(black_king)
            white_king_rank = abs(chess.square_rank(white_king_target) - chess.square_rank(white_king))
            white_queen_rank = abs(chess.square_rank(white_queen) - chess.square_rank(white_queen))

            # Distance between the white king and white queen
            king_2_queen = self.distance(white_king, white_queen)


            # Check if the white queen is under attack
            queen_under_attack = self.board.is_attacked_by(chess.BLACK, white_queen)

            # Encourage the white queen to control squares around the black king
            queen_control = 8 - sum([1 for square in chess.SquareSet(chess.BB_KING_ATTACKS[black_king]) if self.board.is_attacked_by(chess.WHITE, square)])
            
            # Calculate the reward based on the different components
            reward = (
                - 20 * area
                - 30 * (black_king_rank + white_king_rank + white_queen_rank)
                - 30 * white_king_distance
                - 10 * black_king_distance
                - 10 * king_2_queen
                - 30 * queen_control
            )

            # Penalize if the white queen is under attack
            if queen_under_attack:
                reward -= 5000 

            # Penalize if the white queen or king are in the wrong positions
            if white_queen == chess.C6 or white_king == chess.C7:
                reward -= 1000

            # Penalize if the white queen is in a rank higher than the black king
            if chess.square_rank(white_queen) >= chess.square_rank(black_king):
                reward -= 3000
            
            # Penalize if the white king is too far from the black king in terms of file
            if abs(chess.square_file(black_king) - chess.square_file(white_king)) > 2:
                reward -= 3000

            # Penalize if the white queen is too far from the black king in terms of file
            if chess.square_file(white_queen) < chess.square_file(black_king) and chess.square_file(white_king) < chess.square_file(black_king):
                reward -= 1000
            return reward




    def choose_random_move(self,legal_moves):
        return random.choice(legal_moves)

    def is_terminal(self):
        return self.board.is_game_over()

    def get_legal_moves(self):
        return list(self.board.legal_moves)
