import chess
import random
import numpy as np

class ChessEnvironment:
    def __init__(self, initial_position):
        self.board = chess.Board()
        self.initial_position = initial_position
        self.move_count = 0

    def distance(self, square1, square2):
        x1, y1 = chess.square_file(square1), chess.square_rank(square1)
        x2, y2 = chess.square_file(square2), chess.square_rank(square2)
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def reset(self):
        self.board.clear()
        self.move_count = 0
        for piece, square in self.initial_position.items():
            self.board.set_piece_at(square, chess.Piece.from_symbol(piece))
        return self.get_state()

    def step(self, action):
        self.move_count +=1
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
        move_down = 0
        for move in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
            if not self.board.is_attacked_by(enemy_color, move) and not self.board.piece_at(move):
                area += 1
        
        
        king_rank = chess.square_rank(king_square)
        king_file = chess.square_file(king_square)

    # Check down the rank
        for i in range(king_rank - 1, -1, -1):
            square = chess.square(king_file, i)
            if not self.board.is_attacked_by(enemy_color, square) and not self.board.piece_at(square):
                area += 100
                move_down = 1
            else:
                break
        return area, move_down

    def compute_reward(self,state,action):

        white_rook_positions = list(self.board.pieces(chess.ROOK, chess.WHITE))
        white_king = self.board.king(chess.WHITE)
        black_king = self.board.king(chess.BLACK)

        white_king_target = chess.C6
        black_king_target = chess.C8
        white_rook_target = chess.H8

        if white_rook_positions:
            white_rook = white_rook_positions[0]
        else:
            white_rook = 0
        
        if black_king == black_king_target and white_king == white_king_target and white_rook == white_rook_target:
            return 10000  # Win
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_variant_draw() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return -10000  # Lose
        else:
            


            # Encourage the white pieces to control squares around the black king
            area, move_down = self.king_area(black_king, chess.WHITE)

            # Distance of the white king from its target square
            white_king_distance = self.distance(white_king, white_king_target)

            # Distance of the black king from its target square
            black_king_distance = self.distance(black_king, black_king_target)

            white_rook_distance = self.distance(white_rook, white_rook_target)

            # Rank difference of black and white kings from their respective target ranks
            black_king_rank = chess.square_rank(black_king_target) - chess.square_rank(black_king)
            white_king_rank = abs(chess.square_rank(white_king_target) - chess.square_rank(white_king))
            white_rook_rank = abs(chess.square_rank(white_rook) - chess.square_rank(white_rook))

            # Distance between the white king and white rook
            king_2_rook = self.distance(white_king, white_rook)


            # Check if the white rook is under attack
            rook_under_attack = self.board.is_attacked_by(chess.BLACK, white_rook)

            # Encourage the white rook to control squares around the black king
            rook_control = 8 - sum([1 for square in chess.SquareSet(chess.BB_KING_ATTACKS[black_king]) if self.board.is_attacked_by(chess.WHITE, square)])
            # Calculate the reward based on the different components
            reward = (
                - 20 * area
                - 30 * (black_king_rank + white_king_rank + white_rook_rank)
                - 30 * white_king_distance
                - 10 * black_king_distance
                - 20 * white_rook_distance
                - 30 * rook_control
            )

            # Penalize if the white rook is under attack
            if rook_under_attack:
                reward -= 5000 

            # Penalize if the white rook or king are in the wrong positions
            if white_rook == white_king_target or white_king == white_rook_target:
                reward -= 1000

            # Penalize if the white rook is in a rank higher than the black king
            if chess.square_rank(white_rook) >= chess.square_rank(black_king) and move_down == 1:
                reward -= 10000
            
            # Penalize if the white king is too far from the black king in terms of file
            if abs(chess.square_file(black_king) - chess.square_file(white_king)) > 1:
                reward -= 3000

            # Penalize if the white rook is too far from the black king in terms of rank
            if chess.square_rank(black_king) - chess.square_rank(white_rook) > 1 :
                reward -= 5000
            return reward




    def choose_random_move(self,legal_moves):
        return random.choice(legal_moves)

    def is_terminal(self):
        return self.board.is_game_over()

    def get_legal_moves(self):
        return list(self.board.legal_moves)
