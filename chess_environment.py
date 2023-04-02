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
        for move in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
            if not self.board.is_attacked_by(enemy_color, move) and not self.board.piece_at(move):
                area += 1
        return area

    def compute_reward(self,state,action):
        white_queen_positions = list(self.board.pieces(chess.QUEEN, chess.WHITE))
        white_king = self.board.king(chess.WHITE)
        black_king = self.board.king(chess.BLACK)
        if white_queen_positions:
            white_queen = white_queen_positions[0]
        else:
            white_queen = 0
        if black_king == chess.C8 and white_king == chess.C6 and white_queen == chess.C7:
            return 10000  # Win
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_variant_draw() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return -10000  # Lose
        else:
            

            white_king_target = chess.C6
            black_king_target = chess.C8
            white_queen_target = chess.C7
            area = self.king_area(black_king, chess.WHITE)
            white_king_distance = self.distance(white_king, white_king_target)
            black_king_distance = self.distance(white_king, white_king_target)
            black_king_rank = 7 - chess.square_rank(black_king)
            white_king_rank = abs(5 - chess.square_rank(white_king))
            white_queen_rank = abs(6 - chess.square_rank(white_queen))
            king_2_queen = self.distance(white_king, white_queen)

            if white_queen_positions:
                white_queen_distance = self.distance(white_queen, white_queen_target)
                queen_under_attack = self.board.is_attacked_by(chess.BLACK, white_queen_positions[0])

            else:
                white_queen_distance = 0
                queen_under_attack = False
            too_far = 0
            # Encourage the white queen to control squares around the black king
            queen_control = 8 - sum([1 for square in chess.SquareSet(chess.BB_KING_ATTACKS[black_king]) if self.board.is_attacked_by(chess.WHITE, square)])

            reward = (
                - 20 * area
                - 100 * (black_king_rank + white_king_rank + white_queen_rank)
                - 10 * white_king_distance
                - 5 * white_queen_distance
                - 10 * black_king_distance
                - 10 * king_2_queen
                - 20 * queen_control
            )

            # print(state,action,area,white_king_distance,white_queen_distance,queen_control,queen_sacrificed,reward)
            if queen_under_attack:
                reward -= 5000 

            if white_queen == chess.C6 or white_king == chess.C7:
                reward -= 1000

            return reward




    def choose_random_move(self,legal_moves):
        return random.choice(legal_moves)

    def is_terminal(self):
        return self.board.is_game_over()

    def get_legal_moves(self):
        return list(self.board.legal_moves)
