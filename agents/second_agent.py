# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time

@register_agent("second_agent")
class SecondAgent(Agent):

    def __init__(self):
        super(SecondAgent, self).__init__()
        self.name = "SecondAgent"
        self.board_size = None
        self.max_step = 0
        self.time_limit = 1.0 # seconds
        self.moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left

    def step(self, chess_board, my_pos, adv_pos, max_step):
        start_time = time.time()
        depth = 0
        best_move = (my_pos, 0)  # Default move
        self.board_size = int(chess_board[0].size / 4)
        self.max_step = max_step
        while time.time() - start_time < self.time_limit:
            try:
                depth += 1
                score, move = self.minimax(chess_board, my_pos, adv_pos, depth, float("-inf"), float("inf"), True, start_time, self.time_limit, None)
                if move is not None:
                    best_move = move
                    # print(f"found best move in depth {depth}, best move {best_move}")
                else:
                    # print(f"no move found, return best move {best_move}")
                    break  # Break if no move found
            except self.AgentTimeoutException:
                return best_move
            pass
        return best_move


    def minimax(self, chess_board, my_pos, adv_pos, depth, alpha, beta, maximizing_player, start_time, time_limit, move):
        """
        Use minimax algorithm with alpha-beta pruning to find the best move

        Parameters:
        chess_board (np.array): the chess board.
        my_pos (tuple): agent's position
        adv_pos (tuple): adversary's position
        depth (int): current depth
        alpha (float): alpha value
        beta (float): beta value
        maximizing_player (bool): true if it's to maximize the result
        start_time (float): start time of the algo
        time_limit (float): time limit for the step
        move (tuple): the move that led to the current chess_board state

        Returns:
        tuple: score, move
        """

        if time.time() - start_time > time_limit:
            raise self.AgentTimeoutException()
        if depth == 0:
            return self.evaluate_state(chess_board, my_pos, adv_pos, move), None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in self.find_possible_moves(chess_board, my_pos, adv_pos):
                new_chess_board = self.make_move(chess_board, my_pos, move)
                eval, _ = self.minimax(new_chess_board, move[0], adv_pos, depth - 1, alpha, beta, False, start_time, time_limit, move)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in self.find_possible_moves(chess_board, adv_pos, my_pos):
                new_chess_board = self.make_move(chess_board, adv_pos, move)
                eval, _ = self.minimax(new_chess_board, my_pos, move[0], depth - 1, alpha, beta, True, start_time, time_limit, move)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def make_move(self, chess_board, player_pos, move):
        """
        Make a move and update the chess board with the new barrier.

        Parameters
        ----------
        chess_board : np.array
            The current state of the chess board.
        player_pos : tuple
            The current position of the player making the move.
        move : tuple
            The move to make, which is a tuple containing the new position and the barrier direction.

        Returns
        -------
        np.array
            The updated state of the chess board.
        """
        new_chess_board = np.copy(chess_board)
        new_pos, barrier_dir = move

        # Place the barrier at the new position
        r, c = new_pos
        if 0 <= r < self.board_size and 0 <= c < self.board_size:
            new_chess_board[r, c, barrier_dir] = True

            # Also set the barrier on the opposite cell in the corresponding direction
            if barrier_dir == 0 and r > 0:  # Up
                new_chess_board[r-1, c, 2] = True
            elif barrier_dir == 1 and c < self.board_size - 1:  # Right
                new_chess_board[r, c+1, 3] = True
            elif barrier_dir == 2 and r < self.board_size - 1:  # Down
                new_chess_board[r+1, c, 0] = True
            elif barrier_dir == 3 and c > 0:  # Left
                new_chess_board[r, c-1, 1] = True

        return new_chess_board


    # Copied and modified from world.py
    def check_endgame(self, chess_board, my_pos, adv_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        board_size = int(chess_board[0].size / 4)
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        return True, p0_score, p1_score

    # Copy from world.py
    def check_boundary(self, pos):
        r, c = pos
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def find_possible_moves(self, chess_board, my_pos, adv_pos):
        """
        Find all the possible moves given the current position of the player and the adversary player position
        Parameters
        ----------
        chess_board : numpy.ndarray of shape (board_size, board_size, 4)
            The chess board.
        my_pos : tuple
            The position of the agent.
        adv_pos : tuple
            The position of the adversary.
        Returns
        -------
        possible_moves : set of tuple
            The position of the move and the direction
        """
        possible_moves = set()
        visited = set()
        queue = [(my_pos, 0)]  # Include current position and steps taken

        while queue:
            current_pos, steps = queue.pop(0)
            if steps > self.max_step:
                continue
            visited.add(current_pos)

            # Add barriers for the current position
            for barrier_dir in range(4):
                if not chess_board[current_pos[0], current_pos[1], barrier_dir]:
                    possible_moves.add((current_pos, barrier_dir))

            if steps < self.max_step:
                # Check each direction for possible movement
                for dir, move in enumerate(self.moves):
                    next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

                    if self.check_boundary(next_pos) and next_pos != adv_pos and not chess_board[current_pos[0], current_pos[1], dir]:
                        if next_pos not in visited:
                            queue.append((next_pos, steps + 1))
                            visited.add(next_pos)

                            # Add barriers for the next position
                            for barrier_dir in range(4):
                                if not chess_board[next_pos[0], next_pos[1], barrier_dir]:
                                    possible_moves.add((next_pos, barrier_dir))

        # print(f"Possible moves from position {my_pos} with max step {self.max_step}: {possible_moves}")
        return list(possible_moves)

    def evaluate_state(self, chess_board, my_pos, adv_pos, move):
        # count number of accessible blocks for agent and for adversary
        my_accessible = self.count_accessible_blocks(chess_board, my_pos, adv_pos)
        adv_accessible = self.count_accessible_blocks(chess_board, adv_pos, my_pos)

        # score based on the difference between agent and adversary
        score = my_accessible - adv_accessible
        # print(f"Evaluating state. \n chess_board {chess_board} \n my_pos {my_pos} dir {move[1]} My accessible: {my_accessible}, adv_pos {adv_pos} Adversary accessible: {adv_accessible}, Score: {score}")
        return score

    def count_accessible_blocks(self, chess_board, start_pos, adv_pos):
        visited = set()
        queue = [start_pos]

        while queue:
            r, c = queue.pop(0)
            if (r, c) in visited or (r, c) == adv_pos:
                continue
            visited.add((r, c))

            for i, move in enumerate(self.moves):
                nr, nc = r + move[0], c + move[1]
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    # adversary's position can't be counted
                    if (nr, nc) == adv_pos:
                        continue
                    # make sure there's no barrier
                    if not chess_board[r, c, i] and (nr, nc) not in visited:
                        queue.append((nr, nc))

        return len(visited)


    def check_valid_step(self, chess_board, start_pos, end_pos, barrier_dir):
        # make sure the end position is inside the board and there's no barrier
        if not self.check_boundary(end_pos) or chess_board[end_pos[0], end_pos[1], barrier_dir]:
            return False

        if start_pos == end_pos:
            return True

        # make sure the end_pos is reachable within max_step (BFS)
        queue = [(start_pos, 0)]
        visited = set()
        while queue:
            current_pos, steps = queue.pop(0)
            if steps > self.max_step:
                break
            if current_pos == end_pos:
                return True
            visited.add(current_pos)
            for move in self.moves:
                next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
                if self.check_boundary(next_pos) and next_pos not in visited:
                    queue.append((next_pos, steps + 1))
        return False

    class AgentTimeoutException(Exception):
        """throw this exception when almost timeout"""
    pass
