# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time

@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.board_size = None
        self.max_step = 0
        self.time_limit = 1.99 # seconds
        self.moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        self.moves_cache = {}  # Cache for storing possible moves
        self.max_depth = 0
        self.max_breadth = 0
        self.use_move_ordering = False
        self.use_heuristics = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        start_time = time.time()
        depth = 0
        best_move = (my_pos, 0)  # Default move
        best_score = 0
        self.board_size = int(chess_board[0].size / 4)
        self.max_step = max_step
        while time.time() - start_time < self.time_limit:
            try:
                depth += 1
                score, move = self.minimax(chess_board, my_pos, adv_pos, depth, float("-inf"), float("inf"), True, start_time, self.time_limit, None)
                if move is not None:
                    if score == float('inf'):
                        # Found a definitive winning or losing move, no need to search further
                        print(f"Definitive outcome found at depth {depth}. Move: {move}, Score: {score}")
                        return move
                    best_move = move
                    best_score = score
                    print(f"found best move in depth {depth}, best move {best_move} score {score}")
                else:
                    # print(f"no move found, return best move {best_move}")
                    break  # Break if no move found
            except self.AgentTimeoutException as e:
                if depth-1 > self.max_depth:
                    print(f"Timeout: Final depth reached = {depth-1} best move {best_move} best score {best_score}")
                if e.best_move is not None and e.score > best_score:
                    return e.best_move
                else:
                    return best_move
            pass
        if depth > self.max_depth:
            self.max_depth = depth
            print(f"Final depth reached = {depth}")
        return best_move


    def minimax(self, chess_board, my_pos, adv_pos, depth, alpha, beta, maximizing_player, start_time, time_limit, previous_move):
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

        # # Check if the game has ended
        # is_endgame, p0_score, p1_score = self.check_endgame(chess_board, my_pos, adv_pos)
        # if is_endgame:
        #     return (float('inf') if p0_score > p1_score else -float('inf')), None
    # Check if the game has ended
        is_endgame, p0_score, p1_score = self.check_endgame(chess_board, my_pos, adv_pos)
        if is_endgame:
            return (float('inf') if p0_score > p1_score else -float('inf')), None

        if depth == 0:
            return self.evaluate_state(chess_board, my_pos, adv_pos), None
        breadth = 0
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in self.find_possible_moves(chess_board, my_pos, adv_pos, self.use_move_ordering):
                new_chess_board = self.make_move(chess_board, move)
                # print(f"current move: {move} depth {depth} maximizing: {maximizing_player}")
                try:
                    eval, _ = self.minimax(new_chess_board, move[0], adv_pos, depth - 1, alpha, beta, False, start_time, time_limit, move)
                except self.AgentTimeoutException:
                    raise self.AgentTimeoutException(max_eval, best_move)
                pass
                # print(f"from minimize function:{eval}")
                breadth += 1
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            if previous_move is None: # top level depth
                print(f"Breadth considered: {breadth}")
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in self.find_possible_moves(chess_board, adv_pos, my_pos, self.use_move_ordering):
                new_chess_board = self.make_move(chess_board, move)
                # print(f"current move: {move} depth {depth}  maximizing: {maximizing_player}")
                try:
                    eval, _ = self.minimax(new_chess_board, my_pos, move[0], depth - 1, alpha, beta, True, start_time, time_limit, move)
                except self.AgentTimeoutException:
                    raise self.AgentTimeoutException(min_eval, best_move)
                pass
                # print(f"from maximize function:{eval}")
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def make_move(self, chess_board, move):
        """
        Make a move and update the chess board with the new barrier.

        Parameters
        ----------
        chess_board : np.array
            The current state of the chess board.
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
                    if chess_board[r, c, dir + 1]:
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

    def find_possible_moves(self, chess_board, my_pos, adv_pos, sort_move):
        possible_moves = self.generate_possible_moves(chess_board, my_pos, adv_pos)
        if sort_move:
            sorted_moves = sorted(possible_moves, key=lambda move: self.evaluate_move(chess_board, move, my_pos, adv_pos))
            # print(f"sorted moves from position {my_pos} with max step {self.max_step}: {sorted_moves}")
            return sorted_moves
        return possible_moves

    def generate_possible_moves(self, chess_board, my_pos, adv_pos):
        """
        generate all the possible moves given the current position of the player and the adversary player position
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

        # check if the current board state is already cached
        # print(f"generate possible move: my_pos {my_pos} adv_pos {adv_pos}")
        board_key = self.board_to_key(chess_board, my_pos, adv_pos)
        if board_key in self.moves_cache:
            return self.moves_cache[board_key]

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
        # Log the number of possible moves (breadth) for this call
        if len(possible_moves) > self.max_breadth:
            self.max_breadth = len(possible_moves)
            print(f"Max of possible moves at this level = {len(possible_moves)}")
        # Cache the result
        self.moves_cache[board_key] = list(possible_moves)
        return list(possible_moves)

    def evaluate_state(self, chess_board, my_pos, adv_pos):
        # count number of accessible blocks for agent and for adversary
        my_accessible = self.count_accessible_blocks(chess_board, my_pos, adv_pos)
        adv_accessible = self.count_accessible_blocks(chess_board, adv_pos, my_pos)

        # score based on the difference between agent and adversary
        score = my_accessible - adv_accessible

        if self.use_heuristics:
            # Check if surrounded by 3 walls
            if self.is_surrounded_by_three_walls(chess_board, my_pos):
                score -= 100  # Assign a large negative value

            # Check for move options
            move_options = len(self.find_possible_moves(chess_board, my_pos, adv_pos, False))
            score += move_options  # More move options is better
        # if move is not None:
        #     print(f"Evaluating state. \n chess_board {chess_board} \n my_pos {my_pos} dir {move[1]} My accessible: {my_accessible}, adv_pos {adv_pos} Adversary accessible: {adv_accessible}, Score: {score}")
        # else:
        #     print(f"Evaluating state. \n chess_board {chess_board} \n my_pos {my_pos} My accessible: {my_accessible}, adv_pos {adv_pos} Adversary accessible: {adv_accessible}, Score: {score}")
        return score

    def is_surrounded_by_three_walls(self, chess_board, pos):
        r, c = pos
        walls = 0
        for i in range(4):
            if chess_board[r, c, i]:
                walls += 1
        return walls == 3
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

    def board_to_key(self, chess_board, my_pos, adv_pos):
        # Convert the 3D numpy array into a string representation
        board_str = np.array2string(chess_board)
        return (board_str, my_pos, adv_pos)

    def evaluate_move(self, chess_board, move, my_pos, adv_pos):
        new_pos, barrier_dir = move
        move_score = 0

        # Calculate distance to center and opponent
        center = (self.board_size // 2, self.board_size // 2)
        # move_score -= self.distance(new_pos, center)  # Prefer moves closer to center
        move_score += self.distance(new_pos, adv_pos)  # Prefer moves that block opponent

        # Evaluate barrier direction
        move_score += self.evaluate_barrier_direction(chess_board, new_pos, barrier_dir, adv_pos)

        return move_score

    def evaluate_barrier_direction(self, chess_board, new_pos, barrier_dir, adv_pos):
        """
        Evaluate the strategic value of placing a barrier in a given direction.
        """
        barrier_score = 0
        r, c = new_pos

        # check if the barrier blocks a key path for the opponent
        if barrier_dir == 0 and r > 0:  # Up
            if (r - 1, c) == adv_pos:
                barrier_score += 1  # Blocking opponent's path
        elif barrier_dir == 1 and c < self.board_size - 1:  # Right
            if (r, c + 1) == adv_pos:
                barrier_score += 1
        elif barrier_dir == 2 and r < self.board_size - 1:  # Down
            if (r + 1, c) == adv_pos:
                barrier_score += 1
        elif barrier_dir == 3 and c > 0:  # Left
            if (r, c - 1) == adv_pos:
                barrier_score += 1


        return barrier_score

    def distance(self, pos1, pos2):
        # Calculate Manhattan distance between two positions
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    class AgentTimeoutException(Exception):
        def __init__(self, score=None, best_move=None):
                self.best_move = best_move
                self.score = score
                message = "Timeout occurred."
                if best_move is not None and score is not None:
                    message += f" Best move: {best_move}, Score: {score}"
                super().__init__(message)
    pass
