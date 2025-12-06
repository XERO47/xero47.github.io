import pickle
from collections import deque, defaultdict

PLAYER_X = 1
PLAYER_O = -1
EMPTY = 0

BOARD_SIZE = 3
MAX_MARKS = 3

# Solver values
WIN = 1
LOSS = -1
DRAW = 0


class GameState:
    def __init__(self, board=None, current_player=PLAYER_X, positions=None, step_count=0):
        self.board = board if board else [EMPTY] * 9
        self.current_player = current_player
        self.positions = positions if positions else {PLAYER_X: [], PLAYER_O: []}
        self.step_count = step_count

    def clone(self):
        return GameState(
            board=list(self.board),
            current_player=self.current_player,
            positions={k: list(v) for k, v in self.positions.items()},
            step_count=self.step_count
        )

    def get_available_moves(self):
        return [i for i, cell in enumerate(self.board) if cell == EMPTY]

    def place_mark(self, player, pos):
        self.board[pos] = player
        self.positions[player].append(pos)
        
        # Remove oldest mark if we exceed the limit
        if len(self.positions[player]) > MAX_MARKS:
            old_pos = self.positions[player].pop(0)
            if self.board[old_pos] == player:
                self.board[old_pos] = EMPTY

    def check_winner(self):
        # Check all winning lines
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        
        for a, b, c in lines:
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def is_game_over(self):
        winner = self.check_winner()
        if winner:
            return True, winner
        
        if not self.get_available_moves():
            return True, None
            
        return False, None

    def make_move(self, pos):
        new_state = self.clone()
        if pos not in new_state.get_available_moves():
            raise ValueError(f"Invalid move: {pos}")
        
        new_state.place_mark(new_state.current_player, pos)
        new_state.step_count += 1
        new_state.current_player *= -1
        return new_state

    def to_key(self):
        return (
            tuple(self.board),
            self.current_player,
            tuple(self.positions[PLAYER_X]),
            tuple(self.positions[PLAYER_O])
        )


def build_game_tree():
    """Build complete state graph using BFS"""
    start = GameState()
    
    state_map = {}
    all_states = []
    next_states = []
    next_actions = []
    prev_states = defaultdict(list)
    
    queue = deque([start])
    state_map[start.to_key()] = 0
    all_states.append(start)
    next_states.append([])
    next_actions.append([])
    
    while queue:
        state = queue.popleft()
        idx = state_map[state.to_key()]
        
        done, _ = state.is_game_over()
        if done:
            continue
        
        for move in state.get_available_moves():
            next_state = state.make_move(move)
            key = next_state.to_key()
            
            if key not in state_map:
                next_idx = len(all_states)
                state_map[key] = next_idx
                all_states.append(next_state)
                next_states.append([])
                next_actions.append([])
                queue.append(next_state)
            else:
                next_idx = state_map[key]
            
            next_actions[idx].append(move)
            next_states[idx].append(next_idx)
            prev_states[next_idx].append(idx)
    
    print(f"Built game tree: {len(all_states)} states, {sum(len(s) for s in next_states)} edges")
    return all_states, next_actions, next_states, prev_states


def solve_game(states, next_actions, next_states, prev_states):
    """Retrograde analysis to solve the game"""
    n = len(states)
    values = [None] * n
    out_degree = [len(next_states[i]) for i in range(n)]
    queue = deque()
    
    # Handle terminal states first
    for i, state in enumerate(states):
        is_terminal, winner = state.is_game_over()
        if is_terminal:
            if winner is None:
                values[i] = DRAW
            else:
                values[i] = WIN if winner == state.current_player else LOSS
                queue.append(i)
    
    # Propagate values backward
    while queue:
        curr = queue.popleft()
        curr_val = values[curr]
        
        for parent in prev_states.get(curr, []):
            if values[parent] is not None:
                continue
            
            if curr_val == LOSS:
                # Parent can force opponent into losing position
                values[parent] = WIN
                queue.append(parent)
            elif curr_val == WIN:
                out_degree[parent] -= 1
                if out_degree[parent] == 0:
                    # All moves lead to opponent winning
                    values[parent] = LOSS
                    queue.append(parent)
    
    # Everything else is a draw
    for i in range(n):
        if values[i] is None:
            values[i] = DRAW
    
    print(f"Solved: {values.count(WIN)} wins, {values.count(LOSS)} losses, {values.count(DRAW)} draws")
    return values


def find_best_moves(states, next_actions, next_states, values):
    """Figure out optimal moves for each state"""
    best_moves = []
    
    for i, state in enumerate(states):
        is_terminal, _ = state.is_game_over()
        if is_terminal or not next_actions[i]:
            best_moves.append([])
            continue
        
        state_value = values[i]
        moves = next_actions[i]
        children = next_states[i]
        
        if state_value == WIN:
            # Pick moves that make opponent lose
            good_moves = [m for m, child in zip(moves, children) if values[child] == LOSS]
            if not good_moves:
                good_moves = [m for m, child in zip(moves, children) if values[child] != WIN]
            best_moves.append(good_moves if good_moves else moves)
        
        elif state_value == LOSS:
            # All moves are bad, doesn't matter
            best_moves.append(moves)
        
        else:  # DRAW
            # Don't give opponent an easy win
            ok_moves = [m for m, child in zip(moves, children) if values[child] != WIN]
            best_moves.append(ok_moves if ok_moves else moves)
    
    return best_moves


def encode_position(state):
    """Convert game state to neural network input"""
    features = []
    curr = state.current_player
    opp = -curr
    
    my_marks = state.positions[curr]
    their_marks = state.positions[opp]
    
    for i in range(9):
        # My piece features
        if i in my_marks:
            age = my_marks.index(i)
            features.extend([1.0, (age + 1) / 3.0])
        else:
            features.extend([0.0, 0.0])
        
        # Their piece features  
        if i in their_marks:
            age = their_marks.index(i)
            features.extend([1.0, (age + 1) / 3.0])
        else:
            features.extend([0.0, 0.0])
    
    return features


def create_training_data():
    """Generate the full training dataset"""
    print("Building game tree...")
    states, actions, next_st, prev_st = build_game_tree()
    
    print("Solving...")
    values = solve_game(states, actions, next_st, prev_st)
    
    print("Finding best moves...")
    optimal_moves = find_best_moves(states, actions, next_st, values)
    
    print("Creating dataset...")
    X, P, V = [], [], []
    
    for i, state in enumerate(states):
        X.append(encode_position(state))
        
        # Policy target
        policy = [0.0] * 9
        moves = optimal_moves[i]
        if moves:
            prob = 1.0 / len(moves)
            for m in moves:
                policy[m] = prob
        P.append(policy)
        
        V.append(float(values[i]))
    
    print(f"Dataset ready: {len(X)} examples")
    return X, P, V


def save_data(filename, inputs, policies, values):
    """Save everything to a pickle file"""
    data = {"inputs": inputs, "policies": policies, "values": values}
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    print("Generating training data for trailing tic-tac-toe...")
    X, P, V = create_training_data()
    save_data("trailing_ttt_dataset.pkl", X, P, V)