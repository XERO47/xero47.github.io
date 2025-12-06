
import os
from typing import List, Dict, Tuple, Optional

import tkinter as tk
from tkinter import messagebox, simpledialog

import torch
import torch.nn as nn
import random

PLAYER_X = 1
PLAYER_O = -1
EMPTY = 0

BOARD_SIZE = 3
MAX_MARKS_PER_PLAYER = 3

INPUT_DIM = 36
NUM_ACTIONS = 9
MODEL_PATH = "trailing_ttt_nn.pt"


# ==========================
# GameState (must match generator)
# ==========================

class GameState:
    def __init__(self,
                 board: Optional[List[int]] = None,
                 current_player: int = PLAYER_X,
                 positions: Optional[Dict[int, List[int]]] = None,
                 step_count: int = 0):
        if board is None:
            board = [EMPTY] * (BOARD_SIZE * BOARD_SIZE)
        if positions is None:
            positions = {PLAYER_X: [], PLAYER_O: []}

        self.board: List[int] = list(board)
        self.current_player: int = current_player
        self.positions: Dict[int, List[int]] = {
            PLAYER_X: list(positions[PLAYER_X]),
            PLAYER_O: list(positions[PLAYER_O]),
        }
        self.step_count: int = step_count

    def clone(self) -> "GameState":
        return GameState(
            board=self.board,
            current_player=self.current_player,
            positions=self.positions,
            step_count=self.step_count,
        )

    def available_actions(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v == EMPTY]

    def _apply_trailing_rule(self, player: int, action: int) -> None:
        self.board[action] = player
        self.positions[player].append(action)

        if len(self.positions[player]) > MAX_MARKS_PER_PLAYER:
            oldest = self.positions[player].pop(0)
            if self.board[oldest] == player:
                self.board[oldest] = EMPTY

    def _check_winner(self) -> Optional[int]:
        lines = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]
        for a, b, c in lines:
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def is_terminal(self) -> Tuple[bool, Optional[int]]:
        winner = self._check_winner()
        if winner is not None:
            return True, winner
        if not self.available_actions():
            return True, None
        return False, None

    def step(self, action: int) -> "GameState":
        new_state = self.clone()
        if action not in new_state.available_actions():
            raise ValueError("Illegal move in GameState.step")
        new_state._apply_trailing_rule(new_state.current_player, action)
        new_state.step_count += 1
        new_state.current_player *= -1
        return new_state


def encode_state(state: GameState) -> List[float]:
    """
    Same encoding as training:
      for each cell: [my_mark, my_age, opp_mark, opp_age]
    """
    inputs: List[float] = []
    current = state.current_player
    opp = -current

    my_positions = state.positions[current]
    opp_positions = state.positions[opp]

    for idx in range(BOARD_SIZE * BOARD_SIZE):
        if idx in my_positions:
            my_mark = 1.0
            age_index = my_positions.index(idx)
            my_age = (age_index + 1) / 3.0
        else:
            my_mark = 0.0
            my_age = 0.0

        if idx in opp_positions:
            opp_mark = 1.0
            age_index = opp_positions.index(idx)
            opp_age = (age_index + 1) / 3.0
        else:
            opp_mark = 0.0
            opp_age = 0.0

        inputs.extend([my_mark, my_age, opp_mark, opp_age])

    return inputs


# ==========================
# CNN Model (same as in training)
# ==========================

class TTTNet(nn.Module):
    def __init__(self, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc = nn.Linear(64 * 3 * 3, 128)
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # x: (batch, 36) -> (batch, 4, 3, 3)
        batch_size = x.size(0)
        x = x.reshape(batch_size, 9, 4)
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, 4, 3, 3)

        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = h.reshape(batch_size, -1)
        h = self.relu(self.fc(h))

        logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return logits, value.squeeze(-1)


def load_model(path: str, device: str = "cpu") -> TTTNet:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{path}' not found. "
            f"Run 'train_perfect_nn.py' first to create it."
        )
    model = TTTNet()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def nn_choose_action(model: TTTNet,
                     state: GameState,
                     device: str = "cpu") -> int:
    """
    Use NN to pick the best move (argmax over legal moves).
    """
    x_list = encode_state(state)
    x = torch.tensor(x_list, dtype=torch.float32, device=device).unsqueeze(0)   # (1, 36)

    with torch.no_grad():
        logits, _ = model(x)   # (1, 9)

    logits = logits[0].cpu().numpy().tolist()
    available = state.available_actions()

    for i in range(BOARD_SIZE * BOARD_SIZE):
        if i not in available:
            logits[i] = -1e9

    best_action = max(range(len(logits)), key=lambda i: logits[i])
    return best_action


# ==========================
# Tkinter GUI
# ==========================

class PerfectNNGUI:
    def __init__(self, root: tk.Tk, model: TTTNet, device: str = "cpu"):
        self.root = root
        self.root.title("Trailing Tic-Tac-Toe (CNN NN)")

        self.model = model
        self.device = device

        self.state = GameState()
        self.game_over = False

        self.human_player: int = PLAYER_X
        self.agent_player: int = PLAYER_O

        self.buttons: List[tk.Button] = []
        self.status_label: Optional[tk.Label] = None

        self._create_widgets()
        self._ask_player_side()
        self._update_status()
        self._render_board()

        self.root.after(300, self._maybe_agent_first_move)

    def _create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                idx = r * BOARD_SIZE + c
                btn = tk.Button(
                    frame,
                    text="",
                    width=4,
                    height=2,
                    font=("Helvetica", 24),
                    command=lambda i=idx: self.on_cell_click(i),
                )
                btn.grid(row=r, column=c, padx=5, pady=5)
                self.buttons.append(btn)

        self.status_label = tk.Label(self.root, text="", font=("Helvetica", 12))
        self.status_label.pack(pady=(5, 5))

        new_game_btn = tk.Button(
            self.root,
            text="New Game",
            font=("Helvetica", 10),
            command=self.start_new_game,
        )
        new_game_btn.pack(pady=(0, 10))

    def _ask_player_side(self):
        while True:
            side = simpledialog.askstring(
                "Choose Side",
                "Do you want to be X (first) or O (second)?",
                parent=self.root,
            )
            if side is None:
                self.human_player = PLAYER_X
                self.agent_player = PLAYER_O
                break

            side = side.strip().upper()
            if side == "X":
                self.human_player = PLAYER_X
                self.agent_player = PLAYER_O
                break
            elif side == "O":
                self.human_player = PLAYER_O
                self.agent_player = PLAYER_X
                # Place a single random starting X for the agent,
                # using the same trailing-rule application as normal moves.
                # Ensure it only happens on a fresh board.
                available = self.state.available_actions()
                if available:
                    choice = random.choice(available)
                    # use internal helper to apply trailing rule directly
                    self.state._apply_trailing_rule(self.agent_player, choice)
                    self.state.step_count += 1
                    # make sure human goes next
                    self.state.current_player = self.human_player
                break
            else:
                messagebox.showinfo("Invalid choice", "Please enter X or O.")

    def start_new_game(self):
        self.state = GameState()
        self.game_over = False
        # If human chose O (agent is X), place a random starting X for the agent
        if self.human_player == PLAYER_O and self.agent_player == PLAYER_X:
            available = self.state.available_actions()
            if available:
                choice = random.choice(available)
                self.state._apply_trailing_rule(self.agent_player, choice)
                self.state.step_count += 1
                # ensure human (O) moves next
                self.state.current_player = self.human_player
        self._render_board()
        self._update_status()
        self.root.after(300, self._maybe_agent_first_move)

    def _maybe_agent_first_move(self):
        if not self.game_over and self.state.current_player == self.agent_player:
            self.root.after(200, self._agent_move)

    def _symbol_for_cell(self, value: int) -> str:
        if value == PLAYER_X:
            return "X"
        elif value == PLAYER_O:
            return "O"
        else:
            return ""

    def _render_board(self):
        for i, btn in enumerate(self.buttons):
            symbol = self._symbol_for_cell(self.state.board[i])
            btn.config(text=symbol)
            if self.game_over:
                btn.config(state=tk.DISABLED)
            else:
                btn.config(state=tk.NORMAL)

    def _update_status(self, extra: str = ""):
        if self.game_over:
            status = "Game over."
        else:
            turn = "X" if self.state.current_player == PLAYER_X else "O"
            status = f"Turn: {turn}"
        if extra:
            status += " | " + extra
        if self.status_label is not None:
            self.status_label.config(text=status)

    def on_cell_click(self, index: int):
        if self.game_over:
            return
        if self.state.current_player != self.human_player:
            return

        if index not in self.state.available_actions():
            messagebox.showinfo("Invalid move", "That cell is not available.")
            return

        self.state = self.state.step(index)
        self._render_board()

        is_term, winner = self.state.is_terminal()
        if is_term:
            self.game_over = True
            self._handle_game_end(winner)
            return

        self._update_status()
        self.root.after(300, self._agent_move)

    def _agent_move(self):
        if self.game_over:
            return
        if self.state.current_player != self.agent_player:
            return

        action = nn_choose_action(self.model, self.state, device=self.device)
        self.state = self.state.step(action)
        self._render_board()

        is_term, winner = self.state.is_terminal()
        if is_term:
            self.game_over = True
            self._handle_game_end(winner)
        else:
            self._update_status()

    def _handle_game_end(self, winner: Optional[int]):
        if winner is None:
            msg = "It's a draw."
        else:
            if winner == self.human_player:
                msg = "You win! 🎉"
            elif winner == self.agent_player:
                msg = "Agent wins."
            else:
                msg = "Game over."

        self._update_status(msg)
        messagebox.showinfo("Game Over", msg)


# ==========================
# Main
# ==========================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_PATH, device=device)

    root = tk.Tk()
    app = PerfectNNGUI(root, model, device=device)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
