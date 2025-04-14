# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math

import sys


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

class NTupleNetwork:
    def __init__(self, board_size=4, n_tuples=4, tuple_length=4):
        self.board_size = board_size
        self.n_tuples = n_tuples
        self.tuple_length = tuple_length
        self.weights = {}
        self.tuples = self._generate_tuples()
        
    def _generate_tuples(self):
        """Generate n-tuple patterns"""
        tuples = []
        # Main diagonal tuple
        tuples.append([(i, i) for i in range(self.board_size)])
        # Anti-diagonal tuple
        tuples.append([(i, self.board_size-1-i) for i in range(self.board_size)])
        # First row and column
        tuples.append([(0, i) for i in range(self.board_size)])
        tuples.append([(i, 0) for i in range(self.board_size)])
        return tuples[:self.n_tuples]
    
    def get_tuple_value(self, board, tuple_indices):
        """Get the value of a specific tuple"""
        key = tuple(board[i][j] for i, j in tuple_indices)
        if key not in self.weights:
            self.weights[key] = np.random.random() * 0.1
        return self.weights[key]
    
    def evaluate(self, board):
        """Evaluate the entire board"""
        return sum(self.get_tuple_value(board, tuple_indices) 
                  for tuple_indices in self.tuples)
    
    def update(self, board, target, learning_rate=0.1):
        """Update weights based on TD error"""
        current = self.evaluate(board)
        error = target - current
        
        for tuple_indices in self.tuples:
            key = tuple(board[i][j] for i, j in tuple_indices)
            self.weights[key] += learning_rate * error

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state.copy()
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = self._get_legal_actions()
        
    def _get_legal_actions(self):
        """Get all legal actions for the current state"""
        legal = []
        for action in range(4):
            if is_move_legal(self.state, action):
                legal.append(action)
        return legal
    
    def select_child(self, c_param=1.4):
        """Select child node using UCB1"""
        choices = [(child,
                   child.value / (child.visits + 1e-6) +
                   c_param * np.sqrt(2 * np.log(self.visits + 1) / (child.visits + 1e-6)))
                  for action, child in self.children.items()]
        
        _, child = max(choices, key=lambda x: x[1])
        return child
    
    def expand(self):
        """Expand by trying an untried action"""
        action = self.untried_actions.pop()
        next_state = simulate_move(self.state.copy(), action)
        child = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child
        return child
    
    def update(self, result):
        """Update node statistics"""
        self.visits += 1
        self.value += result

class MCTS:
    def __init__(self, root_state, ntuple_network, num_simulations=100):
        self.root = MCTSNode(root_state)
        self.ntuple_network = ntuple_network
        self.num_simulations = num_simulations
    
    def search(self):
        """Perform MCTS search"""
        for _ in range(self.num_simulations):
            node = self.root
            
            # Selection
            while not node.untried_actions and node.children:
                node = node.select_child()
            
            # Expansion
            if node.untried_actions:
                node = node.expand()
            
            # Simulation
            value = self.simulate(node.state)
            
            # Backpropagation
            while node:
                node.update(value)
                node = node.parent
        
        # Return best action
        return max(self.root.children.items(),
                  key=lambda x: x[1].visits)[0]
    
    def simulate(self, state):
        """Run a simulation from the given state"""
        return self.ntuple_network.evaluate(state)

def get_action(state, score):
    """
    Input: The current state (4x4 numpy array) and score
    Output: The action to take (0: up, 1: down, 2: left, 3: right)
    """
    # Initialize the n-tuple network if not exists
    global ntuple_network
    if 'ntuple_network' not in globals():
        ntuple_network = NTupleNetwork()
    
    # Use MCTS to select the best action
    mcts = MCTS(state, ntuple_network, num_simulations=50)
    action = mcts.search()
    
    # Update n-tuple network with new experience
    next_state = simulate_move(state.copy(), action)
    target = max(score, ntuple_network.evaluate(next_state))
    ntuple_network.update(state, target)
    
    return action

def simulate_move(board, action):
    """Simulate a move without modifying the original board"""
    env = Game2048Env()
    env.board = board.copy()
    next_state, _, _, _ = env.step(action)
    return next_state


