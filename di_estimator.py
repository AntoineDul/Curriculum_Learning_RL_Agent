import math
import numpy as np
from collections import Counter
from ctw import *

class DIEstimator:
    """
    CTW-based Directed Information Estimator for a univariate sequence of actions.
    Predicts a_t from (a_{t-1}, ..., a_{t-k}).
    """

    def __init__(self, num_actions=4, k=3):
        self.num_actions = num_actions
        self.k = k

        # Replace manual counts with CTW tree
        self.ctw = CTW(alphabet_size_y=num_actions, depth=k, side_info=False)

        # Keep your original book-keeping
        self.total = 0
        self.total_actions = 0
        self.total_sequences = 0

        # For optional monitoring
        self.entropies = []
        self.sequence_count = 0

    def update(self, actions):
        """
        Update the CTW tree with a new action sequence.
        Converts context and symbol to string format for CTW compatibility.
        """

        self.total_actions += len(actions)
        self.total_sequences += 1

        for t in range(self.k, len(actions)):
            # Get context (a_{t-1}, ..., a_{t-k}) in correct order
            context = [str(actions[t - j - 1]) for j in range(self.k)]
            context = context[::-1]  # oldest first

            # Convert symbol to string
            symbol = str(actions[t])

            # Update CTW tree
            _ = self.ctw.update_tree(context, symbol)
            self.total += 1

    def get_entropy(self):
        """
        Compute conditional entropy per symbol using the CTW tree's root.ws.
        This fully leverages CTW's smoothing across all context depths.
        """
        # Total number of symbols that have been used for prediction
        num_symbols = self.total_actions - self.k
        if num_symbols <= 0:
            return 0.0

        # Root.ws is log2 probability of the whole sequence (blends all depths)
        H_per_symbol = -self.ctw.tree.root.ws / num_symbols
        return H_per_symbol


    def get_average_actions(self):
        """
        Running average number of actions per sequence update.
        """
        if self.total_sequences == 0:
            return 0.0
        avg = self.total_actions / self.total_sequences
        return avg, self.total_actions, self.total_sequences
