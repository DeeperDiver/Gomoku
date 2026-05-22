import math

import numpy as np


class TreeNode:
    def __init__(self, parent=None, prior=1.0):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.w_value = 0.0
        self.prior = prior

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda pair: pair[1].get_score(c_puct))

    def get_score(self, c_puct):
        q = self.w_value / self.n_visits if self.n_visits > 0 else 0.0
        # Child Q is in child-player perspective. Convert to parent perspective for selection.
        if self.parent is not None:
            q = -q
        u = c_puct * self.prior * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return q + u

    def update(self, leaf_value):
        self.n_visits += 1
        self.w_value += leaf_value

    def update_recursive(self, leaf_value):
        self.update(leaf_value)
        if self.parent is not None:
            self.parent.update_recursive(-leaf_value)

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(
        self,
        policy_value_fn,
        C_puct=5,
        n_playout=2000,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
    ):
        self._policy = policy_value_fn
        self.C_puct = C_puct
        self.n_playout = n_playout
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.root = TreeNode(None, 1.0)

    def _playout(self, state):
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.C_puct)
            state.do_move(action)

        end, winner = state.game_end()
        if not end:
            action_probs, leaf_value = self._policy(state)
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.current_player else -1.0

        node.update_recursive(leaf_value)

    def get_move_probs(self, state, temp=1e-3, add_noise=True):
        sensible_moves = state.availables
        if add_noise and len(sensible_moves) > 0 and temp > 1e-3:
            noise = np.random.dirichlet(
                self.dirichlet_alpha * np.ones(len(sensible_moves), dtype=np.float32)
            )

            if self.root.is_leaf():
                action_probs, _ = self._policy(state)
                self.root.expand(action_probs)

            for i, move in enumerate(sensible_moves):
                if move in self.root.children:
                    child = self.root.children[move]
                    child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * noise[i]

        for _ in range(self.n_playout):
            self._playout(state.copy())

        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        if not act_visits:
            return -1, np.zeros(state.size * state.size, dtype=np.float32)

        acts, visits = zip(*act_visits)
        visits = np.array(visits, dtype=np.float32)

        if temp <= 1e-3:
            best_idx = int(np.argmax(visits))
            act = acts[best_idx]
            probs = np.zeros(len(acts), dtype=np.float32)
            probs[best_idx] = 1.0
        else:
            # Numerically stable temperature scaling in log-space.
            # Clip visits to avoid log(0), then softmax(log(visits) / temp).
            safe_temp = max(float(temp), 1e-6)
            safe_visits = np.clip(visits, 1.0, None)
            logits = np.log(safe_visits) / safe_temp
            logits = logits - np.max(logits)
            logits = np.clip(logits, -50.0, 50.0)

            exp_logits = np.exp(logits, dtype=np.float64)
            denom = float(np.sum(exp_logits))

            if not np.isfinite(denom) or denom <= 1e-12:
                # Fallback to greedy if distribution is numerically invalid.
                best_idx = int(np.argmax(visits))
                probs = np.zeros(len(acts), dtype=np.float32)
                probs[best_idx] = 1.0
            else:
                probs = (exp_logits / denom).astype(np.float32)
                probs_sum = float(np.sum(probs))
                if not np.isfinite(probs_sum) or probs_sum <= 1e-12:
                    best_idx = int(np.argmax(visits))
                    probs = np.zeros(len(acts), dtype=np.float32)
                    probs[best_idx] = 1.0
                else:
                    probs /= probs_sum
            act = int(np.random.choice(acts, p=probs))

        full_probs = np.zeros(state.size * state.size, dtype=np.float32)
        full_probs[list(acts)] = probs
        return act, full_probs

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)
