import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, num_filters: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, board_width: int, board_height: int):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.num_filters = 96

        # Shared trunk
        self.conv1 = nn.Conv2d(4, self.num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_filters)
        self.res_blocks = nn.ModuleList([ResBlock(self.num_filters) for _ in range(8)])

        # Policy head
        self.act_conv1 = nn.Conv2d(self.num_filters, 8, kernel_size=1)
        self.act_bn1 = nn.BatchNorm2d(8)
        self.act_fc1 = nn.Linear(8 * board_width * board_height, board_width * board_height)

        # Value head
        self.val_conv1 = nn.Conv2d(self.num_filters, 4, kernel_size=1)
        self.val_bn1 = nn.BatchNorm2d(4)
        self.val_fc1 = nn.Linear(4 * board_width * board_height, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):
        x = F.relu(self.bn1(self.conv1(state_input)))
        for block in self.res_blocks:
            x = block(x)

        x_act = F.relu(self.act_bn1(self.act_conv1(x)))
        x_act = x_act.view(-1, 8 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        x_val = F.relu(self.val_bn1(self.val_conv1(x)))
        x_val = x_val.view(-1, 4 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet:
    def __init__(
        self,
        board_width: int,
        board_height: int,
        model_file: str = None,
        use_gpu: bool = True,
        create_optimizer: bool = True,
    ):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4

        self.policy_value_net = Net(board_width, board_height).to(self.device)
        if model_file:
            try:
                self.policy_value_net.load_state_dict(
                    torch.load(model_file, map_location=self.device)
                )
                print(f"Loaded model: {model_file}")
            except Exception as exc:
                print(f"Failed to load model ({exc}); using random initialization.")

        self.optimizer = None
        if create_optimizer:
            self.optimizer = torch.optim.Adam(
                self.policy_value_net.parameters(), weight_decay=self.l2_const
            )

    def policy_value_fn(self, board):
        legal_positions = board.availables
        current_state = np.ascontiguousarray(
            self.current_state2feature(board).reshape(
                -1, 4, self.board_width, self.board_height
            )
        )

        with torch.no_grad():
            input_tensor = torch.from_numpy(current_state).float().to(self.device)
            log_act_probs, value = self.policy_value_net(input_tensor)
            act_probs = np.exp(log_act_probs.detach().cpu().numpy().flatten())

        legal_probs = act_probs[legal_positions]
        probs_sum = legal_probs.sum()
        if probs_sum > 1e-10:
            legal_probs /= probs_sum
        else:
            legal_probs = np.ones_like(legal_probs) / len(legal_probs)

        return zip(legal_positions, legal_probs), value.item()

    def __call__(self, state):
        return self.policy_value_fn(state)

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not initialized for this PolicyValueNet instance.")

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.optimizer.zero_grad()
        log_act_probs, value = self.policy_value_net(state_batch)

        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item(), policy_loss.item(), value_loss.item()

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save_model(self, model_file):
        torch.save(self.get_policy_param(), model_file)

    def current_state2feature(self, board):
        square_state = np.zeros((4, self.board_width, self.board_height), dtype=np.float32)
        square_state[0] = (board.board == board.current_player).astype(np.float32)

        opponent = 2 if board.current_player == 1 else 1
        square_state[1] = (board.board == opponent).astype(np.float32)

        if board.last_move != -1:
            row, col = board.move_to_location(board.last_move)
            square_state[2][row][col] = 1.0

        if board.current_player == 1:
            square_state[3][:, :] = 1.0

        return square_state
