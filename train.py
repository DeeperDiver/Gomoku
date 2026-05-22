import os
import queue
import random
import time
import traceback
from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp

from game import Board
from mcts import MCTS
from model import PolicyValueNet


class Config:
    # Board
    board_width = 11
    board_height = 11

    # MCTS/search
    n_playout_opening = 160
    n_playout = 480
    playout_switch_move = 14
    c_puct = 5.2

    # Training
    learn_rate = 2e-4
    batch_size = 256
    buffer_size = 60000
    min_buffer_size = 4096
    epochs = 3
    target_kl = 0.02
    check_freq = 50
    updates_per_collect = 1

    # Self-play exploration
    temp_high = 0.8
    temp_low = 0.05
    temp_decay_moves = 12
    dirichlet_alpha = 0.06
    dirichlet_eps = 0.20
    noise_moves = 8
    max_game_moves = 100

    # Multiprocessing throughput
    num_workers = max(4, min(max((os.cpu_count() or 8) - 2, 4), 12))
    queue_timeout_sec = 10
    max_games_per_collect = 4
    update_worker_freq = 8
    worker_log_every = 4

    # Startup
    load_existing_model = True


def worker_process(rank, start_model_state, cfg, data_queue, weight_queue):
    try:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        policy = PolicyValueNet(
            cfg.board_width, cfg.board_height, use_gpu=False, create_optimizer=False
        )
        policy.policy_value_net.load_state_dict(start_model_state)
        policy.policy_value_net.eval()

        mcts = MCTS(
            policy.policy_value_fn,
            cfg.c_puct,
            cfg.n_playout_opening,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_eps=cfg.dirichlet_eps,
        )

        board = Board(size=cfg.board_width)
        game_count = 0
        print(
            f"Worker {rank} started | playout:{cfg.n_playout_opening}->{cfg.n_playout} "
            f"workers:{cfg.num_workers}"
        )

        while True:
            # Keep only the latest weights to avoid applying stale checkpoints.
            new_state_dict = None
            try:
                while True:
                    new_state_dict = weight_queue.get_nowait()
            except queue.Empty:
                pass
            if new_state_dict is not None:
                policy.policy_value_net.load_state_dict(new_state_dict)
                policy.policy_value_net.eval()

            board.__init__(cfg.board_width)
            states, mcts_probs, current_players = [], [], []
            mcts.update_with_move(-1)
            move_count = 0
            start_ts = time.time()

            while True:
                mcts.n_playout = (
                    cfg.n_playout_opening
                    if move_count < cfg.playout_switch_move
                    else cfg.n_playout
                )
                temp = cfg.temp_high if move_count < cfg.temp_decay_moves else cfg.temp_low
                add_noise = move_count < cfg.noise_moves
                move, move_probs = mcts.get_move_probs(board, temp=temp, add_noise=add_noise)

                if move == -1:
                    # Safety fallback to prevent worker deadlock on unexpected search failures.
                    move = random.choice(board.availables)
                    move_probs = np.zeros(cfg.board_width * cfg.board_height, dtype=np.float32)
                    move_probs[move] = 1.0

                states.append(policy.current_state2feature(board))
                mcts_probs.append(move_probs)
                current_players.append(board.current_player)

                board.do_move(move)
                mcts.update_with_move(move)
                move_count += 1

                end, winner = board.game_end()
                if not end and move_count >= cfg.max_game_moves:
                    end, winner = True, -1

                if end:
                    winners_z = np.zeros(len(current_players), dtype=np.float32)
                    if winner != -1:
                        winners_z[np.array(current_players) == winner] = 1.0
                        winners_z[np.array(current_players) != winner] = -1.0

                    data_queue.put(list(zip(states, mcts_probs, winners_z)))
                    game_count += 1
                    if game_count % cfg.worker_log_every == 0:
                        print(
                            f"Worker {rank} games:{game_count} "
                            f"len:{move_count} time:{time.time() - start_ts:.1f}s"
                        )
                    break

    except Exception as exc:
        print(f"Worker {rank} Error: {exc}\n{traceback.format_exc()}")


class TrainPipeline:
    def __init__(self, init_model=None):
        self.cfg = Config()
        self.data_buffer = deque(maxlen=self.cfg.buffer_size)
        self.lr_multiplier = 1.0
        self.policy_value_net = PolicyValueNet(
            self.cfg.board_width,
            self.cfg.board_height,
            model_file=init_model,
            use_gpu=True,
            create_optimizer=True,
        )
        self.device = self.policy_value_net.device

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            prob_matrix = mcts_prob.reshape(self.cfg.board_width, self.cfg.board_height)
            for i in (1, 2, 3, 4):
                equi_state = np.array([np.rot90(s, i) for s in state], dtype=np.float32)
                equi_prob = np.rot90(prob_matrix, i)
                extend_data.append((equi_state, equi_prob.flatten(), winner))

                equi_state = np.array([np.fliplr(s) for s in equi_state], dtype=np.float32)
                equi_prob = np.fliplr(equi_prob)
                extend_data.append((equi_state, equi_prob.flatten(), winner))
        return extend_data

    def policy_update(self):
        if len(self.data_buffer) < self.cfg.min_buffer_size:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        self.policy_value_net.policy_value_net.train()
        minibatch = random.sample(self.data_buffer, self.cfg.batch_size)

        state_batch = torch.tensor(
            np.array([d[0] for d in minibatch], dtype=np.float32), device=self.device
        )
        mcts_probs_batch = torch.tensor(
            np.array([d[1] for d in minibatch], dtype=np.float32), device=self.device
        )
        winner_batch = torch.tensor(
            np.array([d[2] for d in minibatch], dtype=np.float32), device=self.device
        )

        with torch.no_grad():
            old_log_probs, _ = self.policy_value_net.policy_value_net(state_batch)
            old_probs = torch.exp(old_log_probs).detach().cpu().numpy()

        losses, entropies, policy_losses, value_losses = [], [], [], []
        kl = 0.0
        for _ in range(self.cfg.epochs):
            loss, entropy, p_loss, v_loss = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.cfg.learn_rate * self.lr_multiplier,
            )
            losses.append(loss)
            entropies.append(entropy)
            policy_losses.append(p_loss)
            value_losses.append(v_loss)

            with torch.no_grad():
                new_log_probs, _ = self.policy_value_net.policy_value_net(state_batch)
                new_probs = torch.exp(new_log_probs).detach().cpu().numpy()
            kl = np.mean(
                np.sum(
                    old_probs
                    * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            if kl > self.cfg.target_kl * 4:
                break

        # Keep LR control conservative to reduce late-stage oscillation.
        if kl > self.cfg.target_kl * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5

        return (
            float(np.mean(losses)),
            float(np.mean(entropies)),
            float(kl),
            float(np.mean(policy_losses)),
            float(np.mean(value_losses)),
        )

    def run(self):
        ctx = mp.get_context("spawn")
        data_queue = ctx.Queue(maxsize=self.cfg.num_workers * 4)
        weight_queues = [ctx.Queue(maxsize=2) for _ in range(self.cfg.num_workers)]

        print(f"Starting {self.cfg.num_workers} self-play workers...")
        workers = []
        init_state = {
            k: v.cpu() for k, v in self.policy_value_net.policy_value_net.state_dict().items()
        }
        for i in range(self.cfg.num_workers):
            w = ctx.Process(
                target=worker_process,
                args=(i, init_state, self.cfg, data_queue, weight_queues[i]),
            )
            w.start()
            workers.append(w)

        if torch.cuda.is_available():
            print(f"Trainer device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print("Trainer device: CPU (CUDA not available)")

        games_count = 0
        train_steps = 0
        try:
            while True:
                try:
                    play_data = data_queue.get(timeout=self.cfg.queue_timeout_sec)
                except queue.Empty:
                    alive = sum(1 for w in workers if w.is_alive())
                    print(
                        f"Waiting for self-play data... "
                        f"alive_workers:{alive}/{len(workers)} "
                        f"games:{games_count} buffer:{len(self.data_buffer)}"
                    )
                    continue

                games = [play_data]
                while len(games) < self.cfg.max_games_per_collect:
                    try:
                        games.append(data_queue.get_nowait())
                    except queue.Empty:
                        break

                for g in games:
                    self.data_buffer.extend(self.get_equi_data(g))
                games_count += len(games)

                if len(self.data_buffer) < self.cfg.min_buffer_size:
                    continue

                for _ in range(self.cfg.updates_per_collect):
                    loss, entropy, kl, p_loss, v_loss = self.policy_update()
                    train_steps += 1
                    print(
                        f"Batch {train_steps} | Games {games_count} | "
                        f"Loss:{loss:.4f} Policy:{p_loss:.4f} "
                        f"Value:{v_loss:.4f} Entropy:{entropy:.4f} KL:{kl:.4f} "
                        f"LRx:{self.lr_multiplier:.3f}"
                    )

                    if train_steps % self.cfg.update_worker_freq == 0:
                        new_state = {
                            k: v.cpu()
                            for k, v in self.policy_value_net.policy_value_net.state_dict().items()
                        }
                        for q in weight_queues:
                            try:
                                while True:
                                    q.get_nowait()
                            except queue.Empty:
                                pass
                            q.put(new_state)

                    if train_steps % self.cfg.check_freq == 0:
                        self.policy_value_net.save_model("current_policy.model")
                        print("Model saved")
        except KeyboardInterrupt:
            print("Stopping...")
            for w in workers:
                w.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    cfg = Config()
    init_model = (
        "current_policy.model"
        if cfg.load_existing_model and os.path.exists("current_policy.model")
        else None
    )
    if init_model:
        print("Loading existing model...")

    pipeline = TrainPipeline(init_model)
    pipeline.run()
