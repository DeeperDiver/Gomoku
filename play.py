import torch
import numpy as np
from game import Board
from model import PolicyValueNet
from mcts import MCTS
import os

class HumanPlayer:
    def __init__(self):
        self.player = None
    
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            # 提示用户输入
            location = input("请输入坐标 (格式: 行,列 例如 2,3): ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]
            
            # 将坐标转换为数字索引
            move = board.location_to_move(location)
        except Exception:
            move = -1
        
        if move == -1 or move not in board.availables:
            print("输入无效或位置已被占用，请重试！")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human"

def run():
    # -----------------------------------------
    # 参数配置
    # -----------------------------------------
    n = 5
    width, height = 11, 11
    model_file = 'current_policy.model'  # 确保这个文件存在
    
    # 这里的 n_playout 决定了 AI 思考的时间和深度
    # 训练时我们用 400，对战时可以设高一点让它更强（例如 800 或 1000）
    # 如果为了响应快，可以设为 400
    ai_n_playout = 800 
    
    # -----------------------------------------
    # 初始化
    # -----------------------------------------
    try:
        board = Board(size=width)
        board.init_board() # 或者是 board.__init__(width) 取决于你的 game.py 写法
    except:
        board = Board(size=width)

    # 1. 加载模型
    if not os.path.exists(model_file):
        print(f"错误：找不到模型文件 {model_file}")
        return

    print("正在加载模型...")
    # use_gpu=True 如果你想让 AI 思考快一点；如果没有显卡报错，改成 False
    best_policy = PolicyValueNet(width, height, model_file=model_file, use_gpu=True)
    
    # 2. 创建 AI 玩家 (基于 MCTS)
    # 竞技模式 temp=1e-3 (几乎只选胜率最高的步，不乱走)
    mcts_player = MCTS(best_policy.policy_value_fn, C_puct=5, n_playout=ai_n_playout)
    
    # 3. 创建人类玩家
    human = HumanPlayer()
    
    # -----------------------------------------
    # 游戏开始
    # -----------------------------------------
    # 设置先手：start_player = 0 (人类先手), 1 (AI先手)
    # 注意：在你的 game.py 中，玩家1是黑棋(先手)，玩家2是白棋(后手)
    print("\n--- 游戏开始 ---")
    print("输入 '0' : 你先手 (黑棋)")
    print("输入 '1' : AI先手 (黑棋)")
    choice = input("请选择: ")
    
    if choice == '0':
        print("\n你执黑棋 (●)，AI执白棋 (○)")
        players = {1: human, 2: mcts_player}
    else:
        print("\nAI执黑棋 (●)，你执白棋 (○)")
        players = {1: mcts_player, 2: human}

    board.display()
    
    while True:
        p = board.current_player
        player_in_turn = players[p]
        
        # 显示思考提示
        if isinstance(player_in_turn, MCTS):
            print(f"\nAI (Player {p}) 正在思考...", end="", flush=True)
            # 获取 AI 落子
            move, _ = player_in_turn.get_move_probs(board, temp=1e-3)
            print(" 完成!")
        else:
            print(f"\n轮到你了 (Player {p})")
            move = player_in_turn.get_action(board)
        
        # 执行落子
        board.do_move(move)
        
        # 别忘了更新 MCTS 树（这就用到了你之前修好的 update_with_move）
        if isinstance(players[1], MCTS): players[1].update_with_move(move)
        if isinstance(players[2], MCTS): players[2].update_with_move(move)
        
        board.display()
        
        # 判断胜负
        end, winner = board.game_end()
        if end:
            if winner != -1:
                print(f"\n游戏结束! 胜者是: {'Black (●)' if winner == 1 else 'White (○)'}")
            else:
                print("\n游戏结束! 平局!")
            break

if __name__ == '__main__':
    run()
