import numpy as np

class Board:
    def __init__(self, size=11):
        self.size = size
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.last_move = -1
        self.availables = list(range(self.size * self.size))  # 可用位置列表

    def move_to_location(self, move):
        h = move // self.size
        w = move % self.size
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.size + w
        
        if move < 0 or move >= self.size * self.size:
            return -1
        return move
    
    def judge(self):
        if self.last_move == -1:
            return False, -1
        
        row, col = self.move_to_location(self.last_move)
        player = 2 if self.current_player == 1 else 1

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1 
            
            for k in range(1, 5):
                r, c = row + k * dr, col + k * dc
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                    count += 1
                else:
                    break
            
            for k in range(1, 5):
                r, c = row - k * dr, col - k * dc
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                    count += 1
                else:
                    break
        
            if count >= 5:
                return True, player
        
        return False, -1

    def game_end(self):
        win, winner = self.judge()
        if win:
            return True, winner
        elif not self.availables:
            return True, -1
        return False, -1
    
    def display(self):
        print("\n  " + "".join([f"{i:<3}" for i in range(self.size)]))
        
        last_row, last_col = -1, -1
        if self.last_move != -1:
            last_row, last_col = self.move_to_location(self.last_move)

        for r in range(self.size):
            line = f"{r:<2}"
            for c in range(self.size):
                p = self.board[r][c]
                char = ""
                
                is_last = (r == last_row and c == last_col)
                color_start = "\033[91m" if is_last else "" 
                color_end = "\033[0m" if is_last else ""

                if p == 1:
                    char = f"{color_start}●{color_end}"
                elif p == 2:
                    char = f"{color_start}○{color_end}"
                else:
                    if r == 0 and c == 0: char = "┌"
                    elif r == 0 and c == self.size - 1: char = "┐"
                    elif r == self.size - 1 and c == 0: char = "└"
                    elif r == self.size - 1 and c == self.size - 1: char = "┘"
                    elif r == 0: char = "┬"
                    elif r == self.size - 1: char = "┴"
                    elif c == 0: char = "├"
                    elif c == self.size - 1: char = "┤"
                    else: char = "┼"
                
                line += char + "──"
            
            print(line[:-2]) 
        print() 
    
    def do_move(self, action):
        move_loc = self.move_to_location(action)
        row, col = move_loc
        self.board[row][col] = self.current_player
        self.last_move = action
        self.availables.remove(action)
        self.current_player = 2 if self.current_player == 1 else 1

    def copy(self):
        new_board = Board(self.size)
        new_board.board = self.board.copy()  # Numpy 的 copy 很快
        new_board.current_player = self.current_player
        new_board.last_move = self.last_move
        new_board.availables = list(self.availables) 
        return new_board

if __name__ == "__main__":
    game = Board(size=11)
    while True:
        game.display()
        p_name = "Black (●)" if game.current_player == 1 else "White (○)"
        print(f"Current Turn: {p_name}")

        try:
            point = input("Enter: row,column (Enter 'e' to Exit)")
            if point == 'e':
                break

            if ',' not in point:
                 print("Invalid format! Use row,col")
                 continue
                 
            x, y = map(int, point.split(','))
            action = game.location_to_move([x, y])

            if action == -1 or action not in game.availables:
                print("\nInvalid move!")
                continue

            game.do_move(action)
            win, winner = game.judge()

            if win:
                game.display()
                winner_name = "Black" if winner == 1 else "White"
                print(f"\nWinner is {winner_name}!")
                break
            if not game.availables:
                print("\nGame over: Tie!")
                break

        except ValueError:
            print("Invalid input! Please enter numbers.")
        except Exception as e:
            print(f"Error: {e}")
