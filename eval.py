import argparse
import torch
import numpy as np
from game import SnakeGameAI
from model import Linear_QNet


def get_action_from_model(model: Linear_QNet, state_np: np.ndarray):
    state_t = torch.tensor(state_np, dtype=torch.float)
    with torch.no_grad():
        q_values = model(state_t)
        move_idx = torch.argmax(q_values).item()
    final_move = [0, 0, 0]
    final_move[move_idx] = 1
    return final_move

def run_eval(model_path: str, width: int, height: int, max_episodes: int | None):
    model = Linear_QNet(11, 256, 3)
    model.load(model_path, map_location='cpu')

    game = SnakeGameAI(w=width, h=height)

    episodes = 0
    record = 0
    while True:
        state_old = game.snake
        head = game.snake[0]
        point_l = (head.x - 20, head.y)
        point_r = (head.x + 20, head.y)
        point_u = (head.x, head.y - 20)
        point_d = (head.x, head.y + 20)

        dir_l = game.direction.name == 'LEFT'
        dir_r = game.direction.name == 'RIGHT'
        dir_u = game.direction.name == 'UP'
        dir_d = game.direction.name == 'DOWN'

        state = [
            (dir_r and game.is_collision(game.head._replace(x=point_r[0], y=point_r[1]))) or
            (dir_l and game.is_collision(game.head._replace(x=point_l[0], y=point_l[1]))) or
            (dir_u and game.is_collision(game.head._replace(x=point_u[0], y=point_u[1]))) or
            (dir_d and game.is_collision(game.head._replace(x=point_d[0], y=point_d[1]))),

            (dir_u and game.is_collision(game.head._replace(x=point_r[0], y=point_r[1]))) or
            (dir_d and game.is_collision(game.head._replace(x=point_l[0], y=point_l[1]))) or
            (dir_l and game.is_collision(game.head._replace(x=point_u[0], y=point_u[1]))) or
            (dir_r and game.is_collision(game.head._replace(x=point_d[0], y=point_d[1]))),

            (dir_d and game.is_collision(game.head._replace(x=point_r[0], y=point_r[1]))) or
            (dir_u and game.is_collision(game.head._replace(x=point_l[0], y=point_l[1]))) or
            (dir_r and game.is_collision(game.head._replace(x=point_u[0], y=point_u[1]))) or
            (dir_l and game.is_collision(game.head._replace(x=point_d[0], y=point_d[1]))),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        state_np = np.array(state, dtype=int)
        final_move = get_action_from_model(model, state_np)
        reward, done, score = game.play_step(final_move)

        if done:
            record = max(record, score)
            episodes += 1
            print(f"Episode {episodes} | Score: {score} | Record: {record}")
            game.reset()
            if max_episodes is not None and episodes >= max_episodes:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake DQN model.")
    parser.add_argument("--model", default="model.pth", help="Filename inside ./model to load (default: model.pth)")
    parser.add_argument("--width", type=int, default=640, help="Game width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Game height (default: 480)")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to run (default: infinite)")
    args = parser.parse_args()

    run_eval(model_path=args.model, width=args.width, height=args.height, max_episodes=args.episodes)
