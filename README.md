# AI Snake — Deep Q-Learning with PyTorch

An AI agent that learns to play the classic Snake game using **Deep Q-Learning (DQN)** and PyTorch. The agent starts with no knowledge of the game and progressively improves its score through reinforcement learning.

## How It Works

The project uses a **Deep Q-Network (DQN)** — a reinforcement learning technique that combines Q-learning with a neural network to approximate the optimal action-value function.

### State Representation (11 features)

At each frame the agent observes 11 binary features:

| Feature | Description |
|---------|-------------|
| Danger straight | Collision ahead in the current direction |
| Danger right | Collision if the snake turns right |
| Danger left | Collision if the snake turns left |
| Direction (×4) | Current heading: LEFT, RIGHT, UP, DOWN |
| Food location (×4) | Whether food is to the left, right, above, or below |

### Actions

The agent chooses one of three relative actions each step:

- `[1, 0, 0]` — Go straight
- `[0, 1, 0]` — Turn right
- `[0, 0, 1]` — Turn left

### Rewards

| Event | Reward |
|-------|--------|
| Eat food | +10 |
| Hit wall or self | −10 |
| Timeout (no food eaten) | −10 |

### Neural Network (`model.py`)

A simple two-layer feedforward network:

```
Input (11) → Linear → ReLU → Hidden (256) → Linear → Output (3)
```

- **Optimizer**: Adam (lr = 0.001)
- **Loss**: Mean Squared Error (MSE)

### Training (`agent.py`)

The agent is trained with two complementary mechanisms:

1. **Short-term memory** — learns immediately from each individual step.
2. **Long-term memory (experience replay)** — after each game, samples a random mini-batch of 1,000 transitions from a replay buffer (capacity 100,000) and trains on them.

**Exploration vs. exploitation** is managed by an ε-greedy policy:

```
ε = 80 − number_of_games
```

The agent acts randomly while ε > 0 (roughly the first 80 games) and switches to pure exploitation once it has gathered enough experience.

The best model is automatically saved to `./model/model.pth` whenever a new high score is reached.

## Project Structure

```
.
├── agent.py          # Training loop and Agent class
├── game.py           # Snake game engine (Pygame)
├── model.py          # Neural network and QTrainer
├── helper.py         # Real-time training plot (matplotlib)
├── arial.ttf         # Font used by the game UI
└── requirements.txt  # Python dependencies
```

## Requirements

- Python 3.x
- PyTorch 2.x
- Pygame
- NumPy
- Matplotlib
- IPython

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Running

Start training:

```bash
python agent.py
```

This opens the Snake game window and a live plot that shows the score for each game alongside the running mean score. Training runs indefinitely — close the window to stop.

Progress is printed to the console after every game:

```
Game 1  Score 0  Record: 0
Game 2  Score 1  Record: 1
...
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Max replay memory | 100,000 |
| Batch size | 1,000 |
| Learning rate | 0.001 |
| Discount factor (γ) | 0.9 |
| Hidden layer size | 256 |
| Game speed | 40 FPS |
| Board size | 640 × 480 px |
