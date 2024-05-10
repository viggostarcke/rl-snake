# Bachelor Thesis 2024 Project: Learning to Play Snake with RL
![](https://github.com/viggostarcke/rl-snake/blob/main/.img/render_of_game.gif)


## Dependencies
Following libraries should be installed to run everything in this repo\
PyGame: `pip install pygame`\
Gymnasium: `pip install gymnasium`\
PyTorch: `pip install torch`\
Stable-Baselines3: `pip install stable-baselines3`

## Repo Overview
- [snake.py](https://github.com/viggostarcke/rl-snake/blob/main/snake.py): Holds attributes describing the snakes body aswell as position and functions to check for collisions.
- [game.py](https://github.com/viggostarcke/rl-snake/blob/main/game.py): PyGame implementation to play the game manually. Run `python game.py` to play the game.
- [environment.py](https://github.com/viggostarcke/rl-snake/blob/main/environment.py): Custom environment which follows OpenAI's Gymnasium framework structure.
- [dqn_agent.py](https://github.com/viggostarcke/rl-snake/blob/main/dqn_agent.py): A standard stable-baselines3 DQN implementation of a game agent.
- [ppo_agent.py](https://github.com/viggostarcke/rl-snake/blob/main/ppo_agent.py): A standard stable-baselines3 PPO implementation of a game agent.
- [a_star_agent.py](https://github.com/viggostarcke/rl-snake/blob/main/a_star_agent.py): A* implementation to snake environment.

## Environment Overview
### Action space
**0-2:** Turn left, continue path, turn right.

### Observation space:
- **compass_distances:** 8 element array that describes the distance to nearest object and apple in each compass direction. Each element holds a tuple containing (distance to nearest obstacle, distance to apple). If there is no apple in that direction it holds a standard value of 0.

### Hunger:
A countdown, which essentially limits the total amount of moves without obtaining an apple to the total amount of tiles on the board.
This discourages getting stuck in local maxima due to endless loop behavior.

### Reward function:
- **-1:** For an action that results in collision with the wall or snake's body.
- **-0.2:** For an action that reaches the hunger limit (100 moves without eating an apple).
- **-0.01:** For an empty action (i.e. does not result in collision, eating apple or reaching hunger limit).
- **+0.2:** For an action that results in obtaining an apple.

## Instructions
Run the game to play it manually: `python .\game.py`\
Run the game to see the A* agent play it: `python .\a_star_agent.py`

Run an agent, and pass arguments:\
`--learn` (`-l`): Runs sb3 *.learn* method and saves the model in a file `dqn_agent.zip` or `ppo_agent.zip`.\
`--test` (`-t`): Runs the saved model.\
`--render` (`-r`): Renders every game.\
ex: `python .\dqn_agent.py --test -r` (runs the saved DQN model and renders every game.)
