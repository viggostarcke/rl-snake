# Bachelor Thesis 2024 Project: Learning to Play Snake with RL
My WIP repo for my bachelor's thesis project on RL.\
Following libraries should be installed as **dependencies** to run everything in this repo\
PyGame: `pip install pygame`\
Gymnasium: `pip install gymnasium`\
PyTorch: `pip install torch`\
Stable-Baselines3: `pip install stable-baselines3`

# Repo Overview
- [snake.py](https://github.com/viggostarcke/rl-snake/blob/main/snake.py): Holds attributes describing the snakes body aswell as position and functions to check for collisions.
- [game.py](https://github.com/viggostarcke/rl-snake/blob/main/game.py): PyGame implementation to play the game manually. Run `python game.py` to play the game.
- [environment.py](https://github.com/viggostarcke/rl-snake/blob/main/environment.py): Custom environment which follows OpenAI's Gymnasium framework structure.\
The snake is rendered with a blue head, while the rest of the body is green.\
*Action space* is set to a discrete 3. Turn left, turn right or continue path.\
*Observation space* is set to an array of length 5 and a futher default of 10 extra elements. The first 5 elements describe the apples x and y, the snakes head x and y and the length of the snakes body. The 10 further elements describe the 10 last previous moves the snake took.\
*reset()*-function resets the snake position to the middle of the game grid, aswell as the apples position and the score.\
*step()*-function moves the snake and then checks for collisions or if the apple has been eaten.
- [dqn_agent.py](https://github.com/viggostarcke/rl-snake/blob/main/dqn_agent.py): The snake agent that uses standard stable-baselines3 DQN to learn how to make decisions. Configure render mode and learning inside file.
- [ppo_agent.py](https://github.com/viggostarcke/rl-snake/blob/main/ppo_agent.py): ***Not implemented properly yet***: The snake agent that uses standard stable-baselines3 PPO to learn how to make decisions.
# Issues
**DQN**:
- model.learn works properly. The learning however is abit tame, and the snake never reaches later stages in game.
- model.load doesn't work. The snake gets stuck in loops right away. 
