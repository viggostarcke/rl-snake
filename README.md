# Bachelor Thesis 2024 Project: Learning to Play Snake with RL
My WIP repo for my bachelor's thesis project on RL.

# Repo Overview
- [snake.py](https://github.com/viggostarcke/rl-snake/blob/main/snake.py): Holds attributes describing the snakes body aswell as position and functions to check for collisions.
- [game.py](https://github.com/viggostarcke/rl-snake/blob/main/game.py): PyGame implementation to play the game manually. Run `python game.py` to play the game.
- [environment.py](https://github.com/viggostarcke/rl-snake/blob/main/environment.py): Custom environment which follows OpenAI's Gymnasium framework structure.\
Action space is set to a discrete 3. Turn left, turn right or continue path.\
Observation space is set to an array of length 5 and a futher default of 10 extra elements. The first 5 elements describe the apples x and y, the snakes head x and y and the length of the snakes body. The 10 further elements describe the 10 last previous moves the snake took.\
reset()-function resets the snake position to the middle of the game grid, aswell as the apples position and the score.\
step()-function moves the snake and then checks for collisions or if the apple has been eaten.
- [dqn_model.py](https://github.com/viggostarcke/rl-snake/blob/main/dqn_model.py): The model for a neural network the DQN-agent will be using.
- [dqn_agent.py](https://github.com/viggostarcke/rl-snake/blob/main/dqn_agent.py): The snake agent that uses DQN to learn how to make decisions.
- [dqn_trainer.py](https://github.com/viggostarcke/rl-snake/blob/main/dqn_trainer.py): A trainer that trains a snake agent using my custom environment and DQN-agent. Runs 1000 times, updating the target network every 10th game.
