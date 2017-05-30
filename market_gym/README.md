# Market Gym

These are the core files of the environment created to my dissertation. As explained by [OpenAI](https://github.com/openai/gym/blob/master/README.rst), there are two basic concepts in reinforcement learning: the environment and the agent.

For the purpose of this dissertation, the environment is a limit order book (LOB), where the agent will be able to insert limit orders and execute trades at the best prices. So, this folder encompasses all codes needed to run this environment created that consists in a LOB which reproduces historical high-frequency order book data. It is highly specialized in Brazilian interest rate market, but I believe that can be easily adapted to other Brazilian markets as well.

## What you should know about `config.py`

This file includes the configurations of the environment created, as the hour that the agent starts to trade, when the market closes, if it should print debug messages or not, the frequency that should print those messages, and so on. By default, the flag `DEBUG` flag is enabled, but there are other three flags to control the quantity of text printed. So, by default, the agent will write just the last line of the log generated.

## What you should know about `core.py`

This file includes the implementations of the main classes used in this library, which includes: `Env`, `RewardWrapper`, and `Agent`. For more information about each one, read the docs related to the classes that inherited from them, like [agent_frwk](../market_sim/README.md), [Environment and RewardFunc](envs/README.md)

## What you should know about the folders here

The folder [envs](envs/README.md) includes the final implementation of the environment, the reward functions and the simulator used in this dissertation. The folder [lob](lob/README.md) include all files related to the order book reconstruction. The folder [scripts](scripts/README.md) performs some help functions to clean the Bvmf files. Finally, the folder [utils](utils/README.md) includes several functions and libraries used by other components.
