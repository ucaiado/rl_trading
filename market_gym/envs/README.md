# Environment

This folder includes the final implementation of the environment and reward functions used in this dissertation. Modify it as you want. The codes in `registration.py` allows to instantiate different versions of the environment using a string identifier, so, just include new environments in the IMPLEMENTED_ENVS dictionary in this module. `Simulator.py` is what runs the simulation itself. The following code block illustrates the use of `make` function and `Simulator` module to use Bovespa data.

```python
import market_gym
from market_gym.envs import make, Simulator
from market_gym.lob import matching_engine

# set up the stop time object
l_h = [9, 10, 11, 12, 13, 14, 15, 16]
NextStopTime = matching_engine.NextStopTime(l_h, i_milis=500.)

# set up the variables to environment
l_instrument=['PETR4']
NextStopTime=NextStopTime
s_main_intrument='PETR4'
s_root = '../data/preprocessed/'
s_date = '20170411'
l_files = [s_root + s_date + '/' + s_date + '_{}_{}_new.zip']

# set up the environment
obj_env = make('StockAndOption')
e = obj_env(l_fname=l_files,
            l_instrument=l_instrument,
            NextStopTime=NextStopTime,
            s_main_intrument=s_main_intrument)

# run the simulation
sim = Simulator(e, update_delay=1.00, display=False)
%time sim.run(n_trials=1, n_sessions=1)


```

Regarding `Simulator`, one interesting method is `print_when_paused(self)`. If you hit Ctrl + C while running `market_sim/agent.py` script, it will pause the simulation and print out some information about the current state of the environment and the agent. It will ask you if should continue or quit the episode. If you choose to quit, it will follow to the next episode. To stop the simulation, you should hit ctrl+C twice. About reward functions, the following are the `RewardFunc` methods you should know:

- set\_func(self, s_type): Set the kind of reward used by the agent.
- reset(self): Reset the reward function state. It was used by learning agent. Should be put in another place, probably

If you want to implement new reward functions, include the alias for the function in the `implemented` attribute of the `RewardFunc` class and include a new method using your function, so it should look something like:

```python
    class RewardFunc(RewardWrapper):
        implemented = ['your_func', 'pnl', ...]
        (...)
        def reset(self):
            '''
            '''
            (...)
            # reset any other variable here
            pass

       def set_func(self, s_type):
            '''
            Set the reward function used when the get_reward is called
            '''
            if s_type == 'pnl':
                self.s_type = s_type
                self.reward_fun = self._pnl
            elif s_type == 'your_func':
                self.s_type = s_type
                self.reward_fun = self._your_func

        def _your_func(self, e, s, a, pnl, inputs):
            '''
            Return the reward based on PnL from the last step marked to the
            mid-price of the instruments traded

            :param e: Environment object. Environment where the agent operates
            :param a: Agent object. the agent that will perform the action
            :param s: dictionary. The inputs from environment to the agent
            :param pnl: float. The current pnl of the agent
            :param inputs: dictionary. The inputs from environment to the agent
            '''
            # implement something here
            reward = 0.
            return reward
```

Finally, the following are the `Environment` methods you should know:

- _reset_agent_state(self): Reset the agent’s state. Prepare to start a new episode.
- _update_agent_pnl(self, agent, sense, b_isclose=False): Return the current PnL of the agent and also update the PnL information in `agents_state` attribute, from `Environment`. The flag `b_isclose` informs the environment if the market is closed and is mainly used when you are trading future contracts, which have settlement prices.
- sense(self, agent): Return the environment state as a dictionary that the agent can access. An `agent` is an Agent object
- log_trial(self): Log the final data of current trial as a JSON in a file that begins with “result” in its name
