#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Implement an environment where all agents interact, which is composed of
everything that it not part of the learning task, as the limit order book.
However, it also can include other details from the trading model, as stop
orders. This environment was built to simulate the books from future contracts
of interest rate from brazilian future market.

@author: ucaiado

Created on 05/24/2017
"""
from collections import OrderedDict
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bintrees import FastRBTree

from market_gym.lob.matching_engine import BvmfFileMatching
from market_gym.utils.book_rendering import img_init, img_update
import logging
from config import DEBUG, root, s_log_file
from config import START_MKT_TIME, CLOSE_MKT_TIME


'''
Begin help functions
'''


'''
End help functions
'''


class Env(object):
    '''
    The main Environment representation within which all agents operate. As the
    main porpose of this class is replaying historical high-frequency data, in
    fact, the environment encompass two different agents at most: a historical
    agent, that manage all orders from the database used, and the primary, that
    should be implemented by the user
    '''

    valid_actions = [None,
                     'BEST_BID',
                     'BEST_OFFER',
                     'BEST_BOTH',
                     'SELL',
                     'BUY',
                     'correction_by_trade',
                     'crossed_prices']

    def __init__(self, l_fname, l_instrument, NextStopTime, s_main_intrument,
                 i_idx=None, s_log_fname=None):
        '''
        Initialize an Env object

        :param l_fname: list. the container zip files to be used in simulation
        :param l_instrument: list. list of instrument to be simulated.
        :param NextStopTime: NextStopTime object. the hour all books is in sync
        :param s_main_intrument: string. The main instrument traded
        :param l_du: list of lists. Days to maturity of each contract
        :param l_pu: list. settlement prices of each instrument in l_du order
        :param l_price_adj: list. settlement rates in l_du order
        :param i_idx*: integer. The index of the start file to be read
        '''
        self.s_main_intrument = s_main_intrument
        self.l_instrument = l_instrument
        self.l_hedge = [x for x in l_instrument if x not in s_main_intrument]
        self.d_map_book_list = dict(zip(l_instrument,
                                        (np.cumsum([1]*len(l_instrument))-1)))
        self.done = False
        self.t = 0
        self.agent_states = OrderedDict()
        if not i_idx:
            i_idx = 0
        self.initial_idx = i_idx
        self.count_trials = 1
        self.s_log_fname = s_log_fname

        # Include Dummy agents
        self.num_dummies = 1  # no. of dummy agents
        self.last_id_agent = 10
        for i in xrange(self.num_dummies):
            self.create_agent(ZombieAgent)

        # Include Primary agent
        self.primary_agent = None  # to be set explicitly
        self.enforce_deadline = False

        # define stop times
        self.NextStopTime = NextStopTime

        # Initiate Matching Engine
        i_naux = self.num_dummies+1
        self.order_matching = BvmfFileMatching(env=self,
                                               l_instrument=l_instrument,
                                               i_num_agents=i_naux,
                                               l_file=l_fname)

        # define the best bid and offer attributes
        self._i_nrow = self.order_matching.i_nrow
        self.s_log_date = str(datetime.datetime.now().date()).replace('-', '')

        # trial data (updated at the end of each trial)
        self.d_trial_data = {'final_pnl': [],
                             'final_duration': [],
                             'max_pnl': [],
                             'min_pnl': [],
                             'final_reward': []}

        # reward function to be used
        self.reward_fun = RewardWrapper()
        self.reward_fun.set_func('pnl')
        self.s_rwd_fun = 'pnl'
        self.i_COUNT = 0
        # include 5 minutes as random time to start trading
        self.start_mkt_time = START_MKT_TIME + np.random.rand() * 5 * 60.
        self.close_mkt_time = CLOSE_MKT_TIME
        # control img rendering
        self.fig_book = None
        self.d_img_book = None

    def set_reward_function(self, s_type):
        '''
        Set a reward function to be used to judge to agent actions
        '''
        assert self.reward_fun.is_valid(s_type), 'Invalid reward function'
        self.reward_fun.set_func(s_type)
        self.s_rwd_fun = s_type

    def set_log_file(self, s_fname):
        '''
        Set additional name to include in log file name
        '''
        self.s_log_fname = s_fname

    def set_primary_agent(self, agent):
        '''
        Initiate the agent that is supposed to be modeled

        :param agent: Agent Object. The agent used as primary
        '''
        self.primary_agent = agent
        self.agent_states[agent] = {'Pnl': 0.,
                                    'OFI_Pnl': 0.,
                                    'Rwrd_Pnl': 0.,
                                    'Agent': agent,
                                    'best_bid': False,
                                    'best_offer': False}

        for s_instr in self.l_instrument:
            self.agent_states[agent][s_instr] = {}
            for s_key in ['qBid', 'Bid', 'Ask', 'qAsk']:
                f_val = 0
                if s_instr in agent.position:
                    f_val = agent.position[s_instr][s_key]
                self.agent_states[agent][s_instr][s_key] = f_val
            f_pos = self.agent_states[agent][s_instr]['qBid']
            f_pos -= self.agent_states[agent][s_instr]['qAsk']
            self.agent_states[agent][s_instr]['Position'] = f_pos

    def get_reward(self, state, agent):
        '''
        Return the reward based on the current state of the market and position
        accumulated by the agent
        '''
        # calculate the current PnL
        d_input = self.sense(agent)
        f_pnl = self._update_agent_pnl(agent, d_input)
        return self.reward_fun.get_reward(self, state, agent, f_pnl, d_input)

    def get_order_book(self, s_instrument, b_rtn_dataframe=False):
        '''
        Return a dataframe with the first 5 levels of the current order book

        :param s_instrument: string. Intrument to return the book
        :param b_rtn_dataframe*: boolean. return the book into a dataframe
        '''
        i_idx = self.d_map_book_list[s_instrument]
        my_book = self.order_matching.l_order_books[i_idx]
        if b_rtn_dataframe:
            return my_book.get_n_top_prices(5, b_return_dataframe=True)
        return my_book

    @property
    def i_nrow(self):
        '''
        Access the last idx row used by order matching
        '''
        self._i_nrow = self.order_matching.i_nrow
        return self._i_nrow

    def create_agent(self, agent_class, *args, **kwargs):
        '''
        Include a agent in the environment and initiate its env state

        :param agent_class: Agent Object. The agent desired
        :param args*, kwargs: any type. Any other parameter needed by the agent
        '''

        kwargs['i_id'] = self.last_id_agent
        agent = agent_class(self, *args, **kwargs)
        self.agent_states[agent] = {'Pnl': 0.,
                                    'OFI_Pnl': 0.,
                                    'Rwrd_Pnl': 0.,
                                    'Agent': agent,
                                    'best_bid': False,
                                    'best_offer': False}

        for s_instr in self.l_instrument:
            self.agent_states[agent][s_instr] = {}
            for s_key in ['qBid', 'Bid', 'Ask', 'qAsk']:
                f_val = 0
                if s_instr in agent.position:
                    f_val = agent.position[s_instr][s_key]
                self.agent_states[agent][s_instr][s_key] = f_val
            f_pos = self.agent_states[agent][s_instr]['qBid']
            f_pos -= self.agent_states[agent][s_instr]['qAsk']
            self.agent_states[agent][s_instr]['Position'] = f_pos
        self.last_id_agent += 1
        return agent

    def reset_generator(self):
        '''
        Reset the stop-time generator
        '''
        self.NextStopTime.reset()

    def reset(self, testing=False, carry_pos=False):
        '''
        Reset the environment and all variables needed as well as the states
        of each agent

        :param testing: boolean. If should freeze policy of the agents
        :param carry_pos: boolean. Carry position from previous simulation
        '''
        self.done = False
        self.t = 0
        self.order_matching.reset()
        self.reset_generator()

        # NOTE: it is ugly, but reset the order_matching idx. Should reviwe it
        self.order_matching.s_file
        # shuffle seed
        np.random.seed(seed=None)

        # Initialize agent(s)
        for agent in self.agent_states.iterkeys():
            self.agent_states[agent] = self._reset_agent_state(agent)

            # carry position
            d_pos = self._carry_position(agent, testing, carry_pos)
            if len(d_pos) > 0:
                s_add = ' '
                for s_key in d_pos:
                    s_add += '{} {:0.0f}'.format(s_key[-3:],
                                                 d_pos[s_key]['Q'])
                s_msg = 'Environment.reset(): Setting up initial'
                s_msg += ' position: ({}, {:0.4f})'
                s_msg = s_msg.format(s_add, d_pos[s_key]['P'])
                logging.info(s_msg)
            agent.set_initial_position(d_pos)
            agent.reset(testing=testing)

        # log the trial will start
        s_msg = 'Environment.reset(): New Trial will start!'
        i_aux = self.count_trials
        s_msg += ' ID {}'.format(i_aux)
        self.count_trials += 1
        # DEBUG
        logging.info(s_msg)

    def step(self):
        '''
        Perform a discreate step in the environment updating the state of all
        agents
        '''
        # Get the updates from the books that are not related to the history
        # until the stoptime
        l_msg = self.order_matching.next()
        # update the agents that is not the history
        for msg in l_msg:
            agent_aux = self.agent_states[msg['agent_id']]['Agent']
            self.update_agent_state(agent=agent_aux, msg=msg)
        # check if the market is closed
        b_t1 = self.order_matching.last_date >= self.close_mkt_time
        b_t2 = False
        if self.primary_agent:
            b_t2 = self.primary_agent.done
            # check if should update the primary
            if self.order_matching.last_date >= self.start_mkt_time:
                # should_update is defined in the agent.py
                if self.primary_agent.should_update():
                    self.update_agent_state(agent=self.primary_agent,
                                            msg=None)
            # check if should hedge
            b_test = (self.NextStopTime.s_stoptime_was_set == '')
            if self.primary_agent.need_to_hedge() and b_test:
                l_hedge_msg = self.primary_agent.hedge_position()
                for msg in l_hedge_msg:
                    agent_aux = self.agent_states[msg['agent_id']]['Agent']
                    self.update_agent_state(agent=agent_aux, msg=msg)
        # check if the market is closed
        if b_t1 or b_t2:
            self.done = True
            s_msg = 'Environment.step(): Market closed at {}!'
            s_msg += ' The session ended.'
            # log the last state of the primary state
            if self.primary_agent:
                s_act_msg = 'MKT_CLOSED'
                if b_t2:
                    s_act_msg = 'STOP'
                agent = self.primary_agent
                sense = self.sense(agent)
                state = self.agent_states[agent]
                reward = self.act(agent, None)
                self._update_agent_pnl(agent, sense, b_isclose=True)
                agent._pnl_information_update()
                agent.log_step(state, sense, s_act_msg, [], reward)
            # DEBUG
            logging.info(s_msg.format(self.order_matching.s_time))
        self.t += 1

    def sense(self, agent):
        '''
        Return the environment state that the agents can access

        :param agent: Agent object. the agent that will perform the action
        '''
        assert agent in self.agent_states, 'Unknown agent!'

        NotImplementedError('This method should be implemented')

    def act(self, agent, action):
        '''
        Return the environment reward or penalty by the agent's action and
        current state. Also, update the known condition of the agent's state
        by the Environment

        :param agent: Agent object. the agent that will perform the action
        :param action: dictionary. The current action of the agent
        '''
        assert agent in self.agent_states, 'Unknown agent!'
        if action:
            assert action['action'] in self.valid_actions, 'Invalid action!'
            # Update the position using action
            agent.act(action)
        position = agent.position
        # update current position in the agent state
        state = self.agent_states[agent]
        for s_instr in self.l_instrument:
            for s_key in ['Ask', 'qAsk', 'Bid', 'qBid']:
                state[s_instr][s_key] = float(position[s_instr][s_key])
            state[s_instr]['Position'] = state[s_instr]['qBid']
            state[s_instr]['Position'] -= state[s_instr]['qAsk']
        # check if it has orders in the best bid and offer
        tree_bid = agent.d_order_tree[self.s_main_intrument]['BID']
        tree_ask = agent.d_order_tree[self.s_main_intrument]['ASK']
        # Check if the agent has orders at the best prices
        state['best_bid'] = False
        state['best_offer'] = False
        obj_book = self.get_order_book(self.s_main_intrument)
        if tree_bid.count != 0:
            if obj_book.best_bid[0] <= tree_bid.max_key():
                state['best_bid'] = True
        if tree_ask.count != 0:
            if obj_book.best_ask[0] >= tree_ask.min_key():
                state['best_offer'] = True

        # measure the reward
        reward = self.get_reward(state, agent)

        return reward

    def update_agent_state(self, agent, msg):
        '''
        Update the agent state dictionary

        :param agent: Agent Object. The agent used as primary
        :param msg: dict. Order matching message
        '''
        # hold current information about position
        assert agent in self.agent_states, 'Unknown agent!'
        for s_instr in self.l_instrument:
            for s_key in ['qBid', 'Bid', 'Ask', 'qAsk']:
                t_aux = (s_instr, s_key)
                self.agent_states[agent][s_instr][s_key] = agent[t_aux]
            qBid = self.agent_states[agent][s_instr]['qBid']
            qAsk = self.agent_states[agent][s_instr]['qAsk']
            self.agent_states[agent][s_instr]['Position'] = qBid - qAsk
        # execute new action that can change current position
        agent.update(msg_env=msg)

    def update_order_book(self, l_msg):
        '''
        Update the Book and all information related to it

        :param l_msg: list. messages to use to update the book
        '''
        if isinstance(l_msg, dict):
            l_msg = [l_msg]
        if len(l_msg) > 0:
            self.order_matching.update(l_msg)

    def _carry_position(self, agent, testing, carry_pos):
        '''
        Return a dictionary with the positions to carry to the next episode

        :param agent: agent object.
        :param testing: boolean.
        :param carry_pos: boolean.
        '''
        return {}

    def _reset_agent_state(self, agent):
        '''
        Return a dictionary of default values to environment representation of
        agents states
        '''
        d_rtn = {'Pnl': 0., 'Agent': agent, 'best_bid': False,
                 'best_offer': False}
        return d_rtn

    def _update_agent_pnl(self, agent, sense, b_isclose=False):
        '''
        Update the agent's PnL and save is on the environment state

        :param agent. Agent object. the agent that will perform the action
        :param sense: dictionary. The inputs from environment to the agent
        '''
        f_pnl = 0.

        return f_pnl

    def log_trial(self):
        '''
        Log the end of current trial
        '''
        # log other informations
        if self.primary_agent:
            pass

    def render(self, mode='book'):
        '''
        Render one environment timestep

        :param mode: string. not using it yet
        '''
        try:
            img_update(self.d_img_book, self, self.primary_agent)
            plt.pause(0.01)
        except TypeError:
            self.fig_book = plt.figure(figsize=(8, 4))
            self.d_img_book = img_init(self.fig_book, self)
            plt.ion()


class RewardWrapper(object):
    '''
    RewardFunc is a wrapper for different functions used by the environment to
    assess the value of the agent's actions
    '''
    implemented = ['pnl']

    def __init__(self):
        '''
        Initiate a RewardFunc object. Save all parameters as attributes
        '''
        self.s_type = None
        self.log_info = {}
        self.last_time_closed = 0.
        self.last_pos = 0.

    def get_agent_pos(self, e, s, a, pnl, inputs):
        '''
        get position from the agent in the main instrument
        '''
        f_rtn = 0.
        s_main = e.s_main_intrument
        if not a.logged_action:
            return f_rtn
        # calculate the current position
        f_pos = a.position[s_main]['qBid']
        f_pos -= a.position[s_main]['qAsk']
        f_pos_discl = f_pos + a.disclosed_position[s_main]['qBid']
        f_pos_discl -= a.disclosed_position[s_main]['qAsk']
        return f_pos_discl

    def reset(self):
        '''
        '''
        self.log_info = {}
        self.last_time_closed = 0.
        self.last_pos = 0.

    def is_valid(self, s_type):
        '''
        Check if a type of reward was already implemented
        '''
        return s_type in self.implemented

    def set_func(self, s_type):
        '''
        Set the reward function used when the get_reward is called
        '''
        if s_type == 'pnl':
            self.s_type = s_type
            self.reward_fun = self._pnl

    def get_reward(self, env, d_state, agent, f_pnl, d_inputs):
        '''
        Return the reward using the arguments passed

        :param env. Environment object. Environment where the agent operates
        :param agent. Agent object. the agent that will perform the action
        :param d_state: dictionary. The state of the agent
        :param f_pnl: float. The current pnl of the agent
        :param d_inputs: dictionary. The inputs from environment to the agent
        '''
        reward = self.reward_fun(env, d_state, agent, f_pnl, d_inputs)
        if not reward:
            return 0.
        return reward

    def _pnl(self, e, s, a, pnl, inputs):
        '''
        Return the reward based on PnL from the last step marked to the
        mid-price of the instruments traded

        :param e: Environment object. Environment where the agent operates
        :param a: Agent object. the agent that will perform the action
        :param s: dictionary. The inputs from environment to the agent
        :param pnl: float. The current pnl of the agent
        :param inputs: dictionary. The inputs from environment to the agent
        '''
        # calculate the current PnL
        f_old_pnl = s['Pnl']
        # measure the reward depending on the type
        return np.around(pnl - f_old_pnl, 2)


class Agent(object):
    '''
    The main class for an agent
    '''
    # dict to use to find out what side the book was traded by the agent
    trade_side = {'Agressor': {'BID': 'Bid', 'ASK': 'Ask'},
                  'Passive': {'BID': 'Bid', 'ASK': 'Ask'}}
    order_side = {'Sell Order': 'ASK', 'Buy Order': 'BID'}

    def __init__(self, env, i_id):
        '''
        Initiate an Agent object. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        '''
        self.env = env
        self.i_id = i_id
        self.state = None
        self.done = False
        self.b_print_always = False
        self.position = {}
        self.ofi_acc = {}
        self.d_order_tree = {}
        self.d_order_map = {}
        self.d_trades = {}
        self.d_initial_pos = {}
        self.log_info = {'duration': 0., 'total_reward': 0.}
        self.learning = False  # Whether the agent is expected to learn
        for s_instr in self.env.l_instrument:
            self.position[s_instr] = {'qAsk': 0.,
                                      'Ask': 0.,
                                      'qBid': 0.,
                                      'Bid': 0.}
            self.ofi_acc[s_instr] = {'qAsk': 0.,
                                     'Ask': 0.,
                                     'qBid': 0.,
                                     'Bid': 0.}
            self.d_order_tree[s_instr] = {'BID': FastRBTree(),
                                          'ASK': FastRBTree()}
            self.d_order_map[s_instr] = {}
            self.d_trades[s_instr] = {'BID': [], 'ASK': []}

    def set_initial_position(self, d_pos):
        '''
        Set initial position in wach instrument traded

        :poaram d_pos: dict. positions in each instrument. Q and P is the keys
        '''
        self.d_initial_pos = d_pos

    def get_state(self):
        '''
        Return the inner state of the agent
        '''
        return self.state

    def get_position(self):
        '''
        Return the positions of the agent
        '''
        return self.position

    def get_intern_state(self, inputs, position):
        '''
        Return a tuple representing the intern state of the agent

        :param inputs: dictionary. traffic light and presence of cars
        :param position: dictionary. the current position of the agent
        '''
        pass

    def reset(self, testing=False):
        '''
        Reset the state and the agent's memory about its positions

        :param testing: boolean. If should freeze policy
        '''
        self.state = None
        self.done = False
        self.position = {}
        self.d_order_tree = {}
        self.d_order_map = {}
        self.d_trades = {}
        self.log_info = {'duration': 0., 'total_reward': 0.}
        for s_instr in self.env.l_instrument:
            self.position[s_instr] = {'qAsk': 0.,
                                      'Ask': 0.,
                                      'qBid': 0.,
                                      'Bid': 0.}
            self.ofi_acc[s_instr] = {'qAsk': 0.,
                                     'Ask': 0.,
                                     'qBid': 0.,
                                     'Bid': 0.}
            self.d_order_tree[s_instr] = {'BID': FastRBTree(),
                                          'ASK': FastRBTree()}
            self.d_order_map[s_instr] = {}
            self.d_trades[s_instr] = {'BID': [], 'ASK': []}

    def need_to_hedge(self):
        '''
        Return if the agent need to hedge position
        '''
        return False

    def should_update(self):
        '''
        Return a boolean informing if it is time to update the agent
        '''
        if self.env.i_nrow < 5:
            return False
        return True

    def act(self, msg):
        '''
        Update the positions of the agent based on the message passed

        :param msg: dict. Order matching message
        '''
        # recover some variables to use
        s_instr = msg['instrumento_symbol']
        s_status = msg['order_status']
        i_id = msg['order_id']
        s_side = self.order_side[msg['order_side']]
        # update position and order traking dict and tree
        if s_status in ['New', 'Replaced']:
            self.d_order_map[s_instr][i_id] = msg
            self.d_order_tree[s_instr][s_side].insert(msg['order_price'], msg)
        if s_status in ['Canceled', 'Expired']:
            old_msg = self.d_order_map[s_instr].pop(i_id)
            self.d_order_tree[s_instr][s_side].remove(old_msg['order_price'])
        elif s_status in ['Filled', 'Partially Filled']:
            # update the order map, if it was a passive trade
            if msg['agressor_indicator'] == 'Passive':
                if s_status == 'Filled':
                    if i_id in self.d_order_map[s_instr]:
                        old_msg = self.d_order_map[s_instr].pop(i_id)
                        f_aux = old_msg['order_price']
                        self.d_order_tree[s_instr][s_side].remove(f_aux)
                else:
                    # if it was partially filled, should re-include the msg
                    self.d_order_map[s_instr][i_id] = msg
                    f_aux = msg['order_price']
                    self.d_order_tree[s_instr][s_side].insert(f_aux, msg)
            # account the trades
            s_tside = self.trade_side[msg['agressor_indicator']]
            s_tside = s_tside[s_side]
            f_total_qty = float(msg['order_qty'])
            self.position[s_instr]['q' + s_tside] += f_total_qty
            f_volume = msg['order_price'] * f_total_qty
            self.position[s_instr][s_tside] += f_volume
            # account for OFI monetized by the agent
            s_tside = self.trade_side[msg['agressor_indicator']]
            s_tside = s_tside[s_side]
            f_total_qty = float(msg['order_qty'])
            d_inputs = self.env.sense(self)
            self.ofi_acc[s_instr]['q' + s_tside] += f_total_qty
            f_ofi = d_inputs['OFI'][s_instr] * f_total_qty
            self.ofi_acc[s_instr][s_tside] += f_ofi

            #  execute aditional actions
            self.additional_actions_when_exec(s_instr, s_side, msg)

    def additional_actions_when_exec(self, s_instr, s_side, msg):
        '''
        Execute additional action when execute a trade

        :param s_instr: string.
        :param s_side: string.
        :param msg: dictionary. Last trade message
        '''
        # log more information
        self.d_trades[s_instr][s_side].append(msg['order_price'])

    def update(self, msg):
        '''
        Update the inner state of the agent

        :param msg: dict. Order matching message
        '''
        NotImplementedError('This method should be implemented')

    def take_action(self, t_state, msg_env):
        '''
        Return an action according to the agent policy

        :param t_state: tuple. The inputs to be considered by the agent
        :param msg_env: dict. Order matching message
        '''
        return msg_env

    def apply_policy(self, state, action, reward):
        '''
        Learn policy based on state, action, reward

        :param state: dictionary. The current state of the agent
        :param action: string. the action selected at this time
        :param reward: integer. the rewards received due to the action
        '''
        pass

    def log_step(self, **kwargs):
        '''
        Log the current action/update from agent to a file or just to terminal

        :param kwargs*: Inputs to be used in the log string.
        '''
        pass

    def __str__(self):
        '''
        Return the name of the Agent
        '''
        return str(self.i_id)

    def __repr__(self):
        '''
        Return the name of the Agent
        '''
        return str(self.i_id)

    def __eq__(self, other):
        '''
        Return if a Agent has equal i_id from the other

        :param other: agent object. Agent to be compared
        '''
        i_id = other
        if not isinstance(other, int):
            i_id = other.i_id
        return self.i_id == i_id

    def __ne__(self, other):
        '''
        Return if a Agent has different i_id from the other

        :param other: agent object. Agent to be compared
        '''
        return not self.__eq__(other)

    def __hash__(self):
        '''
        Allow the Agent object be used as a key in a hash table. It is used by
        dictionaries
        '''
        return self.i_id.__hash__()

    def __getitem__(self, t_instr_idx):
        '''
        Allow direct access to the position information of the object

        :param t_instr_idx: tuple. two string, one is the intrument to get the
            position and the other is the information desired
        '''
        s_instr, s_idx = t_instr_idx
        return self.position[s_instr][s_idx]


class ZombieAgent(Agent):
    '''
    A ZombieAgent just obeys what the order matching engine determines
    '''

    def __init__(self, env, i_id):
        '''
        Initiate a ZombieAgent object. save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        '''
        super(ZombieAgent, self).__init__(env, i_id)

    def update(self, msg_env):
        '''
        Update the state of the agent.

        :param msg: dict. A message generated by the order matching
        '''
        # This agent dont really need to know about its position because,
        # you know, it is a zombie and it will speed up the simulaion

        # inputs = self.env.sense(self)
        # state = self.env.agent_states[self]

        # Update state (position ,volume and if has an order in bid or ask)
        # self.state = self._get_intern_state(inputs, state)

        # Select action according to the agent's policy
        # action = self._take_action(self.state, msg)

        # # Execute action and get reward
        # print '\ncurrent action: {}\n'.format(action)
        # reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        # self._apply_policy(self.state, action, reward)
        pass
