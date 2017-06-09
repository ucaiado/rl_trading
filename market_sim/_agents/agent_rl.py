#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The current implementation of an agent that learns in a reinforcement learning
framework to trade Brazilian interest rate future contracts

@author: ucaiado

Created on 11/21/2016
"""
import random
import logging
import sys
import time
import numpy as np
import pandas as pd
import pickle
import pprint
import math

from market_gym import Agent
from market_gym.envs import Simulator
from market_gym.lob import matching_engine, translator
from market_gym.config import PRINT_ALL, PRINT_5MIN, DEBUG, root, s_log_file
from market_gym.config import STOP_MKT_TIME
from agent_frwk import BasicAgent
import risk_model
import tile_coding


'''
Begin help functions
'''


class InvalidOptionException(Exception):
    """
    InvalidOptionException is raised by the run() function and indicate that no
    valid test option was selected
    """
    pass


def argmax(elements, unique=True):
    '''
    Return a list with the maximum elemnt in a array.
    Source: https://goo.gl/eeN6Rb

    :param elements: list. values to be checked
    :param unique*: boolean. if shoould return a single value
    '''
    maxValue = np.max(elements)
    candidates = np.where(np.asarray(elements) == maxValue)[0]
    if unique:
        return np.random.choice(candidates)
    return list(candidates)


class ValueFunction(object):
    '''
    wrapper class for state action value function
    Source: https://goo.gl/eeN6Rb

    original author: In this example I use the tiling software instead of
    implementing standard tiling by myself One important thing is that tiling
    is only a map from (state, action) to a series of indices It doesn't matter
    whether the indices have meaning, only if this map satisfy some property
    View the following webpage for more information:
    http://incompleteideas.net/sutton/tiles/tiles3.html
    '''

    def __init__(self, stepSize, d_normalizers, numOfTilings=8, maxSize=None):
        '''
        Initialize a ValueFunction object that represents a compact Q-Function
        with a linearly parameterized approximator applying tile coding

        :param stepSize: float. Also can be understood as the learning factor
        :param d_normalizers: dictionary.
        :param numOfTilings*: integer. (optional)
        :param maxSize*: integer. (optional). the maximum # of indices
        :param s_decayfun*: string. The decay function applyed to epislon
        '''
        # selectin values: https://webdocs.cs.ualberta.ca/~sutton/tiles2.html
        if not maxSize:
            # NOTE 1: number of tiles k should be 2^n >= 4d, where d is the
            #    number of continuos features
            # NOTE 2:  number of tiles \times resolution factor (power of two)
            maxSize = numOfTilings * 2**11  # 2**5
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings

        # divide step size equally to each tiling
        self.stepSize = stepSize * 1. / numOfTilings
        self.last_alpha = stepSize

        # initialize the BFs (Busoniu, p. 6)
        self.hashTable = tile_coding.IHT(maxSize)

        # weight for each tile
        self.weights = np.zeros(maxSize)

        # features needs scaling to satisfy the tile software
        self.featuresScale = {}
        self.d_normalizers = {}
        for s_key, d_value in d_normalizers.iteritems():
            self.d_normalizers[s_key] = {}
            self.d_normalizers[s_key]['MAX'] = d_value['MAX']
            self.d_normalizers[s_key]['MIN'] = d_value['MIN']
            f_value = d_value['MAX'] - d_value['MIN']
            self.featuresScale[s_key] = self.numOfTilings / f_value

    def get_alpha_k(self, step):
        '''
        get the current $\alpha_k$ according to the learning rate schedule
        '''
        self.last_alpha = self.stepSize * self.numOfTilings
        return self.stepSize

    def getActiveTiles(self, d_state, s_action, agent):
        '''
        get indices of active tiles for given state and action

        :param d_state: dictionary. the intern state representation of an agent
        :param s_action: string. a valid action
        :param agent: Agent object. the agent using the value function
        '''
        action = agent.d_translate_to_valuefun[s_action]
        l_features_values = []
        for s_key in agent.features_names:
            f_min = self.d_normalizers[s_key]['MIN']
            f_value = max(0., d_state[s_key] - f_min)
            if d_state[s_key] > self.d_normalizers[s_key]['MAX']:
                f_value = self.d_normalizers[s_key]['MAX']
                f_value -= self.d_normalizers[s_key]['MIN']
            f_value *= self.featuresScale[s_key]
            l_features_values.append(f_value)
        activeTiles = tile_coding.tiles(self.hashTable, self.numOfTilings,
                                        l_features_values, [action])
        return activeTiles

    def value(self, d_state, s_action, agent):
        '''
        estimate the value of given state and action

        :param d_state: dictionary. the intern state representation of an agent
        :param s_action: string. a valid action
        :param agent: Agent object. the agent using the value function
        '''
        # get indices to produce $\phi$ vector weights[activeTiles]
        activeTiles = self.getActiveTiles(d_state, s_action, agent)
        # apply: \hat{Q}(x, u) = \phi^T (x, u) \theta
        return np.sum(self.weights[activeTiles])

    def learn(self, d_state, s_action, target, agent):
        '''
        learn with given state, action and target

        :param d_state: dictionary. the intern state representation of an agent
        :param action: integer. the ID of a valid action
        :param target: float. Q-value target gotten by the agent
        :param agent: Agent object. the agent using the value function
        '''
        activeTiles = self.getActiveTiles(d_state, s_action, agent)
        estimation = np.sum(self.weights[activeTiles])
        # apply: \theta_{k+1} = \theta_{k} + \alpha_k [ Q^{*} - \hat{Q}] \phi
        # where \hat{Q} is estimated target and Q^{*} is the target
        # Q^{*} = r_{t+1} + \gamma max_{a'}(\phi^T(s_{t+1}, a') \theta_t)
        # \hat{Q} = \phi^T(s_t, a_t) \theta_t)
        # step_size = \alpha_k
        step_size = self.get_alpha_k(agent.k_steps)
        try:
            delta = step_size * (target - estimation)
        except:
            print
            print 'stepsize: {}'.format(step_size)
            print 'target: {}'.format(target)
            print 'estimation: {}'.format(estimation)
            print
            raise
        for activeTile in activeTiles:
            # here is: \theta_{k+1} = \theta_{k} + \delta_k
            self.weights[activeTile] += delta


'''
End help functions
'''


class QLearningAgent(BasicAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    actions_to_open = [None, 'BEST_BID', 'BEST_OFFER', 'BEST_BOTH']
    actions_to_close_when_short = [None, 'BEST_BID']
    actions_to_close_when_long = [None, 'BEST_OFFER']
    actions_to_stop_when_short = [None, 'BEST_BID', 'BUY']
    actions_to_stop_when_long = [None, 'BEST_OFFER', 'SELL']

    FROZEN_POLICY = False

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True,  b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        f_aux = f_ttoupdate
        super(QLearningAgent, self).__init__(env, i_id, f_min_time, f_aux,
                                             d_initial_pos=d_initial_pos)
        self.learning = True  # this agent is expected to learn
        self.decayfun = s_decay_fun
        # Initialize any additional variables here
        self.max_pos = 100.
        self.max_disclosed_pos = 10.
        self.orders_lim = 4
        self.order_size = 5
        self.s_agent_name = 'QLearningAgent'
        # control hedging
        obj_aux = risk_model.GreedyHedgeModel
        self.s_hedging_on = s_hedging_on
        self.risk_model = obj_aux(env, s_instrument=s_hedging_on,
                                  s_fairness='closeout')
        self.last_spread = [0.0, 0.0]
        self.f_spread = [0.0, 0.0]
        self.f_gamma = f_gamma
        self.f_alpha = f_alpha
        self.f_epsilon = 1.0
        self.b_hedging = b_hedging
        self.current_open_price = None
        self.current_max_price = -9999.
        self.current_min_price = 9999.
        self.b_keep_pos = b_keep_pos

        # Initialize any additional variables here
        self.f_time_to_buy = 0.
        self.f_time_to_sell = 0.
        self.b_print_always = False
        self.d_normalizers = d_normalizers
        self.d_ofi_scale = d_ofi_scale
        self.numOfTilings = i_numOfTilings
        self.alpha = f_alpha
        i_nTiling = i_numOfTilings
        value_fun = ValueFunction(f_alpha, d_normalizers, i_nTiling)
        self.value_function = value_fun
        self.old_state = None
        self.last_action = None
        self.last_reward = None
        self.disclosed_position = {}
        self.f_stop_time = STOP_MKT_TIME - 1 + 1
        # self.features_names = ['position', 'ofi_new', 'spread_longo',
        #                        'ratio_longo', 'ratio_curto',
        #                        'size_bid_longo', 'size_bid_curto',
        #                        'spread_curto', 'high_low', 'rel_price']
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo',
                               'rel_price']

    def reset_additional_variables(self, testing):
        '''
        Reset the state and the agent's memory about its positions

        :param testing: boolean. If should freeze policy
        '''
        self.risk_model.reset()
        self.f_time_to_buy = 0.
        self.f_time_to_sell = 0.
        self.last_reward = None
        self.current_open_price = None
        self.current_max_price = -9999.
        self.current_min_price = 9999.
        self.spread_position = {}
        self.disclosed_position = {}
        self.env.reward_fun.reset()
        for s_instr in self.env.l_instrument:
            self.disclosed_position[s_instr] = {'qAsk': 0.,
                                                'Ask': 0.,
                                                'qBid': 0.,
                                                'Bid': 0.}

        if testing:
            self.freeze_policy()

    def additional_actions_when_exec(self, s_instr, s_side, msg):
        '''
        Execute additional action when execute a trade

        :param s_instr: string.
        :param s_side: string.
        :param msg: dictionary. Last trade message
        '''
        # check if the main intrument was traded
        s_main = self.env.s_main_intrument
        if msg['instrumento_symbol'] == s_main:
            self.b_has_traded = True
        # check if it open or close a pos
        f_pos = self.position[s_instr]['qBid']
        f_pos -= self.position[s_instr]['qAsk']
        b_zeroout_buy = f_pos == 0 and s_side == 'ASK'
        b_zeroout_sell = f_pos == 0 and s_side == 'BID'
        b_new_buy = f_pos > 0 and s_side == 'BID'
        b_new_sell = f_pos < 0 and s_side == 'ASK'
        b_close_buy = f_pos > 0 and s_side == 'ASK'
        b_close_sell = f_pos < 0 and s_side == 'BID'
        s_other_side = 'BID'

        # set the time to open position if it just close it
        if b_close_buy or b_zeroout_buy:
            self.f_time_to_buy = self.env.order_matching.f_time + 60.
        elif b_close_sell or b_zeroout_sell:
            self.f_time_to_sell = self.env.order_matching.f_time + 60.

        # print when executed
        f_pnl = self.log_info['pnl']
        s_time = self.env.order_matching.s_time
        s_err = '{}: {} - current position {:0.2f}, PnL: {:0.2f}\n'
        print s_err.format(s_time, s_instr, f_pos, f_pnl)

        # keep a list of the opened positions
        if s_side == 'BID':
            s_other_side = 'ASK'
        if b_zeroout_buy or b_zeroout_sell:
            self.current_open_price = None  # update by risk model
            self.current_max_price = -9999.
            self.current_min_price = 9999.
            self.d_trades[s_instr][s_side] = []
            self.d_trades[s_instr][s_other_side] = []
        elif b_new_buy or b_new_sell:
            if b_new_buy:
                self.risk_model.price_stop_sell = None
            if b_new_sell:
                self.risk_model.price_stop_buy = None
            # log more information
            l_info_to_hold = [msg['order_price'], msg['order_qty'], None]
            if 'last_inputs' in self.log_info:
                l_info_to_hold[2] = self.log_info['last_inputs']['TOB']
            self.d_trades[s_instr][s_side].append(l_info_to_hold)
            self.d_trades[s_instr][s_other_side] = []
        elif b_close_buy or b_close_sell:
            f_qty_to_match = msg['order_qty']
            l_aux = []
            for f_price, f_qty, d_tob in self.d_trades[s_instr][s_other_side]:
                if f_qty_to_match == 0:
                    l_aux.append([f_price, f_qty, d_tob])
                elif f_qty <= f_qty_to_match:
                    f_qty_to_match -= f_qty
                elif f_qty > f_qty_to_match:
                    f_qty -= f_qty_to_match
                    f_qty_to_match = 0
                    l_aux.append([f_price, f_qty, d_tob])
            self.d_trades[s_instr][s_other_side] = l_aux
            if abs(f_qty_to_match) > 0:
                l_info_to_hold = [msg['order_price'], f_qty_to_match, None]
                if 'last_inputs' in self.log_info:
                    l_info_to_hold[2] = self.log_info['last_inputs']['TOB']
                self.d_trades[s_instr][s_side].append(l_info_to_hold)

    def need_to_hedge(self):
        '''
        Return if the agent need to hedge position
        '''
        # ask risk model if should hedge
        if not self.b_hedging:
            return False
        if not self.b_keep_pos:
            if self.env.order_matching.last_date > self.f_stop_time:
                if abs(self.log_info['duration']) > 0.01:
                    self.b_need_hedge = True
                    # print 'need_to_hedge(): HERE'
                    return self.b_need_hedge
        if self.risk_model.should_stop_disclosed(self):
            return True
        if self.risk_model.should_hedge_open_position(self):
            # check if should hedge position
            if abs(self.log_info['duration']) > 1.:
                self.b_need_hedge = True
            return self.b_need_hedge
        return False

    def get_valid_actions_old(self):
        '''
        Return a list of valid actions based on the current position
        '''
        # b_stop = False
        valid_actions = list(self.actions_to_open)
        if not self.risk_model.can_open_position('ASK', self):
            valid_actions = list(self.actions_to_close_when_short)  # copy
            if self.risk_model.should_stop_disclosed(self):
                # b_stop = True
                valid_actions = list(self.actions_to_stop_when_short)
        elif not self.risk_model.can_open_position('BID', self):
            valid_actions = list(self.actions_to_close_when_long)
            if self.risk_model.should_stop_disclosed(self):
                # b_stop = True
                valid_actions = list(self.actions_to_stop_when_long)

        return valid_actions

    def get_valid_actions(self):
        '''
        Return a list of valid actions based on the current position
        '''
        # b_stop = False
        valid_actions = list(self.actions_to_open)

        return valid_actions

    def get_intern_state(self, inputs, state):
        '''
        Return a dcitionary representing the intern state of the agent

        :param inputs: dictionary. what the agent can sense from env
        :param state: dictionary. the current state of the agent
        '''
        d_data = {}
        s_main = self.env.s_main_intrument
        d_data['OFI'] = inputs['qOfi']
        d_data['qBID'] = inputs['qBid']
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        # if self.logged_action:
        # compare with the last data
        # if 'to_delta' in self.log_info:
        #     # measure the change in OFI from he last sction taken
        #     for s_key in [s_main]:
        #         i_ofi_now = inputs['OFI'][s_key]
        #         i_ofi_old = self.log_info['to_delta']['OFI'][s_key]
        #         f_aux = i_ofi_now - i_ofi_old
        #         f_last_ofi += f_aux
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.s_main_intrument
        s_crt = self.s_hedging_on
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    # fun(inputs['spread'][s_crt], 'spread_curto'),
                    # fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    # fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    # fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn

    def bound_values(self, f_value, s_feature_name, s_cmm=None):
        '''
        Return the value bounded by the maximum and minimum values predicted.
        Also apply nomalizations functions if it is defined and d_normalizers,
        in the FUN key.

        :param f_value: float. value to be bounded
        :param s_feature_name: string. the name of the feature in d_normalizers
        :param s_cmm*: string. Name of the instrument
        '''
        f_max = self.d_normalizers[s_feature_name]['MAX']
        f_min = self.d_normalizers[s_feature_name]['MIN']
        f_value2 = max(f_min, f_value)
        f_value2 = min(f_max, f_value)
        if 'FUN' in self.d_normalizers[s_feature_name]:
            if s_feature_name == 'ofi_new':
                f = self.d_normalizers[s_feature_name]['FUN'](f_value, s_cmm)
                f = max(f_min, f)
                f = min(f_max, f)
                f_value2 = f
            else:
                f_value2 = self.d_normalizers[s_feature_name]['FUN'](f_value2)
        return f_value2

    def get_epsilon_k(self):
        '''
        Get $\epsilon_k$ according to the exploration schedule
        '''
        trial = self.env.count_trials - 2  # ?
        if self.decayfun == 'tpower':
            # e = a^t, where 0 < z < 1
            # self.f_epsilon = math.pow(0.9675, trial)  # for 100 trials
            self.f_epsilon = math.pow(0.9333, trial)  # for 50 trials

        elif self.decayfun == 'trig':
            # e = cos(at), where 0 < z < 1
            # self.f_epsilon = math.cos(0.0168 * trial)  # for 100 trials
            self.f_epsilon = math.cos(0.03457 * trial)  # for 50 trials
        else:
            # self.f_epsilon = max(0., 1. - (1./45. * trial))  # for 50 trials
            self.f_epsilon = max(0., 1. - (1./95. * trial))  # for 100 trials
        return self.f_epsilon

    def choose_an_action(self, d_state, valid_actions):
        '''
        Return an action from a list of allowed actions according to the
        agent's policy based on epsilon greedy policy and valueFunction

        :param d_state: dictionary. The inputs to be considered by the agent
        :param valid_actions: list. List of the allowed actions
        '''
        # return a uniform random action  with prob $\epsilon_k$ (exploration)
        state_ = d_state['features']
        best_Action = random.choice(valid_actions)
        if not self.FROZEN_POLICY:
            if np.random.binomial(1, self.get_epsilon_k()) == 1:
                return best_Action
        # apply: arg max_{u'} ( \phi^T (x_k, u') \theta_k)
        values = []
        for action in valid_actions:
            values.append(self.value_function.value(state_, action, self))
        # return self.d_value_to_action[argmax(values)]
        return valid_actions[argmax(values)]

    def apply_policy(self, state, action, reward):
        '''
        Learn policy based on state, action, reward. The algo part of "apply
        action u_k" is in the update method from agent frmk as the update just
        occur after one trial, state and reward are at the next step. Return
        True if the policy was updated

        :param state: dictionary. The current state of the agent
        :param action: string. the action selected at this time
        :param reward: integer. the rewards received due to the action
        '''
        # check if there is some state in cache
        state_ = state['features']
        valid_actions = self.get_valid_actions()
        if self.old_state and not self.FROZEN_POLICY:
            # TD Update
            q_values_next = []
            for act in valid_actions:
                # for the vector: $(\phi^t (x_{k+1}, u') * \theta_k)_{u'}$
                # state here plays the role of next state x_{k+1}. act are u's
                f_value = self.value_function.value(state_, act, self)
                q_values_next.append(f_value)

            # Q-Value TD Target
            # apply: Qhat <- r_{k+1} + y max_u' (\phi^T(x_{k+1}, u') \theta_k)
            # note that u' is the result of apply u in x. u' is the action that
            # would maximize the estimated Q-value for the state x'
            td_target = self.last_reward + self.f_gamma * np.max(q_values_next)

            # Update the state value function using our target
            # apply: $\theta_{k+1} <- \alpha_k (Q_  - Qhat) \theta(x_k, u_k)$
            # the remain part of the update is inside the method learn
            # use last_action here because it generated the curremt reward
            self.value_function.learn(self.old_state, self.last_action,
                                      td_target, self)

        # save current state, action and reward to use in the next run
        self.old_state = state_  # in the next run it is x_k <- x_{k+1}
        self.last_action = action  # in the next run it is  u_k
        self.last_reward = reward  # in the next run it is r_{k+1}

        if action in ['SELL', 'BUY']:
            print '=',

        return True

    def set_qtable(self, s_fname, b_freezy_policy=True):
        '''
        Set up the q-table to be used in testing simulation and freeze policy

        :param s_fname: string. Path to the qtable to be used
        '''
        # freeze policy if it is for test and not for traning
        if b_freezy_policy:
            self.freeze_policy()
        # load qtable and transform in a dictionary
        value_fun = pickle.load(open(s_fname, 'r'))
        self.value_function = value_fun
        # log file used
        s_print = '{}.set_qtable(): Setting up the agent to use'
        s_print = s_print.format(self.s_agent_name)
        s_print += ' the Value Function at {}'.format(s_fname)
        # DEBUG
        logging.info(s_print)

    def stop_on_main(self, l_msg, l_spread):
        '''
        Stop on the main instrument

        :param l_msg: list.
        :param l_spread: list.
        '''
        s_main_action = ''
        if self.risk_model.should_stop_disclosed(self):
            if self.log_info['duration'] < 0.:
                print '=',
                s_main_action = 'SELL'
            if self.log_info['duration'] > 0.:
                print '=',
                s_main_action = 'BUY'
        if self.env.order_matching.last_date > self.f_stop_time:
            if not self.b_keep_pos:
                if self.log_info['duration'] < 0.:
                    print '>',
                    s_main_action = 'SELL'
                if self.log_info['duration'] > 0.:
                    print '>',
                    s_main_action = 'BUY'
        # place orders in the best price will be handle by the spread
        # in the next time the agent updates its orders
        # l_spread_main = self._select_spread(self.state, s_code)
        if s_main_action in ['BUY', 'SELL']:
            self.b_need_hedge = False
            l_msg += self.cancel_all_hedging_orders()
            l_msg += self.translate_action(self.state,
                                           s_main_action,
                                           l_spread=l_spread)
            return l_msg
        return []

    def msgs_due_hedge(self):
        '''
        Return messages given that the agent needs to hedge its positions
        '''
        # check if there are reasons to hedge
        l_aux = self.risk_model.get_instruments_to_hedge(self)
        l_msg = []
        if l_aux:
            # print '\nHedging {} ...\n'.format(self.position['DI1F21'])
            s_, l_spread = self._select_spread(self.state, None)
            s_action, s_instr, i_qty = random.choice(l_aux)
            # generate the messages to the environment
            my_book = self.env.get_order_book(s_instr, False)
            row = {}
            row['order_side'] = ''
            row['order_price'] = 0.0
            row['total_qty_order'] = abs(i_qty)
            row['instrumento_symbol'] = s_instr
            row['agent_id'] = self.i_id
            # check if should send mkt orders in the main instrument
            l_rtn = self.stop_on_main(l_msg, l_spread)
            if len(l_rtn) > 0:
                # print 'stop on main'
                s_time = self.env.order_matching.s_time
                print '{}: Stop loss. {}'.format(s_time, l_aux)
                return l_rtn
            # generate trade and the hedge instruments
            s_time = self.env.order_matching.s_time
            print '{}: Stop gain. {}'.format(s_time, l_aux)
            if s_action == 'BUY':
                self.b_need_hedge = False
                row['order_side'] = 'Buy Order'
                row['order_price'] = my_book.best_ask[0]
                l_msg += self.cancel_all_hedging_orders()
                l_msg += translator.translate_trades_to_agent(row, my_book)
                return l_msg
            elif s_action == 'SELL':
                self.b_need_hedge = False
                row['order_side'] = 'Sell Order'
                row['order_price'] = my_book.best_bid[0]
                l_msg += self.cancel_all_hedging_orders()
                l_msg += translator.translate_trades_to_agent(row, my_book)
                return l_msg
            # generate limit order or cancel everything
            elif s_action == 'BEST_BID':
                f_curr_price, i_qty_book = my_book.best_bid
                l_spread = [0., self.f_spread_to_cancel]
            elif s_action == 'BEST_OFFER':
                f_curr_price, i_qty_book = my_book.best_ask
                l_spread = [self.f_spread_to_cancel, 0.]
            if s_action in ['BEST_BID', 'BEST_OFFER']:
                i_order_size = row['total_qty_order']
                l_msg += translator.translate_to_agent(self,
                                                       s_action,
                                                       my_book,
                                                       # worst t/TOB
                                                       l_spread,
                                                       i_qty=i_order_size)
                return l_msg
        else:
            # if there is not need to send any order, so there is no
            # reason to hedge
            self.b_need_hedge = False
            l_msg += self.cancel_all_hedging_orders()
            return l_msg
        self.b_need_hedge = False
        return l_msg

    def cancel_all_hedging_orders(self):
        '''
        Cancel all hedging orders that might be in the books
        '''
        l_aux = []
        for s_instr in self.risk_model.l_hedging_instr:
            my_book = self.env.get_order_book(s_instr, False)
            f_aux = self.f_spread_to_cancel
            l_aux += translator.translate_to_agent(self,
                                                   None,
                                                   my_book,
                                                   # worst t/TOB
                                                   [f_aux, f_aux])
        return l_aux

    def _select_spread(self, t_state, s_code=None):
        '''
        Select the spread to use in a new order. Return the criterium and
        a list od spread

        :param t_state: tuple. The inputs to be considered by the agent
        '''
        l_spread = [0.0, 0.0]
        s_main = self.env.s_main_intrument
        my_book = self.env.get_order_book(s_main, False)
        # check if it is a valid book
        if abs(my_book.best_ask[0] - my_book.best_bid[0]) <= 1e-6:
            return s_code, [0.02, 0.02]
        elif my_book.best_ask[0] - my_book.best_bid[0] <= -0.01:
            return s_code, [0.15, 0.15]
        # check if should stop to trade
        if self.risk_model.b_stop_trading:
            return s_code, [0.04, 0.04]
        # check if it is time to get agressive due to closing market
        if self.env.order_matching.last_date > STOP_MKT_TIME:
            if self.log_info['pos'][s_main] < -0.01:
                return s_code, [0.0, 0.04]
            elif self.log_info['pos'][s_main] > 0.01:
                return s_code, [0.04, 0.0]
            else:
                return s_code, [0.04, 0.04]
        # change spread
        if not self.risk_model.should_open_at_current_price('ASK', self):
            l_spread[1] = 0.01
        elif not self.risk_model.should_open_at_current_price('BID', self):
            l_spread[0] = 0.01

        # if it just have close a position at the specific side
        if self.env.order_matching.f_time < self.f_time_to_buy:
            l_spread[0] = 0.01
        if self.env.order_matching.f_time < self.f_time_to_sell:
            l_spread[1] = 0.01

        # check if can not open positions due to limits
        if not self.risk_model.can_open_position('ASK', self):
            l_spread[1] = 0.02
        elif not self.risk_model.can_open_position('BID', self):
            l_spread[0] = 0.02

        return s_code, l_spread

    def should_print_logs(self, s_question):
        '''
        Return if should print the log based on s_question:

        :param s_question: string. All or 5MIN
        '''
        if self.b_print_always:
            return True
        if s_question == 'ALL':
            return PRINT_ALL
        elif s_question == '5MIN':
            return PRINT_5MIN
        return False

    def set_to_print_always(self):
        '''
        '''
        self.b_print_always = True


class RandomAgent(QLearningAgent):
    '''
    A RandomAgent representation aims to act as a benchmark to compare to other
    implementations
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True,  b_keep_pos=True):
        '''
        Initiate a RandomAgent object. save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param i_ntoupdate*. float. number of steps to choose a diferent action
        '''
        super(RandomAgent, self).__init__(env, i_id, d_normalizers,
                                          d_ofi_scale, f_min_time, f_gamma,
                                          f_alpha, i_numOfTilings, s_decay_fun,
                                          f_ttoupdate, d_initial_pos,
                                          s_hedging_on, b_hedging, b_keep_pos)
        # Initialize any additional variables here
        self.s_agent_name = 'RandomAgent'

    def apply_policy(self, state, action, reward):
        '''
        Learn policy based on state, action, reward. Return True if the policy
        was updated

        :param state: dictionary. The current state of the agent
        :param action: string. the action selected at this time
        :param reward: integer. the rewards received due to the action
        '''
        # save current state, action
        self.old_state = state
        self.last_action = action
        return True

    def choose_an_action(self, d_state, valid_actions):
        '''
        Return an action from a list of allowed actions according to the
        agent's policy based on epsilon greedy policy and valueFunction

        :param d_state: dictionary. The inputs to be considered by the agent
        :param valid_actions: list. List of the allowed actions
        '''
        # return a uniform random action  with prob $\epsilon_k$ (exploration)
        best_Action = random.choice(valid_actions)
        return best_Action
