#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement an basic agent to be used as a framewotk to other implementations

@author: ucaiado

Created on 11/27/2016
"""
import random
import logging
import sys
import time
from bintrees import FastRBTree
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import pickle
import pprint

from market_gym import Agent
from market_gym.envs import Simulator
from market_gym.lob import matching_engine, translator
from market_gym.config import DEBUG, VERBOSE, PRINT_ALL, PRINT_5MIN, root
import risk_model


'''
Begin help functions
'''


'''
End help functions
'''


class BasicAgent(Agent):
    '''
    A Basic agent representation that trades future of interest rate and hedge
    positions when it have been executed. Selects if it stays at a given price
    randomly as well as the immunization instrument (if any decrease its
    currently duration).
    '''
    actions_to_open = [None, 'BEST_BID', 'BEST_OFFER', 'BEST_BOTH']
    actions_to_close_when_short = [None, 'BEST_BID']
    actions_to_close_when_long = [None, 'BEST_OFFER']
    actions_to_stop_when_short = [None, 'BEST_BID', 'BUY']
    actions_to_stop_when_long = [None, 'BEST_OFFER', 'SELL']
    d_translate_to_valuefun = {None: 0, 'BEST_BID': 1, 'BEST_OFFER': 2,
                               'BEST_BOTH': 3, 'BUY': 4, 'SELL': 5}
    d_bets = {'0': -1., '1': 0, '2': 1.}

    FROZEN_POLICY = False

    def __init__(self, env, i_id, f_min_time=3600., f_ttoupdate=0.0,
                 d_initial_pos={}):
        '''
        Initiate a BasicAgent object. save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param f_min_time: float. Minimum time in seconds to the agent react
        :param i_ntoupdate*: float. number of steps to choose a diferent action
        :param f_ttoupdate*: float. time in seconds to choose a diferent action
        '''
        super(BasicAgent, self).__init__(env, i_id)
        # Initialize any additional variables here
        self.k_steps = 0
        self.n_steps = 0
        self.f_min_time = f_min_time
        self.next_time = 0.
        self.next_hedge_time = 0.
        self.max_pos = 40.
        self.orders_lim = 6
        self.order_size = 5
        self.log_info = {'duration': 0., 'total_reward': 0.}
        self.s_agent_name = 'BasicAgent'
        self.last_max_pnl = None
        self.f_pnl = 0.
        self.f_delta_pnl = 0.  # defined at [-inf, 0)
        self.old_state = None
        self.last_action = None
        self.older_code = None  # used to account for precision and recall
        self.cum_reward = 0.
        self.f_spread = 0.01  # cents worst than the best price
        self.last_trade_time = 0.
        self.d_initial_pos = d_initial_pos
        # control hedging
        self.b_need_hedge = False
        self.f_next_stoptime = None
        self.risk_model = risk_model.RiskModel(env)
        self.last_delta_time = None
        self.f_time_to_hedge = 0.005  # in fraction of second (milis)
        self.last_spread = None  # used by scalping
        self.current_code = None
        self.last_hedge_code = None
        self.f_spread_to_cancel = 1.5  # spread to cancel old hedging order
        # control action and policy update
        self.f_timetoupdate = 0.
        self.delta_toupdate = f_ttoupdate
        self.b_new_action = False
        self.logged_action = False
        self.b_has_traded = False
        self.b_new_reward = False
        self.tlast_step_logged = 0.
        self.current_open_price = None
        self.done = False
        self.b_print_always = False
        # dictoinary to fill with precision/recall accountability
        self.d_pr_mat = {}
        self.l_instr_to_bet = self.env.l_instrument
        self.d_bet_idx = {x: i for i, x in enumerate(self.env.l_instrument)}
        self.i_qty_to_be_alone = 5
        # account the actions taken
        action_to_iterate = [None, 'BEST_BID', 'BEST_OFFER', 'BEST_BOTH']
        self.d_actions_acc = dict(zip(action_to_iterate,
                                      [0 for x in action_to_iterate]))

    def reset(self, testing=False):
        '''
        Reset the state and the agent's memory about its positions

        :param testing: boolean. If should freeze policy
        '''
        self.state = None
        # self.k_steps = 0  # maybe I should not zero that
        self.n_steps = 0
        self.position = {}
        self.d_order_tree = {}
        self.d_order_map = {}
        # set new order size
        self.done = False
        if self.env.s_main_intrument in self.d_initial_pos:
            f_pos = abs(self.d_initial_pos[self.env.s_main_intrument]['Q'])
            if f_pos == 0:
                self.order_size = 5
            else:
                self.order_size = max(5, int(f_pos/6))
        # ser position and order tree
        for s_instr in self.env.l_instrument:
            self.position[s_instr] = {'qAsk': 0.,
                                      'Ask': 0.,
                                      'qBid': 0.,
                                      'Bid': 0.}
            if s_instr in self.d_initial_pos:
                f_qty = self.d_initial_pos[s_instr]['Q']
                f_price = self.d_initial_pos[s_instr]['P']
                if f_qty < 0:
                    f_qty = abs(f_qty)
                    self.position[s_instr]['qAsk'] += f_qty
                    self.position[s_instr]['Ask'] += f_qty * f_price
                elif f_qty > 0:
                    self.position[s_instr]['qBid'] += f_qty
                    self.position[s_instr]['Bid'] += f_qty * f_price
            self.d_order_tree[s_instr] = {'BID': FastRBTree(),
                                          'ASK': FastRBTree()}
            self.d_order_map[s_instr] = {}
        self.next_time = 0.
        self.next_hedge_time = 0.
        self.log_info = {'duration': 0., 'total_reward': 0.}
        l_feat_list = ['rwd_hist', 'pnl_hist', 'duration_hist', 'time',
                       'last_inputs_hist', 'features_hist', 'update_hist',
                       'rwd_info_hist']
        for s_feature in l_feat_list:
            self.log_info[s_feature] = []
        self.last_max_pnl = None
        self.f_pnl = 0.
        self.f_delta_pnl = 0.
        self.old_state = None
        self.last_action = None
        self.older_code = None
        self.cum_reward = 0.
        self.b_need_hedge = False
        self.f_next_stoptime = None
        self.last_delta_time = None
        self.last_trade_time = 0.
        self.f_timetoupdate = 0.
        self.b_new_action = False
        self.logged_action = False
        self.tlast_step_logged = 0.
        self.last_spread = None
        self.current_code = None
        self.last_hedge_code = None
        self.b_has_traded = False
        self.b_new_reward = False
        # account the actions taken
        action_to_iterate = [None, 'BEST_BID', 'BEST_OFFER', 'BEST_BOTH']
        self.d_actions_acc = dict(zip(action_to_iterate,
                                      [0 for x in action_to_iterate]))
        # Reset any variables here, if required
        self.reset_additional_variables(testing)

    def reset_additional_variables(self, testing):
        '''
        Reset aditional variables
        '''
        pass

    def additional_actions_when_exec(self, s_instr, s_side, msg):
        '''
        Execute additional action when execute a trade

        :param s_instr: string.
        :param s_side: string.
        :param msg: dictionary. Last trade message
        '''
        # check if the main intrument was traded
        s_main = self.env.s_main_intrument
        # NOTE: include more conditions to check if it open or close a pos
        if msg['instrumento_symbol'] == s_main:
            self.b_has_traded = True
        # log more information
        l_info_to_hold = [msg['order_price'], None]
        if 'last_inputs' in self.log_info:
            l_info_to_hold[1] = self.log_info['last_inputs']['TOB']
        self.d_trades[s_instr][s_side].append(l_info_to_hold)

    def update(self, msg_env):
        '''
        Update the state of the agent

        :param msg_env: dict. A message generated by the order matching
        '''
        # check if should update, if it is not a trade
        # if not isinstance(msg_env, type(None)):
        if not msg_env:
            if not self.should_update():
                return None
        # recover basic infos
        inputs = self.env.sense(self)
        state = self.env.agent_states[self]
        s_cmm = self.env.s_main_intrument

        # Update state (position ,volume and if has an order in bid or ask)
        self.state = self.get_intern_state(inputs, state)

        # Select action according to the agent's policy
        s_action = None
        s_action, l_msg = self.take_action(self.state, msg_env)
        s_action2 = s_action

        # Execute action and get reward
        reward = 0.
        self.env.update_order_book(l_msg)
        l_prices_to_print = []
        if len(l_msg) == 0:
            reward += self.env.act(self, None)
            self.b_new_reward = False
        for msg in l_msg:
            if msg['agent_id'] == self.i_id:
                # check if should hedge the position
                self.should_change_stoptime(msg)
                # form log message
                s_action = msg['action']
                s_action2 = s_action
                s_side_msg = msg['order_side'].split()[0]
                s_indic = msg['agressor_indicator']
                s_cmm = msg['instrumento_symbol']
                d_aux = {'A': msg['order_status'],
                         # log just the last 4 digits of the order
                         'I': msg['order_id'] % 10**4,
                         'Q': msg['order_qty'],
                         'C': msg['instrumento_symbol'],
                         'S': s_side_msg,
                         'P': '{:0.2f}'.format(msg['order_price'])}
                l_prices_to_print.append(d_aux)
                # l_prices_to_print.append('{:0.2f}'.format(msg['order_price']))
                if s_indic == 'Agressor' and s_action == 'SELL':
                    s_action2 = 'HIT'  # hit the bid
                elif s_indic == 'Agressor' and s_action == 'BUY':
                    s_action2 = 'TAKE'  # take the offer
                try:
                    # the agent's positions and orders list are update here
                    # TODO: The reward really should be collect at this point?
                    reward += self.env.act(self, msg)
                    self.b_new_reward = False
                except:
                    print 'BasicAgent.update(): Message with error at reward:'
                    pprint.pprint(msg)
                    raise

        # check if should cancel any order due to excess
        l_msg1 = self.could_include_new(s_action)
        self.env.update_order_book(l_msg1)
        for msg in l_msg1:
            if msg['agent_id'] == self.i_id:
                s_indic = msg['agressor_indicator']
                d_aux = {'A': msg['order_status'],
                         'I': msg['order_id'],
                         'C': msg['instrumento_symbol'],
                         'S': msg['order_side'].split()[0],
                         'P': '{:0.2f}'.format(msg['order_price'])}
                l_prices_to_print.append(d_aux)
                try:
                    # the agent's positions and orders list are update here
                    # there is no meaning in colecting reward here
                    self.env.act(self, msg)
                except:
                    print 'BasicAgent.update(): Message with error at reward:'
                    pprint.pprint(msg)
                    raise
        # === DEBUG ====
        # if len(l_msg1) > 0:
        #     print '\n====CANCEL ORDER DUE TO EXCESS======\n'
        #     pprint.pprint(l_msg1)
        # ==============

        # NOTE: I am not sure about that, but at least makes sense... I guess
        # I should have to apply the reward to the action that has generated
        # the trade (when my order was hit, I was in the book before)
        if s_action2 == s_action:
            if s_action == 'BUY':
                s_action = 'BEST_BID'
            elif s_action == 'SELL':
                s_action = 'BEST_OFFER'
        if s_action in ['correction_by_trade', 'crossed_prices']:
            if s_side_msg == 'Buy':
                s_action = 'BEST_BID'
            elif s_side_msg == 'Sell':
                s_action = 'BEST_OFFER'
        # Learn policy based on state, action, reward
        if s_cmm == self.env.s_main_intrument:
            if self.policy_update(self.state, s_action, reward):
                self.k_steps += 1
                self.n_steps += 1
                # print 'new step: {}\n'.format(self.n_steps)
        # calculate the next time that the agent will react
        if not isinstance(msg_env, type(dict)):
            self.next_time = self.env.order_matching.last_date
            f_delta_time = self.f_min_time
            # add additional miliseconds to the next_time to act
            if self.f_min_time > 0.004:
                if np.random.rand() > 0.4:
                    i_mult = 1
                    if np.random.rand() < 0.5:
                        i_mult = -1
                    f_add = min(1., self.f_min_time*100)
                    f_add *= np.random.rand()
                    f_delta_time += (int(np.ceil(f_add))*i_mult)/1000.
            self.next_time += f_delta_time
            self.last_delta_time = int(f_delta_time * 1000)

        # print agent inputs
        self._pnl_information_update()
        self.log_step(state, inputs, s_action2, l_prices_to_print, reward)

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
        # d_data['BOOK_RATIO'] = inputs['qBid'] * 1. / inputs['qAsk']
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']

        # i_cluster = self.scaler.transform(d_data)
        i_cluster = 0
        d_rtn = {}
        d_rtn['cluster'] = i_cluster
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']

        return d_rtn

    def take_action(self, t_state, msg_env):
        '''
        Return a list of messages according to the agent policy and the action
        choosen as a string

        :param t_state: tuple. The inputs to be considered by the agent
        :param msg_env: dict. Order matching message
        '''
        # check if have occured a trade
        if msg_env:
            # if msg_env['order_status'] in ['Filled', 'Partialy Filled']:
            # check if should hedge the position
            self.should_change_stoptime(msg_env)
            return msg_env['action'], [msg_env]
        # select a randon action, but not trade more than the maximum position
        valid_actions = self.get_valid_actions()
        # NOTE: I should change just this function when implementing
        # the learning agent
        s_action = self.which_action(t_state, valid_actions)
        # build a list of messages based on the action taken
        # this method will always generate orders in the main instrument
        l_msg2 = self.translate_action(t_state, s_action)
        return s_action, l_msg2

    def get_valid_actions(self):
        '''
        Return a list of valid actions based on the current position
        '''
        s_main = self.env.s_main_intrument
        valid_actions = list(self.actions_to_open)
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        if f_pos <= (self.max_pos * -1):
            valid_actions = list(self.actions_to_close_when_short)  # copy
        elif f_pos >= self.max_pos:
            valid_actions = list(self.actions_to_close_when_long)
        return valid_actions

    def translate_action(self, t_state, s_action, l_spread=None):
        '''
        Translate the action taken into messaged to environment in the main
        instrument

        :param t_state: tuple. The inputs to be considered by the agent
        :param s_action: string. The action taken
        :param l_spread*: list. Spread to use to place orders
        '''
        my_book = self.env.get_order_book(self.env.s_main_intrument,
                                          b_rtn_dataframe=False)
        row = {}
        row['order_side'] = ''
        row['order_price'] = 0.0
        row['total_qty_order'] = self.order_size
        row['instrumento_symbol'] = self.env.s_main_intrument
        row['agent_id'] = self.i_id
        my_book = self.env.get_order_book(self.env.s_main_intrument,
                                          b_rtn_dataframe=False)
        # generate trade
        if s_action == 'BUY':
            row['order_side'] = 'Buy Order'
            row['order_price'] = my_book.best_ask[0]
            return translator.translate_trades_to_agent(row, my_book)
        elif s_action == 'SELL':
            row['order_side'] = 'Sell Order'
            row['order_price'] = my_book.best_bid[0]
            return translator.translate_trades_to_agent(row, my_book)
        # generate limit order or cancel everything
        else:
            # continue
            if not l_spread:
                s_current_code = self.current_code
                s_aux, l_spread = self._select_spread(t_state, s_current_code)
            return translator.translate_to_agent(self,
                                                 s_action,
                                                 my_book,
                                                 # worst t/TOB
                                                 l_spread)
        return []

    def policy_update(self, state, action, reward):
        '''
        Check if should return the policy and apply it

        :param state: dictionary. The current state of the agent
        :param action: string. the action selected at this time
        :param reward: integer. the rewards received due to the action
        '''
        self.cum_reward += reward
        if self.b_new_action:
            f_cum_reward = self.cum_reward
            # debug
            # s_err = '\n\nREWARD PASSED TO THE Q-TABLE: {}\n\n'
            # print s_err.format(f_cum_reward)
            self.cum_reward = 0.
            self.b_new_action = False
            return self.apply_policy(state, action, f_cum_reward)
        return False

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

    def which_action(self, t_state, valid_actions):
        '''
        Check if should return a new action or return the last one

        :param valid_actions: list. List of the allowed actions
        :param t_state: tuple. The inputs to be considered by the agent
        '''
        self.b_new_action = False
        # if (self.i_ntoupdate >= self.max_toupdate) or self.FROZEN_POLICY:
        f_mkt_time = self.env.order_matching.last_date
        b_mkt_time = (f_mkt_time > self.f_timetoupdate)
        # if b_mkt_time or self.FROZEN_POLICY or self.b_has_traded:
        if b_mkt_time or self.b_has_traded:
            # self.i_ntoupdate = 1
            self.f_timetoupdate = f_mkt_time + self.delta_toupdate
            self.b_new_action = True
            self.logged_action = True
            self.b_has_traded = False
            self.b_new_reward = True
            self.older_code = self.current_code
            self.current_code, self.last_spread = self._select_spread(t_state)
            return self.choose_an_action(t_state, valid_actions)
        s_, l_aux = self._select_spread(t_state, s_code=self.current_code)
        self.last_spread = l_aux
        return self.last_action

    def choose_an_action(self, t_state, valid_actions):
        '''
        Return an action from a list of allowed actions according to the
        agent policy

        :param valid_actions: list. List of the allowed actions
        :param t_state: tuple. The inputs to be considered by the agent
        '''
        return random.choice(valid_actions)

    def need_to_hedge(self):
        '''
        Return if the agent need to hedge position
        '''
        if abs(self.log_info['duration']) > 1.:
            self.b_need_hedge = True
        return self.b_need_hedge

    def hedge_position(self):
        '''
        Hedge Open positions. This method is called by environment when it
        has already passed need_to_hedge() checks
        '''
        if self.env.order_matching.last_date > self.next_hedge_time:
            # after the first stop time to hedge, the agent can change hedge
            # strategy in 2 milis
            self.next_hedge_time = self.env.order_matching.last_date + 0.002
            # if self.need_to_hedge():
            #     return self.msgs_due_hedge()
            #
            return self.msgs_due_hedge()

        return []

    def msgs_due_hedge(self):
        '''
        Return messages given that the agent needs to hedge its positions
        '''
        # self.b_need_hedge = False
        l_aux = self.risk_model.get_instruments_to_hedge(self)
        if l_aux:
            s_action, s_instr, i_qty = random.choice(l_aux)
            # generate the messages to the environment
            my_book = self.env.get_order_book(s_instr, False)
            # s_action = 'BUY'
            # if i_qty < 0:
            #     s_action = 'SELL'
            row = {}
            row['order_side'] = ''
            row['order_price'] = 0.0
            row['total_qty_order'] = abs(i_qty)
            row['instrumento_symbol'] = s_instr
            row['agent_id'] = self.i_id
            # generate trade
            if s_action == 'BUY':
                self.b_need_hedge = False
                row['order_side'] = 'Buy Order'
                row['order_price'] = my_book.best_ask[0]
                return translator.translate_trades_to_agent(row, my_book)
            elif s_action == 'SELL':
                self.b_need_hedge = False
                row['order_side'] = 'Sell Order'
                row['order_price'] = my_book.best_bid[0]
                return translator.translate_trades_to_agent(row, my_book)
            # generate limit order or cancel everything
            elif s_action == 'BEST_BID':
                l_spread = [0.0, self.f_spread_to_cancel]
            elif s_action == 'BEST_OFFER':
                l_spread = [self.f_spread_to_cancel, 0.0]
            if s_action in ['BEST_BID', 'BEST_OFFER']:
                return translator.translate_to_agent(self,
                                                     s_action,
                                                     my_book,
                                                     # worst t/TOB
                                                     l_spread)
        else:
            f_aux = self.f_spread_to_cancel
            self.b_need_hedge = False
            l_msgs = []
            if not self.risk_model:
                return l_msgs
            for s_instr in self.risk_model.l_hedging_instr:
                i_count = self.d_order_tree[s_instr]['ASK'].count
                i_count += self.d_order_tree[s_instr]['BID'].count
                if i_count > 0:
                    my_book = self.env.get_order_book(s_instr, False)
                    l_msgs += translator.translate_to_agent(self,
                                                            None,
                                                            my_book,
                                                            # worst t/TOB
                                                            [f_aux, f_aux])
            return l_msgs
        self.b_need_hedge = False
        return []

    def freeze_policy(self):
        '''
        Freeze agent's policy so it will not update the qtable in simulation
        '''
        self.FROZEN_POLICY = True
        s_print = '{}.freeze_policy(): Policy has been frozen !'
        s_print = s_print.format(self.s_agent_name)
        root.debug(s_print)

    def should_update(self):
        '''
        Return a boolean informing if it is time to update the agent
        '''
        if self.env.i_nrow < 5:
            return False
        return self.env.order_matching.last_date >= self.next_time

    def greater_than_tradetime(self, f_milis):
        '''
        Check if the current time is greater than the last trade time

        :param f_milies: float. time in miliseconds
        '''
        f_aux = self.last_trade_time + f_milis
        return self.env.order_matching.last_date >= f_aux

    def should_change_stoptime(self, msg):
        '''
        Check if should change the stoptime to hedge position

        :param msg: dict. Message from the environment
        '''
        if not self.env.NextStopTime.has_already_used_param():
            return False
        if msg['instrumento_symbol'] != self.env.s_main_intrument:
            return False
        if msg['order_status'] not in ['Partially Filled', 'Filled']:
            return False
        # if it was executed, should check if need hedge
        self.b_need_hedge = True
        # check if should change the stoptime
        l_books = self.env.order_matching.l_order_books
        l_instruments = self.env.order_matching.l_instrument
        l_ftime = [x.f_time for x in l_books]
        l_fstoptime = [x.last_stop_time for x in l_books]
        # NOTE: easier to check if the books is updating using larger milis
        f_next_time = max(l_ftime) + self.f_time_to_hedge
        self.last_trade_time = max(l_ftime)
        if f_next_time > max(l_fstoptime):
            return False
        # calculate the new stoptime
        i_hour = int(f_next_time/3600)
        i_min = int((f_next_time-i_hour*3600)/60)
        i_sec = int((f_next_time-i_hour*3600 - i_min*60))
        i_mil = (f_next_time - i_hour*3600 - i_min * 60 - i_sec)
        i_mil = int(np.round(i_mil*1000))
        s_aux = self.env.NextStopTime.set_stoptime(i_hour, i_min,
                                                   i_sec, i_mil)
        self.f_next_stoptime = f_next_time
        self.env.order_matching.s_stoptime = s_aux
        return True

    def could_include_new(self, s_action):
        '''
        Check if should reduce the number of orders on the LOB and return
        additional messages to cancel the excess if needed

        :param s_action: string. The next action of the agent
        '''
        l_msg = []
        i_n_to_cancel = min(3, self.orders_lim/2)  # NOTE: RAISE THAT
        if i_n_to_cancel == 0:
            return l_msg
        # cancel orders in excess if it is needed
        s_instr = self.env.s_main_intrument
        if s_action in ['BEST_OFFER', 'BEST_BOTH']:
            i_ask_tot = self.d_order_tree[s_instr]['ASK'].count
            if i_ask_tot > self.orders_lim:
                return translator.translate_cancel_to_agent(self,
                                                            s_instr,
                                                            s_action,
                                                            'ASK',
                                                            i_n_to_cancel)
        if s_action in ['BEST_BID', 'BEST_BOTH']:
            i_bid_tot = self.d_order_tree[s_instr]['BID'].count
            if i_bid_tot > self.orders_lim:
                return translator.translate_cancel_to_agent(self,
                                                            s_instr,
                                                            s_action,
                                                            'BID',
                                                            i_n_to_cancel)
        return l_msg

    def _pnl_information_update(self):
        '''
        Update Pnl informations
        '''
        f_delta_pnl = 0.
        s_main = self.env.s_main_intrument
        f_pnl = self.env.agent_states[self]['Pnl']
        self.f_pnl = f_pnl
        # check the last maximum pnl considering just the current position
        if self.env.agent_states[self][s_main]['Position'] == 0:
            self.last_max_pnl = None
        else:
            self.last_max_pnl = max(self.last_max_pnl,
                                    self.env.agent_states[self]['Pnl'])
            f_delta_pnl = f_pnl - self.last_max_pnl
            self.f_delta_pnl = f_delta_pnl

    def _select_spread(self, t_state, s_code=None):
        '''
        Select the spread to use in a new order. Return the code criterium
        to select the spread and a list of the bid and ask spread

        :param t_state: tuple. The inputs to be considered by the agent
        '''
        return None, self.f_spread

    def account_precision_recall(self, s_change):
        '''
        Account for precision and recall

        :param d_change: dictionary. change in the mid prices in the last step
        '''
        pass

    def should_print_logs(self, s_question):
        '''
        Return if should print the log based on s_question:

        :param s_question: string. All or 5MIN
        '''
        if s_question == 'ALL':
            return PRINT_ALL
        elif s_question == '5MIN':
            return PRINT_5MIN
        return False

    def log_step(self, state, inputs, s_action2, l_prices_to_print, reward):
        '''
        Log the current action/update from agent to a file or just to terminal

        :param args*: Inputs to be used in the log string.
        '''
        # drop input entries
        # s_main = self.env.s_main_intrument
        # for s_key in inputs['midPrice']:
        #     s = '{:0.3f}'
        #     inputs['midPrice'][s_key] = s.format(inputs['midPrice'][s_key])
        d_pos = {}
        for s_key in inputs['midPrice']:
            d_pos[s_key] = int(state[s_key]['Position'])
        l_drop = ['deltaMid', 'logret', 'qAggr', 'qTraded', 'qBid', 'qAsk',
                  'qOfi', 'spread', 'size', 'ratio', 'midPrice', 'deltaMid2',
                  'reallAll', 'HighLow']
        inputs_to_print = {}
        for s_key in inputs:
            if s_key not in l_drop:
                inputs_to_print[s_key] = inputs[s_key]
        if self.current_open_price:
            inputs_to_print['open_price'] = self.current_open_price
        # for s_key in l_drop:
        #     inputs.pop(s_key)
        if self.last_spread:
            inputs['last_spread'] = self.last_spread
        # print agent inputs
        f_duration = self.risk_model.portfolio_duration(self.position)
        f_pnl = self.f_pnl
        f_delta_pnl = self.f_delta_pnl
        s_main = self.env.s_main_intrument
        s_date = self.env.order_matching.s_time
        f_date = self.env.order_matching.f_time

        s_rtn = '{}.update(): time = {}, action = {:12s}, pnl = {:0.2f}'
        s_rtn += ', reward = {:0.2f}, duration = {:0.2f}, position = {}'
        s_rtn += ', time_to_update = {} , code = {}, deltas = {}, inputs = {}'
        s_rtn += ', msgs_to_env = {}'
        s_txt_to_add = ''
        if VERBOSE:
            s_need_to_hedge = 'y' if self.need_to_hedge() else 'n'
            s_txt_to_add = ', need_hedge = {}'.format(s_need_to_hedge)
            s_txt_to_add += ', features = {}'.format(self.state['features'])
        # save infos to log_info (uncomment that to study feat distribution)
        # self.log_info['duration_hist'].append(f_duration)
        # self.log_info['pnl_hist'].append(f_pnl)
        # self.log_info['rwd_hist'].append(self.log_info['total_reward'])
        # self.log_info['last_inputs_hist'].append(inputs)
        # self.log_info['features_hist'].append(self.state['features'])
        # self.log_info['time'].append(f_date)

        self.log_info['pos'] = d_pos
        self.log_info['duration'] = f_duration
        self.log_info['pnl'] = f_pnl
        self.log_info['reward'] = reward
        self.log_info['total_reward'] += reward
        self.log_info['trades'] = self.position
        self.log_info['last_inputs'] = inputs
        if 'rwd' in self.env.reward_fun.log_info:
            d_aux = self.env.reward_fun.log_info['rwd'].copy()
            d_aux2 = self.env.reward_fun.log_info.copy()
            self.log_info['last_rwd_info'] = d_aux2
            # self.log_info['rwd_info_hist'].append(d_aux)
        f_delta_time = self.last_delta_time
        # check if it selected a new action in the last
        s_time_to_update = 'n'
        d_deltas = {}
        if self.logged_action:
            self.logged_action = False
            s_time_to_update = 'y'
            # compare with the last data
            if 'to_delta' in self.log_info:
                # measure the change in OFI from he last sction taken
                d_deltas['OFI'] = {}
                for s_key in inputs['dOFI']:
                    d_deltas['OFI'][s_key] = inputs['dOFI'][s_key]
                # account the reward accumulated from the last action taken
                f_rwrd_now = self.log_info['total_reward']
                f_rwrd_old = self.log_info['to_delta']['Rwrd']
                f_aux = float('{:0.2f}'.format(f_rwrd_now - f_rwrd_old))
                d_deltas['Reward'] = f_aux
                # calculate the direction of the market
                l_instruments = self.l_instr_to_bet
                # l_instruments = self.env.l_instrument
                l_price_change = [1] * len(l_instruments)
                for idx, s_key in enumerate(l_instruments):
                    # calculate old mid-price
                    f_mid_old = self.log_info['to_delta']['TOB'][s_key]['Bid']
                    f_mid_old += self.log_info['to_delta']['TOB'][s_key]['Ask']
                    f_mid_old /= 2.
                    # calculate new mid-price
                    f_mid_new = inputs['TOB'][s_key]['Bid']
                    f_mid_new += inputs['TOB'][s_key]['Ask']
                    f_mid_new /= 2.
                    # account the side of the change
                    if f_mid_new > f_mid_old:
                        l_price_change[idx] = 2
                    elif f_mid_new < f_mid_old:
                        l_price_change[idx] = 0
                d_deltas['Change'] = ''.join(str(x) for x in l_price_change)
                # account the actions taken
                if self.last_action not in ['BUY', 'SELL']:
                    self.d_actions_acc[self.last_action] += 1
                # account precision and recall
                # NOTE: maybe i should pass a function here to be used just by
                # new implementations
                self.account_precision_recall(d_deltas['Change'])

            # hold information from this step to use in the next one
            self.log_info['to_delta'] = {'TOB': inputs['TOB'],
                                         'OFI': inputs['OFI'],
                                         'Rwrd': self.log_info['total_reward']}
            # append values
            self.log_info['duration_hist'].append(f_duration)
            self.log_info['pnl_hist'].append(f_pnl)
            self.log_info['rwd_hist'].append(self.log_info['total_reward'])
            self.log_info['last_inputs_hist'].append(inputs)
            self.log_info['features_hist'].append(self.state['features'])
            self.log_info['time'].append(f_date)

            self.log_info['update_hist'].append(s_time_to_update)
            if 'rwd' in self.env.reward_fun.log_info:
                d_aux = self.env.reward_fun.log_info['rwd'].copy()
                self.log_info['rwd_info_hist'].append(d_aux)

        # Print inputs and agent state
        # self.log_info['update_hist'].append(s_time_to_update)
        # DEBUG
        s_txt_to_add
        s_rtn = s_rtn.format(self.s_agent_name, s_date, s_action2, f_pnl,
                             reward, f_duration, d_pos, s_time_to_update,
                             {'C': self.current_code,
                              'H': self.last_hedge_code},
                             d_deltas, inputs_to_print, l_prices_to_print)
        s_rtn += s_txt_to_add + '\n'
        if self.should_print_logs('ALL'):
            if self.should_print_logs('5MIN'):
                f_time = self.env.order_matching.last_date
                b_t1 = s_action2 == 'MKT_CLOSED'
                b_t2 = f_time > self.tlast_step_logged
                if b_t2:
                    self.tlast_step_logged = self.env.order_matching.last_date
                    self.tlast_step_logged += (5. * 60)
                # uncomment this to log all order changes
                # b_t3 = len(l_prices_to_print) > 0
                # uncomment this to log just the filled orders
                b_t3 = s_action2 in ['correction_by_trade', 'crossed_prices',
                                     'TAKE', 'HIT', 'BUY', 'SELL']
                if b_t1 or b_t2 or b_t3 or self.env.done:
                    root.debug(s_rtn)
            else:
                root.debug(s_rtn)
        else:
            if s_action2 == 'MKT_CLOSED' or self.env.done:
                root.debug(s_rtn)
