#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement an Environment where all agents interact with. The environment is the
limit order book from interest rates future contracts

@author: ucaiado

Created on 11/07/2016
"""
from collections import OrderedDict
import datetime
import json
import logging
import numpy as np
import pandas as pd

from market_gym import Env
from market_gym.config import DEBUG, root, s_log_file
from market_gym.config import START_MKT_TIME, CLOSE_MKT_TIME
from market_gym.lob import BvmfFileMatching
from market_gym.envs.reward_funcs import RewardFunc


'''
Begin help functions
'''


class Foo(Exception):
    """
    Foo is raised by any class to help in debuging
    """
    pass


def filterout_outliers(l_data, l_date):
    '''
    Return a list with data filtered from outliers

    :param l_data: list. data to be analyzed
    '''
    # === old code to filter outliers =======
    # Q3 = np.percentile(l_data, 98)
    # Q1 = np.percentile(l_data, 2)
    # step = (Q3 - Q1) * 1.5
    # step = max(3000., step)
    # na_val = np.array(l_data)
    # na_val = na_val[(na_val >= Q1 - step) & (na_val <= Q3 + step)]
    # return na_val
    # =======================================
    # group by minute
    df_filter = pd.Series(np.array(l_date)/60).astype(int)
    l_filter = list((df_filter != df_filter.shift()).values)
    l_filter[0] = True
    l_filter[-1] = True
    return np.array(pd.Series(l_data)[l_filter].values)


'''
End help functions
'''


class YieldCrvEnv(Env):
    '''
    Yield curve Environment within which all agents operate.
    '''

    def __init__(self, l_fname, l_instrument, NextStopTime, s_main_intrument,
                 l_du, l_pu, l_price_adj, i_idx=None, s_log_fname=None):
        '''
        Initialize an YieldCrvEnv object

        :param l_fname: list. the container zip files to be used in simulation
        :param l_instrument: list. list of instrument to be simulated.
        :param NextStopTime: NextStopTime object. the hour all books is in sync
        :param s_main_intrument: string. The main instrument traded
        :param l_du: list of lists. Days to maturity of each contract
        :param l_pu: list. settlement prices of each instrument in l_du order
        :param l_price_adj: list. settlement rates in l_du order
        :param i_idx*: integer. The index of the start file to be read
        '''
        super(YieldCrvEnv, self).__init__(l_fname, l_instrument, NextStopTime,
                                          s_main_intrument, i_idx, s_log_fname)
        # reward function to be used
        f_err = 'The lentgh of l_du and l_instrument shoould be the same'
        assert len(l_du[0]) == len(l_instrument), f_err
        self.l_du = l_du
        self.l_pu = l_pu
        self.l_price_adj = l_price_adj
        self.reward_fun = RewardFunc()
        self.reward_fun.set_func('pnl')
        self.s_rwd_fun = 'pnl'
        # set the IQR values
        self.set_iqr_of_size()  # NOTE: it should be improved (16/02/17)

    def set_iqr_of_size(self, d_iqr=None):
        '''
        Set the value for IQRs related to the quantities on the books of each
        instrument

        :param d_iqr: dictionary. The IQR pased as float of each instrument
        '''
        self.d_iqr = d_iqr
        if isinstance(d_iqr, type(None)):
            # self.d_iqr = {'DI1F19': 835., 'DI1F21': 320., 'DI1F23': 1455.}
            self.d_iqr = {'DI1F25': 283.75, 'DI1F20': 800., 'DI1F21': 995.,
                          'DI1F23': 445., 'DI1F19': 1570., 'DI1F18': 1145.,
                          'DI1N19': 400., 'DI1N18': 415., 'DI1N20': 420.}

    def _carry_position(self, agent, testing, carry_pos):
        '''
        Return a dictionary with the positions to carry to the next episode

        :param agent: agent object.
        :param testing: boolean.
        :param carry_pos: boolean.
        '''
        d_pos = {}
        l_price_adj = self.l_price_adj[self.order_matching.idx]
        l_opt = self.l_instrument
        for s_instr, f_price_adj in zip(l_opt, l_price_adj):
            self.agent_states[agent][s_instr] = {}
            self.agent_states[agent][s_instr]['Position'] = 0.
            # calculate the last position
            f_pos = agent.position[s_instr]['qBid']
            f_pos -= agent.position[s_instr]['qAsk']
            if s_instr in agent.d_initial_pos:
                f_pos += agent.d_initial_pos[s_instr]['Q']
            # zero variable in env
            for s_key in ['qBid', 'Bid', 'Ask', 'qAsk']:
                self.agent_states[agent][s_instr][s_key] = 0.
            # set up carry position to the next day, if it is the case
            if self.primary_agent:
                if self.primary_agent == agent:
                    if carry_pos and f_pos != 0:
                        # set up d_pos variable properly
                        if s_instr not in d_pos:
                            d_pos[s_instr] = {}
                        d_pos[s_instr]['Q'] = f_pos
                        d_pos[s_instr]['P'] = f_price_adj
        return d_pos

    def _reset_agent_state(self, agent):
        '''
        Return a dictionary of default values to environment representation of
        agents states
        '''
        d_rtn = {'Pnl': 0., 'OFI_Pnl': 0., 'Rwrd_Pnl': 0., 'Agent': agent,
                 'best_bid': False, 'best_offer': False}
        return d_rtn

    def _update_agent_pnl(self, agent, sense, b_isclose=False):
        '''
        Update the agent's PnL and save is on the environment state

        :param agent. Agent object. the agent that will perform the action
        :param sense: dictionary. The inputs from environment to the agent
        '''
        state = self.agent_states[agent]
        f_pnl = 0.
        l_pu = [None for x in self.l_instrument]
        if b_isclose:
            l_pu = self.l_pu[self.order_matching.idx]
        l_du = self.l_du[self.order_matching.idx]
        for s_instr, f_du, f_pu in zip(self.l_instrument, l_du, l_pu):
            f_pu_bid = 0.
            if state[s_instr]['qBid'] != 0:
                f_pu_bid = state[s_instr]['Bid'] / state[s_instr]['qBid']
                f_pu_bid = 10**5 * (1+f_pu_bid/100.)**(-f_du/252.)
            f_pu_ask = 0.
            if state[s_instr]['qAsk'] != 0:
                f_pu_ask = state[s_instr]['Ask'] / state[s_instr]['qAsk']
                f_pu_ask = 10**5 * (1+f_pu_ask/100.)**(-f_du/252.)
            f_pnl += f_pu_bid * state[s_instr]['qBid']
            f_pnl -= f_pu_ask * state[s_instr]['qAsk']
            f_qty = state[s_instr]['qAsk'] + state[s_instr]['qBid']
            f_pu_mid = 0
            if f_pu:
                f_pu_mid = f_pu
            elif sense['midPrice'][s_instr] != 0:
                f_pu_mid = sense['midPrice'][s_instr]
                f_pu_mid = 10**5 * (1+f_pu_mid/100.)**(-f_du/252.)
            f_pnl += -state[s_instr]['Position'] * f_pu_mid
            # include costs
            f_pnl -= (f_qty * 0.80)

        # substitute the last pnl by the current value
        state['Pnl'] = f_pnl
        return f_pnl

    def sense(self, agent):
        '''
        Return the environment state that the agents can access

        :param agent: Agent object. the agent that will perform the action
        '''
        assert agent in self.agent_states, 'Unknown agent!'

        # price related inputs
        d_mid = {}
        d_spread = {}
        d_book = {}
        d_ofi = {}
        d_delta_ofi = {}
        d_highlow = {}
        d_10secRelAll = {}
        d_delta_mid = {}
        d_best_prices = {'agentBid': None, 'agentAsk': None}
        d_ratio = dict(zip(self.l_instrument,
                           [{'BID': {}, 'ASK': {}}
                            for x in self.l_instrument]))
        d_size = dict(zip(self.l_instrument,
                          [{'BID': {}, 'ASK': {}}
                           for x in self.l_instrument]))
        for s_instr in self.l_instrument:
            # get mid prices
            book_aux = self.order_matching.get_order_book_obj(s_instr)
            f_mid_aux = (book_aux.best_ask[0] + book_aux.best_bid[0]) / 2.
            d_mid[s_instr] = f_mid_aux
            # get TOB
            d_book[s_instr] = {'qBid': book_aux.best_bid[1],
                               'Bid': book_aux.best_bid[0],
                               'qAsk': book_aux.best_ask[1],
                               'Ask': book_aux.best_ask[0]}
            # get OFI
            d_ofi[s_instr] = book_aux.stats.get_elapsed_data('all_ofi')
            # get delta OFI
            d_delta_ofi[s_instr] = book_aux.stats.get_elapsed_data('delta_ofi')
            # get relative price
            f_rel_price = book_aux.stats.get_elapsed_data('rel_price')
            d_10secRelAll[s_instr] = f_rel_price
            # get delta mid
            d_delta_mid[s_instr] = book_aux.stats.get_elapsed_data('delta_mid')
            # get high low
            d_highlow[s_instr] = book_aux.stats.get_elapsed_data('high_low')
            # get spread
            f_spread = (book_aux.best_ask[0]-book_aux.best_bid[0])*100
            d_spread[s_instr] = f_spread
            # get ratios
            # f_ratio = (book_aux.best_ask[1]*1./book_aux.best_bid[1])
            f_ratio = (book_aux.best_bid[1] - book_aux.best_ask[1]*1.)
            f_ratio /= (book_aux.best_bid[1] + book_aux.best_ask[1]*1.)
            d_ratio[s_instr]['BID'] = f_ratio
            # f_ratio = (book_aux.best_bid[1]*1./book_aux.best_ask[1])
            d_ratio[s_instr]['ASK'] = f_ratio
            # get size-transformed
            f_size = book_aux.best_bid[1]*1./self.d_iqr[s_instr]
            d_size[s_instr]['BID'] = f_size
            f_size = book_aux.best_ask[1]*1./self.d_iqr[s_instr]
            d_size[s_instr]['ASK'] = f_size
        # get agent best prices
        agent_orders = agent.d_order_tree[self.s_main_intrument]['ASK']
        if agent_orders.count > 0:
            d_best_prices['agentAsk'] = agent_orders.min_key()
        agent_orders = agent.d_order_tree[self.s_main_intrument]['BID']
        if agent_orders.count > 0:
            d_best_prices['agentBid'] = agent_orders.max_key()

        d_rtn = {'qOfi': 0,
                 'qAggr': 0,
                 'qTraded': 0,
                 'spread': d_spread,
                 'qBid': 0.,
                 'qAsk': 0.,
                 'midPrice': d_mid,
                 'size': d_size,
                 'ratio': d_ratio,
                 'TOB': d_book,
                 'OFI': d_ofi,
                 'dOFI': d_delta_ofi,
                 'HighLow': d_highlow,
                 'reallAll': d_10secRelAll,
                 'agentOrders': d_best_prices,
                 'deltaMid': 0.,
                 'deltaMid2': d_delta_mid,
                 'logret': 0.}

        return d_rtn

    def log_trial(self):
        '''
        Log the end of current trial
        '''
        # log when the trial ended
        if self.count_trials > 1:
            i_aux = self.count_trials
            s_msg = 'Environment.log_trial(): Trial Ended.'
            s_msg += ' ID {}\n'.format(i_aux - 1)
            # DEBUG
            logging.info(s_msg)
        # log other informations
        if self.primary_agent:
            a = self.primary_agent
            d_info = self.primary_agent.log_info
            if 'pnl' not in d_info:
                print 'Environment.log_trial(): No PnL to be logged'
                return
            f_pnl = float('{:0.2f}'.format(d_info['pnl']))
            f_duration = float('{:0.2f}'.format(d_info['duration']))
            s_log_name = s_log_file.split('/')[-1].split('.')[0]
            f_tot_traded = a.position[self.s_main_intrument]['qBid']
            f_tot_traded += a.position[self.s_main_intrument]['qAsk']
            f_tot_buy = a.position[self.s_main_intrument]['qBid']

            # log metrics
            f_pnl = float('{:0.2f}'.format(d_info['pnl']))
            na_pnl = filterout_outliers(d_info['pnl_hist'], d_info['time'])

            # calculate the MDD
            f_max = 0.
            l_aux = []
            for f_pnl_aux in na_pnl:
                if f_pnl_aux > f_max:
                    f_max = f_pnl_aux
                l_aux.append(f_max-f_pnl_aux)
            f_mdd = float('{:0.2f}'.format(max(l_aux)))
            f_first = float('{:0.2f}'.format(na_pnl[0]))
            f_q3 = float('{:0.2f}'.format(np.percentile(na_pnl, 75)))
            f_q1 = float('{:0.2f}'.format(np.percentile(na_pnl, 25)))
            f_max_pnl = float('{:0.2f}'.format(max(na_pnl)))
            f_median_pnl = float('{:0.2f}'.format(np.median(na_pnl)))
            f_min_pnl = float('{:0.2f}'.format(min(na_pnl)))
            na_duration = filterout_outliers(d_info['duration_hist'],
                                             d_info['time'])
            f_duration = float('{:0.2f}'.format(d_info['duration']))
            f_max_dur = float('{:0.2f}'.format(max(na_duration)))
            f_min_dur = float('{:0.2f}'.format(min(na_duration)))
            f_avg_dur = float('{:0.2f}'.format(np.mean(na_duration)))
            s_fname = 'log/train_test/results_{}.json'.format(self.s_log_date)
            if self.s_log_fname:
                s_txt = 'log/train_test/results_{}_{}.json'
                s_fname = s_txt.format(self.s_log_fname, self.s_log_date)
            # exclude initial position
            pos = d_info['trades']
            for s_instr in self.l_instrument:
                if s_instr in a.d_initial_pos:
                    f_qty = a.d_initial_pos[s_instr]['Q']
                    f_price = a.d_initial_pos[s_instr]['P']
                    if f_qty < 0:
                        f_qty = abs(f_qty)
                        pos[s_instr]['qAsk'] -= f_qty
                        pos[s_instr]['Ask'] -= f_qty * f_price
                    elif f_qty > 0:
                        pos[s_instr]['qBid'] -= f_qty
                        pos[s_instr]['Bid'] -= f_qty * f_price
            # make string from all number in position
            for s_key in pos:
                for s_key2 in pos[s_key]:
                    f_aux = pos[s_key][s_key2]
                    f_aux = float('{:0.4f}'.format(f_aux))
                    pos[s_key][s_key2] = f_aux
            # do the same to pnl
            f_pnl = float('{:0.2f}'.format(f_pnl))
            # check if it is on training phase
            s_sim_type = 'Train'
            if a.FROZEN_POLICY:
                s_sim_type = 'Test'
            # log in results file
            with open(s_fname, 'a') as f:
                s_fdate = self.order_matching.s_time.split(' ')[0]
                f_tot_reward = d_info['total_reward']
                el = [('Agent', a.s_agent_name),
                      ('File_Date', s_fdate),
                      ('Reward', float('{:0.2f}'.format(f_tot_reward))),
                      ('PnL', {'last': f_pnl,
                               'min': f_min_pnl,
                               # '25%': f_q1,
                               # '50%': f_median_pnl,
                               # '75%': f_q3,
                               'MDD': f_mdd,
                               'first': f_first,
                               'max': f_max_pnl}),
                      ('Duration', {'last': f_duration,
                                    'max': f_max_dur,
                                    'min': f_min_dur,
                                    'avg': f_avg_dur}),
                      ('Position', d_info['pos']),
                      ('Trades', d_info['trades']),
                      ('spread', a.f_spread),
                      ('min_time', a.f_min_time),
                      ('Total_Traded', {'total': f_tot_traded,
                                        'buy': f_tot_buy}),
                      ('Risk_Model', a.risk_model.s_risk_model),
                      ('log_file', s_log_name),
                      ('trial', self.count_trials-1)]
                if a.learning:
                    f_ap = float('{:0.3f}'.format(a.value_function.last_alpha))
                    f_gamma = float('{:0.3f}'.format(a.f_gamma))
                    f_epsilon = float('{:0.3f}'.format(a.f_epsilon))
                    s_rwrd_fun = self.s_rwd_fun
                    s_decay_fun = a.decayfun
                    el.append(('parameters', {'alpha': f_ap,
                                              'gamma':  f_gamma,
                                              'episilon': f_epsilon,
                                              'nsteps': a.n_steps,
                                              'sim': s_sim_type,
                                              'reward_fun': s_rwrd_fun,
                                              'decay_fun': s_decay_fun,
                                              'PR': a.d_pr_mat,
                                              'actions': a.d_actions_acc}))
                # f.write(json.dumps(el, indent=2)+'\n')
                f.write(json.dumps(OrderedDict(el))+'\n')
            # save to the environment
            self.d_trial_data['final_pnl'].append(d_info['pnl'])
            self.d_trial_data['final_duration'].append(d_info['duration'])
            self.d_trial_data['max_pnl'].append(f_max_pnl)
            self.d_trial_data['min_pnl'].append(f_min_pnl)
            self.d_trial_data['final_reward'].append(d_info['total_reward'])
