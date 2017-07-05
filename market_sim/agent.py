#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Run one of the agents implemented in agents/ to trade Brazilian interest rate
future contracts

@author: ucaiado

Created on 11/21/2016
"""
import argparse
import textwrap
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
from market_gym.envs import make, Simulator
from market_gym.lob import matching_engine, translator
import market_gym.utils.di_utilities as di_utilities
import market_gym.config
from market_gym.config import DEBUG, WEIGHTING_TIME, PRINT_ON_TEST, root
# import agent tp use in simulations
from _agents import BasicAgent
from _agents import QLearningAgent, RandomAgent
from _agents import dissertation_tests


'''
Begin help functions and variables
'''
d_weighting = {10.: 'data/misc/ofi_ecdf.dat', 30.: 'data/misc/ofi_ecdf2.dat'}
OFI_SCALE = pickle.load(open(d_weighting[WEIGHTING_TIME], 'r'))


def scale_ofi(f_val, s_cmm):
    '''
    Scale ofi according to the commoditie passed
    '''
    f_val2 = abs(f_val)
    # f_val2 = 0.
    # if f_val != 0.:
    #     f_val2 = np.log(abs(f_val))
    f_rtn = OFI_SCALE[s_cmm](f_val2)
    return f_rtn * np.sign(f_val)

NORMALIZERS = {'position': {'MIN': -50., 'MAX': 50.},
               'duration': {'MIN': -20., 'MAX': 20.},
               'ofi_new': {'MIN': -1., 'MAX': 1., 'FUN': scale_ofi},
               'ofi': {'MIN': -500., 'MAX': 500.},
               'hour': {'MIN': 8., 'MAX': 17.},
               # number of ticks
               'spread_longo': {'MIN': 1., 'MAX': 5.},
               'spread_curto': {'MIN': 1., 'MAX': 5.},
               # ratio between bid and ask
               # 'ratio_longo': {'MIN': 0.01, 'MAX': 50., 'FUN': np.log},
               # 'ratio_curto': {'MIN': 0.01, 'MAX': 150., 'FUN': np.log},
               'ratio_longo': {'MIN': -1., 'MAX': 1.},
               'ratio_curto': {'MIN': -1., 'MAX': 1.},
               # size is in numbers of IQRs
               'size_ask_longo': {'MIN': 0.01, 'MAX': 3.5},
               'size_ask_curto': {'MIN': 0.01, 'MAX': 3.5},
               'size_bid_longo': {'MIN': 0.01, 'MAX': 3.5},
               'size_bid_curto': {'MIN': 0.01, 'MAX': 3.5},
               # relative prices and highlow
               'rel_price': {'MIN': 0., 'MAX': 4.},
               'high_low': {'MIN': 0., 'MAX': 50}}

OFI_DICT = {'DI1F25': 283.75, 'DI1F20': 800., 'DI1F21': 580., 'DI1F23': 335.,
            'DI1F19': 650., 'DI1F18': 1145., 'DI1N19': 400., 'DI1N18': 415.}

# the key is the date where the valuefunc will be used, but the func is from
# the previous day

VALUE_FUNCS = {'F21': {'20170215': 'sim_Thu_May__4_113121_2017_valuefun_100',
                       '20170221': 'sim_Mon_May__8_041745_2017_valuefun_100',
                       'PARAM': {'GAIN': 5000., 'LOSS': -5000.}},
               'F19': {'20170215': 'sim_Mon_May__8_042235_2017_valuefun_100',
                       '20170221': 'sim_Mon_May__8_042424_2017_valuefun_100',
                       'PARAM': {'GAIN': 5000., 'LOSS': -5000.}},
               'F23': {'20170215': 'sim_Mon_May__8_042110_2017_valuefun_100',
                       '20170221': 'sim_Mon_May__8_042443_2017_valuefun_100',
                       'PARAM': {'GAIN': 5000., 'LOSS': -5000.}}}


class InvalidOptionException(Exception):
    """
    InvalidOptionException is raised by the run() function and indicate that no
    valid test option was selected
    """
    pass

'''
End help functions
'''


class Run(object):
    '''
    '''

    d_available_data = {201702: ['20170201', '20170202', '20170203',
                                 '20170206', '20170207', '20170208',
                                 '20170209', '20170210', '20170213',
                                 '20170214', '20170215', '20170216',
                                 '20170217', '20170220', '20170221',
                                 '20170222', '20170223', '20170224'],
                        201703: ['20170301', '20170302', '20170303',
                                 '20170306', '20170307', '20170308',
                                 '20170309', '20170310', '20170313',
                                 '20170314', '20170315', '20170316',
                                 '20170320', '20170321', '20170322',
                                 '20170323', '20170324', '20170327',
                                 '20170328', '20170330', '20170331'],
                        201701: ['20170102', '20170103', '20170104',
                                 '20170105', '20170106', '20170109',
                                 '20170110', '20170111', '20170112',
                                 '20170113', '20170116', '20170117',
                                 '20170118', '20170119', '20170120',
                                 '20170123', '20170124', '20170126',
                                 '20170127', '20170131'],
                        201612: ['20161201', '20161207', '20161208',
                                 '20161209', '20161212', '20161213',
                                 '20161214', '20161216', '20161219',
                                 '20161220', '20161222', '20161223',
                                 '20161226', '20161227', '20161228',
                                 '20161229'],
                        201611: ['20161101', '20161103', '20161104',
                                 '20161107', '20161108', '20161110',
                                 '20161116', '20161117', '20161118',
                                 '20161121', '20161122', '20161123',
                                 '20161124', '20161128', '20161129',
                                 '20161130']}

    def __init__(self):
        '''
        '''
        if not DEBUG:
            s_smg = '\n\n=======================\n=== LOG NOT ENABLED ===\n'
            s_smg += '=======================\n\n'
            print s_smg
            time.sleep(5)
        self.d_experiments = self.set_experiments_options()
        self.s_log_name = None
        s_root = 'data/preprocessed/'

    def set_log_name(self, s_log_name):
        '''
        ...

        :param s_log_name: string. Name to append to the log file
        '''
        self.s_log_name = s_log_name

    def set_experiments_options(self):
        '''
        '''
        d_experiments = {'_qlearning': self._qlearning,
                         '_domino_q': self._domino_q,
                         '_random': self._random,
                         '_domino_rand': self._domino_rand}
        return d_experiments

    def run(self, s_option, **argv):
        '''
        Run the simulation according to the option selected and return the
        agent created to the simulation

        :param s_option: string. The kind of the test to run
        :param i_trials*: integer. number of trials to run
        '''
        self.argv = argv
        # check if was passed a new root
        s_root = self._get_from_argv('s_root')
        if not s_root:
            s_root = 'data/preprocessed/'
        self.s_root = s_root
        # check if was passed number of trials
        n_trials = self._get_from_argv('i_trials')
        if not n_trials:
            n_trials = 1
        # if you not want to run in different days, tou should specify the date
        self.s_date = self._get_from_argv('s_date')
        # time to run the agent
        i_milis = self._get_from_argv('i_milis')
        if not i_milis:
            i_milis = 10
        self.i_milis = i_milis
        # check the number of sessions to run
        i_sess = self._get_from_argv('i_sess')
        if not i_sess:
            i_sess = 1
        self.i_sess = i_sess
        # if i_sess != 1:
        #     self.s_date = None
        # which intruments compose the market
        l_opt = self._get_from_argv('l_instruments')
        if not l_opt:
            l_opt = ['DI1F21', 'DI1F19', 'DI1F23']
        self.l_opt = l_opt

        # check the month of the data to use
        i_month = self._get_from_argv('i_month')
        if not i_month:
            i_month = 201702
        if self.s_date:
            i_month = int(self.s_date[:6])

        # check if should render the LOBs
        b_render = self._get_from_argv('b_render')

        # set up the environment and agent to run the simulation
        s_main_instr = self._get_from_argv('s_main_instr')
        if not s_main_instr:
            s_main_instr = 'DI1F21'
            if s_main_instr not in l_opt:
                s_err = 'select an instrument as s_main_instr'
                raise InvalidOptionException(s_err)
        e = self.set_environment(n_trials, s_main_instr, i_month,
                                 n_sessions=i_sess)
        a = self.set_agent(s_option, e)

        #######################
        # Follow the trading agent
        e.set_primary_agent(a)  # specify agent to track

        #######################
        # Create the simulation
        # flags:
        #    env            - Environment Object. The environment simulated
        #    * update_delay - Float. Seconds elapsed to print out the book
        #    * display      - Boolean. If should open a visualizer
        if b_render:
            sim = Simulator(e, update_delay=1.00, display=True)
        else:
            sim = Simulator(e, update_delay=1.00, display=False)

        #######################
        # Run the simulator
        # flags:
        #    * s_qtable   - string. path to the qtable to be used
        #    * n_start    - integer. start file to use in simulation
        #    * n_sessions - integer. Number of files to read
        #    * n_trials   - integer. Iterations over the same files
        s_print = 'run(): Starting to iterate sessions !'
        root.debug(s_print)
        if s_option != 'qlearning':
            # run for a specified number of trials
            sim.run(n_trials=n_trials, n_sessions=i_sess)
        elif s_option in ['qlearning']:
            # run for a specified number of trials
            s_valfunc = self._get_from_argv('s_valfunc')
            b_kplrn = self._get_from_argv('b_kplrn')
            b_testing = False
            if s_valfunc and not b_kplrn:
                b_testing = True
            sim.run(n_trials=n_trials, f_tolerance=0.05, b_testing=b_testing,
                    n_sessions=i_sess)
        if 'domino' in s_option:
            # record diagnostic statistics for domino
            if e.primary_agent:
                d_info = e.d_trial_data
                s_pnl = d_info['final_pnl'][-1]
                f_last_pnl = float('{:0.2f}'.format(s_pnl))
                f_avg_pnl = np.mean(d_info['final_pnl'])
                f_avg_pnl = float('{:0.2f}'.format(f_avg_pnl))
                f_avg_duration = np.mean(d_info['final_duration'])
                f_avg_duration = float('{:0.2f}'.format(f_avg_duration))
                l_rwd = d_info['final_reward']
                with open('dominostats.json', 'wb') as f:
                    f.write(json.dumps({'Agent': a.s_agent_name,
                                        'num_trials': len(l_rwd),
                                        'avg_duration': f_avg_duration,
                                        'avg_pnl': f_avg_pnl,
                                        'last_pnL': f_last_pnl,
                                        'last_reward': l_rwd[-1]}))
        return a

    def set_environment(self, n_trials, s_main_instr, i_month, n_sessions=1):
        '''
        Create the environment to be used in simulation

        :param d_data: dictionary. string. the container zip file to be used
        :param l_instrument: list. list of instrument to be simulated.
        :param NextStopTime: NextStopTime object. the hour all books is in sync
        :param s_main_instr: string. The main instrument traded
        :param i_month: integer. Month of the data used
        :param n_sessions*: integer. number of files to iterate
        '''
        i_m = i_month
        l_opt = self.l_opt
        # set up the stop time object
        l_h = [9, 10, 11, 12, 13, 14, 15]
        NextStopTime = matching_engine.NextStopTime(l_h, i_milis=self.i_milis)
        # set the env variables l_du, l_pu and l_price_adj
        o_aux = di_utilities.Settlements()
        if self.s_date and self.i_sess <= 1:
            s_root = self.s_root + self.s_date + '/'
            l_files = [s_root + self.s_date + '_{}_{}_new.zip']
            s_date = self.s_date
            s_date = '{}/{}/{}'.format(s_date[-2:], s_date[-4:-2], s_date[:4])
            func_du = di_utilities.calulateDI1expirationDays
            l_du = []
            l_pu = []
            l_price_adj = []
            for s_cmm in l_opt:
                f_du, dt1, dt2 = func_du(s_cmm, s_today=s_date)
                l_du.append(f_du)
                df = o_aux.getData(s_cmm, s_date, s_date, b_notInclude=False)
                l_pu.append(df['PU_Atual'].values[0])
                f_pu_anterior = df['PU_Anterior'].values[0]
                f_price = ((10.**5/f_pu_anterior)**(252./f_du)-1)*100
                l_price_adj.append(f_price)
            l_du = [l_du]
            l_pu = [l_pu]
            l_price_adj = [l_price_adj]
        else:
            l_data = self.d_available_data[i_m]
            if self.s_date:
                b_include = False
                l_aux = []
                for s_date in l_data:
                    if s_date == self.s_date:
                        b_include = True
                    if b_include:
                        l_aux.append(s_date)
                l_data = l_aux[:]
                s_err = 'Not enough data for the parameters passed'
                assert len(l_data) > 0, s_err

            l_files = [self.s_root + s_date + '/' + s_date + '_{}_{}_new.zip'
                       for s_date in l_data[:n_sessions]]
            l_du = []
            l_pu = []
            l_price_adj = []
            func_du = di_utilities.calulateDI1expirationDays
            for s_date in self.d_available_data[i_m][:n_sessions+1]:
                s_aux = '{}/{}/{}'.format(s_date[-2:], s_date[-4:-2],
                                          s_date[:4])
                l_du_aux = []
                l_pu_aux = []
                l_price_aux = []
                for s_cmm in l_opt:
                    f_du, dt1, dt2 = func_du(s_cmm, s_today=s_aux)
                    l_du_aux.append(f_du)
                    df = o_aux.getData(s_cmm, s_aux, s_aux, b_notInclude=False)
                    l_pu_aux.append(df['PU_Atual'].values[0])
                    f_pu_anterior = df['PU_Anterior'].values[0]
                    f_price = ((10.**5/f_pu_anterior)**(252./f_du)-1)*100
                    l_price_aux.append(f_price)
                l_du.append(l_du_aux)
                l_pu.append(l_pu_aux)
                l_price_adj.append(l_price_aux)
        # set up the environment
        obj_env = make('YieldCurve')
        e = obj_env(l_fname=l_files,
                    l_instrument=l_opt,
                    NextStopTime=NextStopTime,
                    s_main_intrument=s_main_instr,
                    l_du=l_du,
                    l_pu=l_pu,
                    l_price_adj=l_price_adj)
        return e

    def set_agent(self, s_option, e):
        '''
        '''
        # test if the option desired is defined
        s_vlf = self._get_from_argv('s_valfunc')
        i_vr = self._get_from_argv('i_version')
        if '_' + s_option not in self.d_experiments.keys():
            print '_' + s_option
            l_aux = self.d_experiments.keys()
            s_err = 'Select an <OPTION> between: \n{}'.format(l_aux)
            raise InvalidOptionException(s_err)
        func = self.d_experiments['_' + s_option]
        if s_vlf or i_vr:
            if i_vr:
                i_vr = int(i_vr)
            return func(e, self.i_milis/1000., s_valfunc=s_vlf, i_version=i_vr)
        return func(e, self.i_milis/1000.)

    def _get_from_argv(self, s_variable):
        '''
        '''
        if s_variable in self.argv:
            return self.argv[s_variable]
        return False

    def _qlearning(self, e, f_aux, s_valfunc=None, i_version=None):
        '''
        '''
        s_fdate = e.order_matching.s_file.split('_')[0].split('/')[-1]
        # e.set_reward_function('pnl_greedy')
        # e.set_reward_function('ofi_new')
        e.set_reward_function('ofi_pnl_new')
        e.set_log_file('qlearn_{}'.format(s_fdate))
        s_hedging_on = self._get_from_argv('s_hedging_on')
        b_hedging = self._get_from_argv('b_hedging')
        b_hedging = True
        # check if there are initial positions
        d_initial_pos = self._get_from_argv('d_initial_pos')
        b_keep_pos = self._get_from_argv('b_keep_pos')
        if not d_initial_pos:
            d_initial_pos = {}
        if not s_hedging_on:
            s_hedging_on = 'DI1F19'
        f_ttoupdate = 30.
        b_kplrn = False
        if s_valfunc:
            b_kplrn = not self._get_from_argv('b_kplrn')
            if b_kplrn:
                s1 = e.s_main_intrument[-3:]
                s2 = s_hedging_on[-3:]
                e.set_log_file('qlearn_ofs_{}{}_{}'.format(s1, s2, s_fdate))
                # f_ttoupdate = 0.5  # i dont know if i should zero that
                f_ttoupdate = 30.  # i dont know if i should zero that

        # check if shoulf use another verion of Qlearning
        obj_agent = QLearningAgent
        if i_version:
            e.set_log_file('qlearn{}_{}'.format(int(i_version), s_fdate))
            if i_version == 1:
                obj_agent = dissertation_tests.QLearningAgent1
            elif i_version == 2:
                obj_agent = dissertation_tests.QLearningAgent2
            elif i_version == 3:
                obj_agent = dissertation_tests.QLearningAgent3
            elif i_version == 4:
                obj_agent = dissertation_tests.QLearningAgent4
            elif i_version == 5:
                obj_agent = dissertation_tests.QLearningAgent5
            elif i_version == 6:
                obj_agent = dissertation_tests.QLearningAgent6
            elif i_version == 7:
                obj_agent = dissertation_tests.QLearningAgent7
            elif i_version == 8:
                obj_agent = dissertation_tests.QLearningAgent8
            elif i_version == 9:
                obj_agent = dissertation_tests.QLearningAgent9
            elif i_version == 10:
                obj_agent = dissertation_tests.QLearningAgent10
            elif i_version == 11:
                obj_agent = dissertation_tests.QLearningAgent11
            elif i_version == 12:
                obj_agent = dissertation_tests.QLearningAgent12
            elif i_version == 14:
                # alpha = 0.1
                obj_agent = dissertation_tests.QLearningAgent14
            elif i_version == 15:
                # alpha = 0.3
                obj_agent = dissertation_tests.QLearningAgent15
            elif i_version == 16:
                # alpha = 0.5
                obj_agent = dissertation_tests.QLearningAgent16
            elif i_version == 17:
                # alpha = 0.7
                obj_agent = dissertation_tests.QLearningAgent17
            elif i_version == 18:
                # alpha = 0.9
                obj_agent = dissertation_tests.QLearningAgent18
            elif i_version == 19:
                # gamma = 0.1
                obj_agent = dissertation_tests.QLearningAgent19
            elif i_version == 20:
                # gamma = 0.3
                obj_agent = dissertation_tests.QLearningAgent20
            elif i_version == 21:
                # gamma = 0.5
                obj_agent = dissertation_tests.QLearningAgent21
            elif i_version == 22:
                # gamma = 0.7
                obj_agent = dissertation_tests.QLearningAgent22
            elif i_version == 23:
                # gamma = 0.9
                obj_agent = dissertation_tests.QLearningAgent23
            elif i_version == 24:
                # decay func = linear
                obj_agent = dissertation_tests.QLearningAgent24
            elif i_version == 25:
                # decay func = tpower
                obj_agent = dissertation_tests.QLearningAgent25
            elif i_version == 26:
                # decay func = trig
                obj_agent = dissertation_tests.QLearningAgent26
            elif i_version == 27:
                obj_agent = dissertation_tests.QLearningAgent27
            elif i_version == 28:
                e.set_reward_function('ofi_new')
                obj_agent = dissertation_tests.QLearningAgent28
            elif i_version == 29:
                e.set_reward_function('pnl_greedy')
                obj_agent = dissertation_tests.QLearningAgent29
            elif i_version == 30:
                e.set_reward_function('ofi_pnl_new')
                obj_agent = dissertation_tests.QLearningAgent30
            elif i_version == 31:
                # to prove that size variables is not important
                e.set_reward_function('ofi_pnl_new')
                obj_agent = dissertation_tests.QLearningAgent31
            elif i_version == 32:
                # to prove that OFI from the past 10 sec is important
                e.set_reward_function('ofi_pnl_new')
                obj_agent = dissertation_tests.QLearningAgent32
            elif i_version == 33:
                # to prove that is worst to not use any short variable
                e.set_reward_function('ofi_pnl_new')
                obj_agent = dissertation_tests.QLearningAgent33
            elif i_version == 34:
                # to prove that the short bid-ask is not relevant
                e.set_reward_function('ofi_pnl_new')
                obj_agent = dissertation_tests.QLearningAgent34
            elif i_version == 35:
                # to prove that should use long bid-ask
                e.set_reward_function('ofi_pnl_new')
                obj_agent = dissertation_tests.QLearningAgent35
            elif i_version == 36:
                # to prove that should use long book ratio
                e.set_reward_function('ofi_pnl_new')
                obj_agent = dissertation_tests.QLearningAgent36
            elif i_version == 37:
                # to show that high-low does not help
                e.set_reward_function('ofi_pnl_new')
                obj_agent = dissertation_tests.QLearningAgent37
            elif i_version == 38:
                # to show that rl_price also does not help
                e.set_reward_function('ofi_pnl_new')
                obj_agent = dissertation_tests.QLearningAgent38

        # create the agent
        a = e.create_agent(obj_agent,
                           f_min_time=f_aux,
                           s_hedging_on=s_hedging_on,
                           d_normalizers=NORMALIZERS,
                           d_ofi_scale=OFI_DICT,
                           b_hedging=b_hedging,
                           d_initial_pos=d_initial_pos,
                           s_decay_fun='linear',
                           # s_decay_fun='tpower',
                           f_gamma=0.5,
                           f_alpha=0.5,
                           b_keep_pos=b_keep_pos,
                           f_ttoupdate=f_ttoupdate)
        if s_valfunc:
            a.set_qtable(s_valfunc, b_freezy_policy=b_kplrn)
            if b_kplrn:
                f_gain = self._get_from_argv('f_gain')
                f_loss = self._get_from_argv('f_loss')
                a.risk_model.set_gain_loss(f_gain, f_loss)
                # set to print always
                if PRINT_ON_TEST:
                    a.set_to_print_always()

        return a

    def _random(self, e, f_aux, s_valfunc=None, i_version=None):
        '''
        '''
        s_fdate = e.order_matching.s_file.split('_')[0].split('/')[-1]
        e.set_reward_function('ofi_new')
        s_hedging_on = self._get_from_argv('s_hedging_on')

        s1 = e.s_main_intrument[-3:]
        s2 = s_hedging_on[-3:]
        e.set_log_file('random_{}{}_{}'.format(s1, s2, s_fdate))

        b_hedging = self._get_from_argv('b_hedging')
        b_hedging = True
        # check if there are initial positions
        d_initial_pos = self._get_from_argv('d_initial_pos')
        b_keep_pos = self._get_from_argv('b_keep_pos')
        if not d_initial_pos:
            d_initial_pos = {}
        if not s_hedging_on:
            s_hedging_on = 'DI1F19'
        a = e.create_agent(RandomAgent,
                           f_min_time=f_aux,
                           s_hedging_on=s_hedging_on,
                           d_normalizers=NORMALIZERS,
                           d_ofi_scale=OFI_DICT,
                           b_hedging=b_hedging,
                           d_initial_pos=d_initial_pos,
                           f_ttoupdate=30.,
                           f_gamma=0.5,
                           f_alpha=0.5,
                           b_keep_pos=b_keep_pos)

        # set to print always
        f_gain = self._get_from_argv('f_gain')
        f_loss = self._get_from_argv('f_loss')
        a.risk_model.set_gain_loss(f_gain, f_loss)
        if PRINT_ON_TEST:
            a.set_to_print_always()

        return a

    def _domino_q(self, e, f_aux, s_valfunc=None, i_version=None):
        '''
        '''
        return self._qlearning(e, f_aux, s_valfunc, i_version)

    def _domino_rand(self, e, f_aux, s_valfunc=None, i_version=None):
        '''
        '''
        return self._random(e, f_aux)


if __name__ == '__main__':
    # treat bad specification command
    # try:
    # initiate Run object
    obj_to_run = Run()
    # initiate parser and process arguments
    l_valid_opt = [s_key[1:] for s_key in obj_to_run.d_experiments.keys()]
    s_txt = '''\
            Run simulation
            --------------------------------
            Run one of the agents implemented in agents
            to trade Brazilian interest rate future
            contracts
            '''
    s_txt2 = 'the type of the agent to run the simulation. '
    s_txt2 += '<OPTION> values are '+', '.join(l_valid_opt)
    obj_formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=obj_formatter,
                                     description=textwrap.dedent(s_txt))
    parser.add_argument('agent', default='debug', choices=l_valid_opt,
                        help=s_txt2, metavar='<OPTION>')
    parser.add_argument('-t', '--trials', default=None, type=int, metavar='',
                        help='number of trials to perform in the same file')
    parser.add_argument('-d', '--date', default=None, type=str, metavar='',
                        help='date of the file to be used (formart AAAAMMDD)')
    s_help = 'number of different sessions to iterate on each trial'
    parser.add_argument('-s', '--sessions', default=None, type=int, metavar='',
                        help=s_help)
    parser.add_argument('-v', '--version', default=None, type=int, metavar='',
                        help='version of the learner to be tested')
    s_help = 'If should use the value function from the previous day'
    parser.add_argument('-vf', '--valfunc', action='store_true',
                        help=s_help)
    s_help = 'If should keep learning (when using a previous value function)'
    parser.add_argument('-kl', '--keeplrn', action='store_true',
                        help=s_help)
    parser.add_argument('-i', '--instr', default=None, type=str, metavar='',
                        help='Intrument to trade. Default is DI1F21')
    s_help = 'month of the data to use when iterating on multiple trials'
    parser.add_argument('-m', '--month', default=None, type=int, metavar='',
                        help=s_help)
    s_help = 'pop up a window to render the limit order books being simulated'
    parser.add_argument('-viz', '--visualization', action='store_true',
                        help=s_help)
    # recover arguments
    args = parser.parse_args()
    b_use_valfunc = args.valfunc
    b_kplrn = args.keeplrn
    i_trials = args.trials
    s_date = args.date
    i_sess = args.sessions
    i_version = args.version
    s_main_instr = args.instr
    i_month = args.month
    b_render = args.visualization

    # check the instruments to use
    if s_main_instr:
        if s_main_instr == 'DI1F21':
            l_opt = ['DI1F19', 'DI1F21']
            s_hedge = 'DI1F19'
        elif s_main_instr == 'DI1F19':
            l_opt = ['DI1F19', 'DI1F21']
            s_hedge = 'DI1F21'
        elif s_main_instr == 'DI1F23':
            l_opt = ['DI1F21', 'DI1F23']
            s_hedge = 'DI1F21'
    else:
        l_opt = ['DI1F19', 'DI1F21']
        s_hedge = 'DI1F19'

    # check othe variables
    s_valfunc = None
    f_gain = 100000.
    f_loss = -100000.
    if b_use_valfunc or args.agent in ['random', 'domino_rand']:
        if not s_main_instr:
            s_ifunc = 'F21'
        else:
            s_ifunc = s_main_instr[-3:]
        f_gain = VALUE_FUNCS[s_ifunc]['PARAM']['GAIN']
        f_loss = VALUE_FUNCS[s_ifunc]['PARAM']['LOSS']
    if b_use_valfunc:
        s_valfunc = 'data/qtables/{}.dat'.format(VALUE_FUNCS[s_ifunc][s_date])
    # run simulation
    obj_to_run.run(args.agent, i_trials=i_trials, s_date=s_date, i_sess=i_sess,
                   i_version=i_version, b_kplrn=b_kplrn, s_valfunc=s_valfunc,
                   b_keep_pos=False, s_main_instr=s_main_instr, f_gain=f_gain,
                   f_loss=f_loss, l_instruments=l_opt, s_hedging_on=s_hedge,
                   i_month=i_month, b_render=b_render)
