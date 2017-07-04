#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement a simulator to mimic a dynamic order book environment

@author: ucaiado

Created on 10/24/2016
"""
import datetime
import logging
import os
import sys
import pandas as pd
import numpy as np
import pickle
import pprint
import gzip
import random
import time
from market_gym.config import DEBUG, root, s_log_file
sys.path.append('../../')

'''
Begin help functions
'''


def save_q_table(e, i_trial):
    '''
    Log the final Q-table of the algorithm

    :param e: Environment object. The order book
    :param i_trial: integer. id of the current trial
    '''
    agent = e.primary_agent
    try:
        q_table = agent.value_function
        # define the name of the files
        s_log_name = s_log_file.split('/')[-1].split('.')[0]
        s_fname = 'log/qtables/{}_valuefun_{}.dat'
        s_fname = s_fname.format(s_log_name, i_trial)
        # save data structures
        pickle.dump(q_table, open(s_fname, 'w'))
        root.info('save_q_table(): Q-table saved successfully')
    except:
        root.info('save_q_table(): No Q-table to be printed')


def save_log_info(e, i_trial):
    '''
    Log the final log_info attribute from the agent

    :param e: Environment object. The order book
    :param i_trial: integer. id of the current trial
    '''
    agent = e.primary_agent
    if agent.b_print_always:
        return
    log_info = agent.log_info
    # define the name of the files
    s_log_name = s_log_file.split('/')[-1].split('.')[0]
    s_fname = 'log/loginfo/{}_loginfo_{}.pklz'
    s_fname = s_fname.format(s_log_name, i_trial)
    # f = gzip.open(s_log_name,'rb')  to read
    # save data structures
    f = gzip.open(s_fname, 'wb')
    pickle.dump(log_info, f)
    f.close()

'''
End help functions
'''


class Simulator(object):
    """
    Simulates agents in a dynamic order book environment.
    """
    def __init__(self, env, update_delay=1.0, display=True):
        '''
        Initiate a Simulator object. Save all parameters as attributes
        Environment Object. The Environment where the agent acts

        :param env: Environment Object. The environment to be simulated
        :param update_delay*: Float. Seconds elapsed to print out the book
        :param display*: Boolean. If should open a visualizer
        '''
        self.env = env

        self.quit = False
        self.start_time = None
        self.current_time = 0.0
        self.last_updated = 0.0
        self.update_delay = update_delay

        if display:
            self.func_render = self._render_env
        else:
            self.func_render = self._render_not

    def run(self, n_trials=1, n_sessions=1, f_tolerance=0.05, b_testing=False):
        '''
        Run the simulation to train the algorithm

        :param n_sessions*: integer. Number of different files to read
        :param n_trials*: integer. Iterations over the same files
        :param f_tolerance*: float. Minimum epsilon necessary to begin testing
        :param b_testing*: boolean. should use the value function already fit
        :param b_render*: boolean. If should render the environment
        '''
        if self.env.primary_agent:
            if not b_testing and self.env.primary_agent.learning:
                root.info('Simulator.run(): Starting training session !')
        for trial in xrange(n_trials):
            if self.env.primary_agent:
                if self.env.primary_agent.learning:
                    # assumes epsilon decays to 0 (freeze policy updates)
                    if self.env.primary_agent.f_epsilon <= f_tolerance:
                        if not b_testing and self.env.primary_agent.learning:
                            s_err = 'Simulator.run(): Starting test session !'
                            root.info(s_err)
                        b_testing = True
                    elif b_testing:
                        s_err = 'Simulator.run(): Starting test session !'
                        root.info(s_err)
            for i_sess in xrange(n_sessions):
                self.quit = False
                self.env.reset(testing=b_testing, carry_pos=i_sess > 0)
                self.env.step()  # give the first step
                # set variable to control rendering
                self.current_time = 0.0
                self.last_updated = 0.0
                self.start_time = 0.
                self.start_time = time.time()
                # iterate over the current dataset
                while True:
                    try:
                        # Update current time
                        # self.current_time = time.time() - self.start_time
                        self.current_time = self.env.order_matching.f_time
                        # Update environment
                        f_time_step = self.current_time - self.last_updated
                        self.env.step()
                        # print information to be used by a visualization
                        if f_time_step >= self.update_delay:
                            # TODO: Print out the scenario to be visualized
                            self.func_render()
                            self.last_updated = self.current_time
                    except StopIteration:
                        self.quit = True
                    except KeyboardInterrupt:
                        self.print_when_paused()
                        s_msg = 'Should quit this trial (y/n)?'
                        programPause = raw_input(s_msg)
                        if programPause == 'y':
                            self.quit = True
                        else:
                            continue
                    except:
                        print 'Unexpected error:', sys.exc_info()[0]
                        raise
                    finally:
                        if self.quit or self.env.done:
                            break
                # # save the current Q-table
                # save_q_table(self.env, trial+1)
                # log the end of the trial
                self.env.log_trial()
        # save the last Q-table
        if self.env.primary_agent:
            if not self.env.primary_agent.b_print_always:
                save_q_table(self.env, trial+1)
        # save log info
        # save_log_info(self.env, trial+1)

    def print_when_paused(self):
        '''
        Print information when a simulation is interupted by ctrl+c
        '''
        if self.env.primary_agent:
            agent = self.env.primary_agent
            s_time = self.env.order_matching.s_time
            f_duration = agent.log_info['duration']
            s_msg = '\n=============================\n'
            s_msg += '{}: PnL {:0.2f}, duration {:0.2f}'
            print s_msg.format(s_time, agent.f_pnl, f_duration)
            s_main = self.env.s_main_intrument
            s_hedge = self.env.primary_agent.risk_model.main_hedge
            for s_key in ['BID', 'ASK']:
                for s_instr in self.env.l_instrument:
                    l_aux = agent.d_trades[s_instr][s_key]
                    if len(l_aux) > 0:
                        s_err = '\nLast opened positions in {} {}:'
                        print s_err.format(s_key, s_instr)
                        for idx, (s_side, f_price, d_) in enumerate(l_aux):
                            print '    {} {}'.format(s_side, f_price)
                            if idx > 4:
                                break
            print '\nOpenPrices:'
            pprint.pprint(agent.current_open_price)
            print '\nPositions:'
            # calculate the current position
            f_pos = agent.position[s_main]['qBid']
            f_pos -= agent.position[s_main]['qAsk']
            f_pos_discl = f_pos + agent.disclosed_position[s_main]['qBid']
            f_pos_discl -= agent.disclosed_position[s_main]['qAsk']
            print '    Pos: {} Disclosed: {}'.format(f_pos, f_pos_discl)
            print '\nBest Prices:'
            for s_instr in self.env.l_instrument:
                agent_orders = agent.d_order_tree[s_instr]
                if agent_orders['BID'].count > 0:
                    f_bid = agent_orders['BID'].max_key()
                    print '    Bid: {} {}'.format(s_instr,
                                                  f_bid)
                if agent_orders['ASK'].count > 0:
                    f_ask = agent_orders['ASK'].min_key()
                    print '    Ask: {} {}'.format(s_instr,
                                                  f_ask)
            for s_instr in agent.env.l_instrument:
                print '\n{} book:'.format(s_instr)
                print self.env.get_order_book(s_instr, b_rtn_dataframe=True)
            print '\n'

    def _render_env(self):
        '''
        Call the render method from environment
        '''
        self.env.render()

    def _render_not(self):
        '''
        Do nothing
        '''
        pass
