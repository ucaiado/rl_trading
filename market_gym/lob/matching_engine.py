#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement different matching engines to handle the actions taken by each agent
in the environment

@author: ucaiado

Created on 10/24/2016
"""
import random
import logging
import zipfile
import csv
import numpy as np
import itertools
import book as book
import pprint
from translator import translate_trades, translate_row
from market_gym.config import DEBUG, root, START_MKT_TIME, CLOSE_MKT_TIME


'''
Begin help functions
'''


class Foo(Exception):
    """
    Foo is raised by any class to help in debuging
    """
    pass


def get_next_stoptime(l_hours, i_milis=2, i_sec=None, i_noise=None):
    '''
    Return a generator to handle the tme to stop the updates in the books

    :param l_hours: list. Hours to be used in stoptime calculation
    :param i_milis*: integer. Number of miliseconds between each stoptime
    :param i_sec*: integer. Number of seconds between each stoptime. If
        defined, the i_milis is not used
    :param i_noise*: integer: add noise to the stoptime in milis
    '''
    s_time = "{:0>2}:{:0>2}:{:0>2}.{:0>3}"
    f_time = l_hours[0] * 60**2 + np.random.rand() * 2 * 60.
    f_delta_time = i_milis/1000.
    if i_sec:
        f_delta_time = i_sec
    f_start_time = START_MKT_TIME
    f_close_time = CLOSE_MKT_TIME + 60*5  # should add to work MARKET_CLOSED

    while (f_time <= (f_close_time)):
        f_time += f_delta_time
        # add additional miliseconds to the next stop time when b_rand is set
        if i_noise:
            s_err = 'noise should be less than delta time'
            assert i_noise/1000. < f_delta_time, s_err
            if np.random.rand() > 0.4:
                i_mult = 1
                if np.random.rand() < 0.5:
                    i_mult = -1
                f_time += (int(np.ceil(np.random.rand()*i_noise))*i_mult)/1000.
        if f_time < (f_start_time):
            pass
        else:
            i_hour = int(f_time/3600)
            i_minute = int((f_time - i_hour*60**2)/60)
            i_second = int((f_time - i_hour*60**2 - i_minute*60))
            i_mili = f_time - i_hour*60**2 - i_minute*60 - i_second
            i_mili = int(i_mili*1000)
            yield s_time.format(i_hour,
                                i_minute,
                                i_second,
                                i_mili)


class NextStopTime():
    '''
    StopTime controller to define the next time to sync all the LOBs
    '''
    def __init__(self, l_hours, i_milis=2, i_sec=None):
        '''
        Initiate a NextStopTime object. Save all parameters as attributes

        :param l_hours: list. Hours to be used in stoptime calculation
        :param i_milis*: integer. Number of miliseconds between each stoptime
        :param i_sec*: integer. Number of seconds between each stoptime. If
            defined, the i_milis is not used
        '''
        i_noise = None
        if i_milis > 4:
            i_noise = min(1, i_milis/5)
        self.gen_stoptime = get_next_stoptime(l_hours, i_milis, i_sec, i_noise)
        self.gen_stoptime, self.gen_backup = itertools.tee(self.gen_stoptime)
        self.s_last_stoptime = ''
        self.s_stoptime_was_set = ''
        self.s_time = "{:0>2}:{:0>2}:{:0>2}.{:0>3}"
        self.b_use_last = False

    def set_stoptime(self, i_hour, i_minute, i_second, i_mili):
        '''
        Set a new stoptime if is not already set

        :param i_hour: integer. Hour of the day
        :param i_minute: integer. Minutes of the day
        :param i_second: integer. seconds of the day
        :param i_mili: integer. milis of the day
        '''
        s_rtn = self.s_stoptime_was_set
        if self.s_stoptime_was_set == '':
            s_rtn = self.s_time.format(i_hour, i_minute, i_second, i_mili)
            self.s_stoptime_was_set = s_rtn
            self.b_use_last = True
        return s_rtn

    def reset(self):
        '''
        Reset the stop-time generator
        '''
        self.gen_stoptime, self.gen_backup = itertools.tee(self.gen_backup)

    def next(self):
        '''
        Return a string representation of the next stoptime
        '''
        if self.s_stoptime_was_set != '':
            # if there is a stoptime externally set, use it
            s_rtn = self.s_stoptime_was_set
            self.s_stoptime_was_set = ''
            return s_rtn
        elif self.b_use_last:
            # if did not use the last_stoptime, set to use it again
            self.b_use_last = False
            return self.s_last_stoptime
        # get the nest stoptime from iterator
        self.s_last_stoptime = self.gen_stoptime.next()
        return self.s_last_stoptime

    def has_already_used_param(self):
        '''
        Check if have already used the stoptime that was set
        '''
        return self.s_stoptime_was_set == '' and not self.b_use_last


'''
End help functions
'''


class OrderMatching(object):
    '''
    An order matching representation that access the agents from an environment
    and handle the interation  of the individual behaviours, translating  it as
    instructions to the Order Book
    '''

    def __init__(self, env):
        '''
        Initialize a OrderMatching object. Save all parameters as attributes

        :param env: Environment object. The Market
        :param s_instrument: string. name of the instrument of book
        '''
        # save parameters as attributes
        self.env = env
        # attributes to control the qty trades by each side
        self.i_agr_ask = 0
        self.i_agr_bid = 0
        # order flow count
        self.i_ofi = 0

    def __iter__(self):
        '''
        Return the self as an iterator object. Use next() to iterate
        '''
        return self

    def next(self):
        '''
        '''
        raise NotImplementedError

    def __call__(self):
        '''
        Return the next list of messages of the simulation
        '''
        return self.next()


class BvmfFileMatching(OrderMatching):
    '''
    Order matching engine that use Level II data from Bvmf to reproduce the
    order book
    '''

    def __init__(self, env, l_instrument, i_num_agents, l_file):
        '''
        Initialize an OrderMatching object. Save all parameters as attributes

        :param env: Environment object. The Market
        :param l_instrument: list. name of the instruments of the simulation
        :param i_num_agents: integer. Number of agents
        :param l_file: string. Format of the name of the zip files used
        :param i_idx: integer. The index of the start file to be read
        '''
        super(BvmfFileMatching, self).__init__(env)
        self.l_instrument = l_instrument
        self.i_num_agents = i_num_agents
        if not isinstance(l_file, list):
            l_file = [l_file]
        self.l_file = l_file
        self._s_file = None
        self.l_order_books = []
        # NOTE: it is increased by 1 in reset, that is the first method called
        # in a simulation
        self.idx = -1
        self.i_nrow = 0.
        self.s_time = ''
        self.f_time = 0.
        self.last_date = 0
        self.best_bid = (0, 0)
        self.best_ask = (0, 0)
        self.obj_best_bid = None
        self.obj_best_ask = None
        self.i_ofi = 0
        self.i_ofi_10s = 0
        self.i_qty_traded_at_bid_10s = 0
        self.i_qty_traded_at_ask_10s = 0
        self.i_qty_traded_at_bid = 0
        self.i_qty_traded_at_ask = 0
        self.mid_price_10s = 0.
        self.b_get_new_row = True
        self.f_last_bucket = 0.
        self.f_seconds_to_group = 21.
        self.s_stoptime = ''
        self.f_stoptime = 0.
        self.d_map_book_list = dict(zip(l_instrument,
                                        (np.cumsum([1]*len(l_instrument))-1)))

    def get_trial_identification(self):
        '''
        Return the name of the files used in the actual trial
        '''
        if int(self.idx) > len(self.l_fnames)-1:
            return None
        return self.l_fnames[int(self.idx)].filename

    def get_order_book_obj(self, s_instrument):
        '''
        Return the object of the desired order book

        :param s_instrument: string. The instrument to get the LOB
        '''
        i_idx = self.d_map_book_list[s_instrument]
        return self.l_order_books[i_idx]

    def reshape_row(self, idx, row, s_side=None):
        '''
        Translate a line from a file of the bloomberg level I data

        :param idx: integer.
        :param row: dict.
        :param s_side*: string. 'BID' or 'ASK'. Determine the side of the trade
        '''
        return translate_row(idx, row, self, s_side)

    def reset(self):
        '''
        Reset the order matching and all variables needed
        '''
        # make sure that dont reset twice
        self.l_order_books = []
        self.s_time = ''
        self.f_time = 0.
        self.last_date = 0
        self.best_bid = (0, 0)
        self.best_ask = (0, 0)
        self.obj_best_bid = None
        self.obj_best_ask = None
        self.i_ofi = 0
        self.i_ofi_10s = 0
        self.i_qty_traded_at_bid_10s = 0
        self.i_qty_traded_at_ask_10s = 0
        self.i_qty_traded_at_bid = 0
        self.i_qty_traded_at_ask = 0
        self.mid_price_10s = 0.
        self.b_get_new_row = True
        self.f_last_bucket = 0.
        self.f_seconds_to_group = 21.
        self.s_stoptime = ''
        self.f_stoptime = 0.
        self.i_nrow = 0
        self._s_file = None
        self.idx += 1

    def update(self, l_msg):
        '''
        Update the Book and all information related to it

        :param l_msg: list. messages to use to update the book
        :param b_print*: boolean. If should print the messaged generated
        '''
        if l_msg:
            # process each message generated by translator
            for msg in l_msg:
                i_idx = self.d_map_book_list[msg['instrumento_symbol']]
                self.l_order_books[i_idx].update(msg)
            # process the last message and use info from row
            # to compute the number of shares traded by aggressor
            if msg['order_status'] in ['Partially Filled', 'Filled']:
                if msg['agressor_indicator'] == 'Agressor':
                    pass
        # terminate
        self.i_nrow += 1

    @property
    def s_file(self):
        '''
        Access the current s_file in simulation. It is extracted from l_file
        '''
        if self.idx > len(self.l_file)-1:
            self.idx = 0
        self._s_file = self.l_file[self.idx]
        return self._s_file

    def next(self, b_print=False):
        '''
        Return a list of messages from the agents related to the current step

        :param b_print*: boolean. If should print the messaged generated
        '''
        # if it is the first line of the file, instanciate new books
        if self.i_nrow == 0:
            for s_name in self.l_instrument:
                s_fbid = self.s_file.format('BID', s_name)
                s_fask = self.s_file.format('ASK', s_name)
                self.l_order_books.append(book.LimitOrderBook(s_fbid,
                                                              s_fask,
                                                              s_name))
        self.i_nrow += 1
        # try to read a row of an already opened file
        try:
            self.s_stoptime = self.env.NextStopTime.next()
            l_msg = []
            b_exit_for = False
            for s_name, book_obj in zip(self.l_instrument, self.l_order_books):
                # set the stop time and update the books until that
                book_obj.set_stop_time(self.s_stoptime)
                for l_msg_aux in book_obj:
                    l_msg += l_msg_aux
                    if len(l_msg_aux) > 0:
                        b_exit_for = True
                        break
                # keep the maximum time
                if book_obj.f_time >= self.f_time:
                    self.f_time = book_obj.f_time
                    self.s_time = book_obj.s_time
                if b_exit_for:
                    b_exit_for = False
                    break
            self.last_date = self.f_time
            return l_msg
        except StopIteration:
            self.i_nrow = 0
            self.i_qty_traded_at_bid_10s = 0
            self.i_qty_traded_at_ask_10s = 0
            self.i_qty_traded_at_bid = 0
            self.i_qty_traded_at_ask = 0
            self.i_ofi_10s = 0
            self.i_ofi = 0
            self.last_date = 0
            self.best_bid = (0, 0)
            self.best_ask = (0, 0)
            self.obj_best_bid = None
            self.obj_best_ask = None
            self.mid_price_10s = 0.
            raise StopIteration
