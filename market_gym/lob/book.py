#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Library to recreate a limit order book from data given by BmfBovespa

@author: ucaiado

Created on 07/05/2016
"""
# import libraries
from bintrees import FastRBTree
import numpy as np
import pandas as pd
import parser_data
import zipfile
import translator as translator
import pprint
from market_gym.utils.microstruct_stats import BookStats

'''
Begin help functions
'''


class DifferentPriceException(Exception):
    """
    DifferentPriceException is raised by the update() method in the PriceLevel
    class to indicate that the price of order object is different from the
    PriceLevel
    """
    pass


class InvalidTypeException(Exception):
    """
    InvalidTypeException is raised by the init() method in the Bookside
    class to indicate that the the book type chose is invalid
    """
    pass


def open_file(s_fname):
    '''
    Open a Zipped file and return the archice (zip object) and the file opened

    :param s_fname: string. path to the zipped file
    '''
    archive = zipfile.ZipFile(s_fname, 'r')
    fr = archive.open(archive.infolist()[0])
    return fr, archive


def get_relative_price(best_queue, order_obj):
    '''
    Measure the relative price of the order pased, in 100th
    '''
    f_relprice = 0.
    if best_queue[0]:
        if order_obj['order_side'] == 'Sell Order':
            # \Delta_a(p) = p - p^a_t
            f_relprice = order_obj['order_price'] - best_queue[0]
        else:
            # \Delta_b(p) = p^b_t - p
            f_relprice = best_queue[0] - order_obj['order_price']
    return int((f_relprice)*100)

'''
End help functions
'''


class Order(object):
    '''
    A representation of a single Order
    '''
    def __init__(self, d_msg):
        '''
        Instantiate a Order object. Save all parameter as attributes

        :param d_msg: dictionary.
        '''
        # keep data extract from file
        self.d_msg = d_msg.copy()
        f_q1 = self.d_msg['org_total_qty_order']
        f_q2 = self.d_msg['traded_qty_order']
        self.d_msg['total_qty_order'] = f_q1 - f_q2
        self.order_id = int(d_msg['seq_order_number'])
        self.sec_order_id = int(d_msg['priority_indicator'])
        self.name = d_msg['seq_order_number']
        self.main_id = self.sec_order_id
        if self.sec_order_id == 0:
            self.main_id = self.order_id

    def set_values(self, s_key, f_value):
        '''
        Include new information in the order dictionary

        :param s_key: string. new key to include
        :param f_value: float. value to include
        '''
        self.d_msg[s_key] = f_value

    def __str__(self):
        '''
        Return the name of the Order
        '''
        return self.name

    def __repr__(self):
        '''
        Return the name of the Order
        '''
        return self.name

    def __eq__(self, other):
        '''
        Return if a Order has equal order_id from the other

        :param other: Order object. Order to be compared
        '''
        if isinstance(other, str):
            i_aux = int(other)
            return self.order_id == i_aux
        return self.order_id == other.order_id

    def __ne__(self, other):
        '''
        Return if a Order has different order_id from the other

        :param other: Order object. Order to be compared
        '''
        return not self.__eq__(other)

    def __hash__(self):
        '''
        Allow the Order object be used as a key in a hash table. It is used by
        dictionaries
        '''
        return self.order_id.__hash__()

    def __getitem__(self, s_key):
        '''
        Allow direct access to the inner dictionary of the object

        :param i_index: integer. index of the l_legs attribute list
        '''
        return self.d_msg[s_key]


class PriceLevel(object):
    '''
    A representation of a Price level in the book
    '''
    def __init__(self, f_price):
        '''
        A representation of a PriceLevel object
        '''
        self.f_price = f_price
        self.i_qty = 0
        self.f_time = 0
        self.order_tree = FastRBTree()

    def add(self, order_aux):
        '''
        Insert the information in the tree using the info in order_aux. Return
        is should delete the Price level or not

        :param order_aux: Order Object. The Order message to be updated
        '''
        # check if the order_aux price is the same of the self
        s_status = order_aux['order_status']
        self.f_time = order_aux['priority_seconds']
        if order_aux['order_price'] != self.f_price:
            raise DifferentPriceException
        elif s_status in ['New', 'Replaced', 'Partially Filled']:
            self.order_tree.insert(order_aux.main_id, order_aux)
            self.i_qty += int(order_aux['total_qty_order'])
        # check if there is no object in the updated tree (should be deleted)
        return self.order_tree.count == 0

    def delete(self, i_old_sec_order, i_old_qty):
        '''
        Delete the information in the tree using the info in order_aux. Return
        is should delete the Price level or not

        :param i_old_sec_order: Integer. The previous secondary order id
        :param i_old_qty: Integer. The previous order qty
        '''
        # check if the order_aux price is the same of the self
        try:
            self.order_tree.remove(i_old_sec_order)
            self.i_qty -= i_old_qty
        except KeyError:
            raise DifferentPriceException
        # check if there is no object in the updated tree (should be deleted)
        return self.order_tree.count == 0

    def __str__(self):
        '''
        Return the name of the PriceLevel
        '''
        return '{:,.0f}'.format(self.i_qty)

    def __repr__(self):
        '''
        Return the name of the PriceLevel
        '''
        return '{:,.0f}'.format(self.i_qty)

    def __eq__(self, other):
        '''
        Return if a PriceLevel has equal price from the other

        :param other: PriceLevel object. PriceLevel to be compared
        '''
        # just to make sure that there is no floating point discrepance
        f_aux = other
        if not isinstance(other, float):
            f_aux = other.f_price
        return abs(self.f_price - f_aux) < 1e-4

    def __gt__(self, other):
        '''
        Return if a PriceLevel has a gerater price from the other.
        Bintrees uses that to compare nodes

        :param other: PriceLevel object. PriceLevel to be compared
        '''
        # just to make sure that there is no floating point discrepance
        f_aux = other
        if not isinstance(other, float):
            f_aux = other.f_price
        return (f_aux - self.f_price) > 1e-4

    def __lt__(self, other):
        '''
        Return if a Order has smaller order_id from the other. Bintrees uses
        that to compare nodes

        :param other: Order object. Order to be compared
        '''
        f_aux = other
        if not isinstance(other, float):
            f_aux = other.f_price
        return (f_aux - self.f_price) < -1e-4

    def __ne__(self, other):
        '''
        Return if a Order has different order_id from the other

        :param other: Order object. Order to be compared
        '''
        return not self.__eq__(other)


class BookSide(object):
    '''
    A side of the lmit order book representation
    '''
    def __init__(self, s_side, fr_data, i_member=None):
        '''
        Initialize a BookSide object. Save all parameters as attributes

        :param s_side: string. BID or ASK
        :param fr_data: ZipExtFile object. data to read
        :param i_member*: integer. Member number to be used as a filter
        '''
        if s_side not in ['BID', 'ASK']:
            raise InvalidTypeException('side should be BID or ASK')
        self.i_member = i_member
        self.s_side = s_side
        self.price_tree = FastRBTree()
        self._i_idx = 0
        self.fr_data = fr_data
        self.parser = parser_data.LineParser(s_side)
        self.d_order_map = {}
        self.last_price = 0.
        # control other statistics
        self.best_queue = (None, None)
        self.i_qty_rel = 0
        self.i_cum_rel = 0

    def set_last_best_queue(self, t_best):
        '''
        Set the best queue of this side
        '''
        self.best_queue = t_best

    def update(self, d_data):
        '''
        Update the state of the order book given the data pased

        :param d_data: dict. data from the last row
        '''
        # update the book information
        order_aux = Order(d_data)
        s_status = order_aux['order_status']
        b_sould_update = True
        i_rel_price = 0
        # treat Bovespa files at the begining f the day
        if s_status != 'New':
            try:
                i_old_id = self.d_order_map[order_aux]['main_id']
            except KeyError:
                # is not securing changes, also change part. filled status
                l_check = ['Canceled', 'Filled']
                if not self.b_secure_changes:
                    l_check = ['Canceled', 'Filled', 'Partially Filled']
                # change order status when it is not found
                if s_status in l_check:
                    b_sould_update = False
                    s_status = 'Invalid'
                elif s_status == 'Replaced':
                    s_status = 'New'
        # process
        if s_status == 'New':
            b_sould_update = self._new_order(order_aux)
            i_rel_price = get_relative_price(self.best_queue, order_aux)
        elif s_status != 'Invalid':
            i_old_id = self.d_order_map[order_aux]['main_id']
            f_old_pr = self.d_order_map[order_aux]['price']
            i_old_q = self.d_order_map[order_aux]['qty']
            i_rel_price = self.d_order_map[order_aux]['relative_price']
            # hold the last traded price
            if s_status in ['Partially Filled', 'Filled']:
                self.last_price = order_aux['order_price']
            # process message
            if s_status in ['Canceled', 'Expired', 'Filled']:
                b_sould_update = self._canc_expr_filled_order(order_aux,
                                                              i_old_id,
                                                              f_old_pr,
                                                              i_old_q)
            elif s_status == 'Replaced':
                i_rel_price = get_relative_price(self.best_queue, order_aux)
                b_sould_update = self._replaced_order(order_aux,
                                                      i_old_id,
                                                      f_old_pr,
                                                      i_old_q)
            elif s_status == 'Partially Filled':
                b_sould_update = self._partially_filled(order_aux,
                                                        i_old_id,
                                                        f_old_pr,
                                                        i_old_q)
        # remove from order map
        if s_status not in ['New', 'Invalid']:
            self.d_order_map.pop(order_aux)
        # update the order map
        if b_sould_update:
            f_qty = int(order_aux['total_qty_order'])
            f_prior_time = d_data['priority_seconds']
            self.d_order_map[order_aux] = {}
            self.d_order_map[order_aux]['price'] = d_data['order_price']
            self.d_order_map[order_aux]['sec_order'] = order_aux.sec_order_id
            self.d_order_map[order_aux]['qty'] = f_qty
            self.d_order_map[order_aux]['main_id'] = order_aux.main_id
            self.d_order_map[order_aux]['priority_seconds'] = f_prior_time
            self.d_order_map[order_aux]['relative_price'] = i_rel_price
            if s_status in ['New', 'Replaced']:
                self.i_qty_rel += f_qty * 1.
                self.i_cum_rel += i_rel_price * 1. * f_qty

        # return that the update was done
        return True

    def _canc_expr_filled_order(self, order_obj, i_old_id, f_old_pr, i_old_q):
        '''
        Update price_tree when passed canceled, expried or filled orders

        :param order_obj: Order Object. The last order in the file
        :param i_old_id: integer. Old id of the order_obj
        :param f_old_pr: float. Old price of the order_obj
        :param i_old_q: integer. Old qty of the order_obj
        '''
        b_break = False
        this_price = self.price_tree.get(f_old_pr)
        if this_price.delete(i_old_id, i_old_q):
            self.price_tree.remove(f_old_pr)
        # remove from order map
        if b_break:
            raise NotImplementedError
        return False

    def _replaced_order(self, order_obj, i_old_id, f_old_pr, i_old_q):
        '''
        Update price_tree when passed replaced orders

        :param order_obj: Order Object. The last order in the file
        :param i_old_id: integer. Old id of the order_obj
        :param f_old_pr: float. Old price of the order_obj
        :param i_old_q: integer. Old qty of the order_obj
        '''
        # remove from the old price
        this_price = self.price_tree.get(f_old_pr)
        if this_price.delete(i_old_id, i_old_q):
            self.price_tree.remove(f_old_pr)
        # insert in the new price
        f_price = order_obj['order_price']
        if not self.price_tree.get(f_price):
            self.price_tree.insert(f_price, PriceLevel(f_price))
        # insert the order in the due price
        this_price = self.price_tree.get(f_price)
        this_price.add(order_obj)
        return True

    def _partially_filled(self, order_obj, i_old_id, f_old_pr, i_old_q):
        '''
        Update price_tree when passed partially filled orders

        :param order_obj: Order Object. The last order in the file
        :param i_old_id: integer. Old id of the order_obj
        :param f_old_pr: float. Old price of the order_obj
        :param i_old_q: integer. Old qty of the order_obj
        '''
        # delete old price, if it is needed
        this_price = self.price_tree.get(f_old_pr)
        if this_price.delete(i_old_id, i_old_q):
            self.price_tree.remove(f_old_pr)

        # add/modify order
        # insert in the new price
        f_price = order_obj['order_price']
        if not self.price_tree.get(f_price):
            self.price_tree.insert(f_price, PriceLevel(f_price))
        this_price = self.price_tree.get(f_price)
        this_price.add(order_obj)
        return True

    def _new_order(self, order_obj):
        '''
        Update price_tree when passed new orders

        :param order_obj: Order Object. The last order in the file
        '''
        # if it was already in the order map
        if order_obj in self.d_order_map:
            i_old_sec_id = self.d_order_map[order_obj]['main_id']
            f_old_price = self.d_order_map[order_obj]['price']
            i_old_qty = self.d_order_map[order_obj]['qty']
            this_price = self.price_tree.get(f_old_price)
            # remove from order map
            self.d_order_map.pop(order_obj)
            if this_price.delete(i_old_sec_id, i_old_qty):
                self.price_tree.remove(f_old_price)
        # insert a empty price level if it is needed
        f_price = order_obj['order_price']
        if not self.price_tree.get(f_price):
            self.price_tree.insert(f_price, PriceLevel(f_price))
        # add the order
        this_price = self.price_tree.get(f_price)
        this_price.add(order_obj)

        return True

    def get_n_top_prices(self, n):
        '''
        Return a dataframe with the N top price levels

        :param n: integer. Number of price levels desired
        '''
        raise NotImplementedError

    def get_n_botton_prices(self, n=5):
        '''
        Return a dataframe with the N botton price levels

        :param n: integer. Number of price levels desired
        '''
        raise NotImplementedError

    def _readline(self):
        '''
        Return a line from the fr_data file if available. Return false
        otherwiese
        '''
        row = self.fr_data.readline()
        if row == '':
            self.fr_data.close()
            return False, False
        self._i_idx += 1
        d_aux = self.parser(row)
        # treat when the line is zero
        # TODO: I should move it to preprocessment step
        if 'order_price' in d_aux:
            if d_aux['order_price'] == 0.:
                while True:
                    row = self.fr_data.readline()
                    d_aux = self.parser(row)
                    if d_aux['order_price'] != 0.:
                        break
        return d_aux, self.parser.last_identification

    def __iter__(self):
        '''
        Return the self as an iterator object. Use next() to check the rows
        '''
        return self

    def next(self):
        '''
        Return the next item from the fr_data in iter process. If there are no
        further items, raise the StopIteration exception
        '''
        d_rtn, last_identification = self._readline()
        if not d_rtn:
            raise StopIteration
        return d_rtn, last_identification


class BidSide(BookSide):
    '''
    The BID side of the limit order book representation
    '''
    def __init__(self, fr_data, i_member=None, b_secure_changes=False):
        '''
        Initialize a BidSide object. Save all parameters as attributes

        :param fr_data: ZipExtFile object. data to read
        :param i_member*: integer. Member number to be used as a filter
        :param b_secure_changes*: boolean.
        '''
        super(BidSide, self).__init__('BID', fr_data, i_member)
        self.b_secure_changes = b_secure_changes

    def get_n_top_prices(self, n, b_return_dataframe=True):
        '''
        Return a dataframe with the N top price levels

        :param n: integer. Number of price levels desired
        :param b_return_dataframe: boolean. If should return a dataframe
        '''
        t_rtn = self.price_tree.nlargest(n)
        if not b_return_dataframe:
            return t_rtn
        df_rtn = pd.DataFrame(t_rtn)
        df_rtn.columns = ['PRICE', 'QTY']
        return df_rtn

    def get_n_botton_prices(self, n, b_return_dataframe=True):
        '''
        Return a dataframe with the N botton price levels

        :param n: integer. Number of price levels desired
        :param b_return_dataframe: boolean. If should return a dataframe
        '''
        t_rtn = self.price_tree.nsmallest(n)
        if not b_return_dataframe:
            return t_rtn
        df_rtn = pd.DataFrame(t_rtn)
        df_rtn.columns = ['PRICE', 'QTY']
        return df_rtn


class AskSide(BookSide):
    '''
    The ASK side of the limit order book representation
    '''
    def __init__(self, fr_data, i_member=None, b_secure_changes=False):
        '''
        Initialize a BidSide object. Save all parameters as attributes

        :param fr_data: ZipExtFile object. data to read
        :param i_member*: integer. Member number to be used as a filter
        :param b_secure_changes*: boolean.
        '''
        super(AskSide, self).__init__('ASK', fr_data, i_member)
        self.b_secure_changes = b_secure_changes

    def get_n_top_prices(self, n, b_return_dataframe=True):
        '''
        Return a dataframe with the N top price levels

        :param n: integer. Number of price levels desired
        :param b_return_dataframe: boolean. If should return a dataframe
        '''
        t_rtn = self.price_tree.nsmallest(n)
        if not b_return_dataframe:
            return t_rtn
        df_rtn = pd.DataFrame(t_rtn)
        df_rtn.columns = ['PRICE', 'QTY']
        return df_rtn

    def get_n_botton_prices(self, n, b_return_dataframe=True):
        '''
        Return a dataframe with the N botton price levels

        :param n: integer. Number of price levels desired
        :param b_return_dataframe: boolean. If should return a dataframe
        '''
        t_rtn = self.price_tree.nlargest(n)
        if not b_return_dataframe:
            return t_rtn
        df_rtn = pd.DataFrame(t_rtn)
        df_rtn.columns = ['PRICE', 'QTY']
        return df_rtn


class LimitOrderBook(object):
    '''
    A limit Order book representation. Keep the book sides synchronized
    '''
    def __init__(self, s_fbid, s_fask, s_instrument, b_secure_changes=False):
        '''
        Initialize a LimitOrderBook object. Save all parameters as attributes

        :param s_instrument: string. name of the instrument of book
        :param s_fbid: string. Path to the bid file
        :param s_fask: string. Path to the ask file
        :param b_secure_changes*: boolean. If should break with corrections
        '''
        # open files
        self.fr_bid, self.archive_bid = open_file(s_fbid)
        self.fr_ask, self.archive_ask = open_file(s_fask)
        self.s_fbid = s_fbid
        self.s_fask = s_fask
        # save opened data files
        self.book_bid = BidSide(self.fr_bid, b_secure_changes=b_secure_changes)
        self.book_ask = AskSide(self.fr_ask, b_secure_changes=b_secure_changes)
        self.s_instrument = s_instrument
        self.f_time = 0
        self.s_time = ''
        self.stop_iteration = False
        self.stop_time = None
        self.last_stop_time = None
        self.i_last_order_id = 0
        # initiate control variables
        self.i_sec_ask = 9999999999999999
        self.i_sec_bid = 9999999999999999
        self._last_priority_id = 0
        self.last_ident_bid = ''
        self.last_ident_ask = ''
        self.i_my_order_id = 0
        self.d_bid = {}  # hold the last information get from the file
        self.d_ask = {}  # hold the last information get from the file
        # initiate loop control variables
        self.i_read_bid = True
        self.i_read_ask = True
        self.i_get_new_bid = True
        self.i_get_new_ask = True
        self.i_sec_bid_greatter = True
        # best prices tracker
        self.f_top_bid = None
        self.f_top_ask = None
        # bext price tracker
        self.obj_best_bid = None
        self.best_bid = None
        self.obj_best_ask = None
        self.best_ask = None
        self.b_secure_changes = b_secure_changes
        # check if should correct the books
        # need to control that because the books sometimes is in a transition
        # state and the algo has to give sometime to all updates take place
        self.i_count_crossed_books = 0
        # holder for OFI and other variables
        self.stats = BookStats(b_keep_data=True)

    @property
    def last_priority_id(self):
        '''
        Access and control the last last_priority_id used by the book. Used
        by the agent when inserting a new limit order
        '''
        return self._last_priority_id

    @last_priority_id.setter
    def last_priority_id(self, new_id):
        '''
        Access and control the last last_priority_id used by the book. Used
        by the agent when inserting a new limit order
        '''
        self._last_priority_id = max(self._last_priority_id, new_id)

    def get_n_top_prices(self, n, b_return_dataframe=False):
        '''
        Return a dataframe with the n top prices of the current order book

        :param n: integer. Number of price levels desired
        '''
        t_rtn1 = self.book_bid.get_n_top_prices(n, b_return_dataframe=False)
        t_rtn2 = self.book_ask.get_n_top_prices(n, b_return_dataframe=False)
        if b_return_dataframe:
            df1 = pd.DataFrame(t_rtn1, columns=['Bid', 'qBid'])
            df2 = pd.DataFrame(t_rtn2, columns=['Ask', 'qAsk'])
            df1 = df1.reset_index(drop=True)
            df2 = df2.reset_index(drop=True)
            df_rtn = df1.join(df2)
            df_rtn = df_rtn.ix[:, ['qBid', 'Bid', 'Ask', 'qAsk']]
            return df_rtn

        return [t_rtn1, t_rtn2]

    def get_best_price(self, s_side):
        '''
        Return the best price of the specified side

        :param s_side: string. The side of the book
        '''
        if s_side == 'BID':
            obj_aux = self.book_bid.get_n_top_prices(1, False)
            if obj_aux:
                return obj_aux[0][0]
        elif s_side == 'ASK':
            obj_aux = self.book_ask.get_n_top_prices(1, False)
            if obj_aux:
                return obj_aux[0][0]

    def get_orders_by_price(self, s_side, f_price=None, b_rtn_obj=False):
        '''
        Recover the orders from a specific price level

        :param s_side: string. The side of the book
        :param f_price*: float. The price level desired. If not set, return
            the best price
        :param b_rtn_obj*: bool. If return the price object or tree of orders
        '''
        # side of the order book
        obj_price = None
        if s_side == 'BID':
            if not f_price:
                f_price = self.get_best_price(s_side)
            obj_price = self.book_bid.price_tree.get(f_price)
        elif s_side == 'ASK':
            if not f_price:
                f_price = self.get_best_price(s_side)
            obj_price = self.book_ask.price_tree.get(f_price)
        # return the order tree
        if obj_price:
            if b_rtn_obj:
                return obj_price
            return obj_price.order_tree

    def get_basic_stats(self):
        '''
        Return the number of price levels and number of orders remain in the
        dictionaries and trees
        '''
        i_n_order_bid = len(self.book_bid.d_order_map.keys())
        i_n_order_ask = len(self.book_ask.d_order_map.keys())
        i_n_price_bid = len([x for x in self.book_bid.price_tree.keys()])
        i_n_price_ask = len([x for x in self.book_ask.price_tree.keys()])
        i_n_order_bid, i_n_order_ask, i_n_price_bid, i_n_price_ask
        d_rtn = {'n_order_bid': i_n_order_bid,
                 'n_order_ask': i_n_order_ask,
                 'n_price_bid': i_n_price_bid,
                 'n_price_ask': i_n_price_ask}
        return d_rtn

    def set_stop_time(self, s_stop_time):
        '''
        Set a stop time to iterator. After is is reached, the time is reset to
        None

        :param s_stop_time: string. Time in seconds to stop the iteration in
            the format HH:MM:SS
        '''
        f_stop_time = float(s_stop_time[0:2])*60**2
        f_stop_time += float(s_stop_time[3:5])*60
        f_stop_time += float(s_stop_time[6:8])
        if len(s_stop_time) > 8:
            f_stop_time += (float(s_stop_time[9:14])/1000.)
        self.last_stop_time = f_stop_time
        self.stop_time = f_stop_time

    def _readline(self):
        '''
        Update the current state of the book depending on the secondary id of
        each side.
        '''
        # read bid
        if self.i_read_bid and self.i_get_new_bid:
            try:
                self.d_bid, self.last_ident_bid = self.book_bid.next()
                if self.last_ident_bid == 'MSG':
                    self.i_get_new_bid = False
                    self.i_sec_bid = long(self.d_bid['idx'])
            except StopIteration:
                self.i_read_bid = False
        # read ask
        if self.i_read_ask and self.i_get_new_ask:
            try:
                self.d_ask, self.last_ident_ask = self.book_ask.next()
                if self.last_ident_ask == 'MSG':
                    self.i_get_new_ask = False
                    self.i_sec_ask = long(self.d_ask['idx'])
            except StopIteration:
                self.i_read_ask = False
        # update the book
        l_msg = []
        if self.last_ident_bid == 'MSG' and self.last_ident_ask == 'MSG':
            if self.i_sec_ask < self.i_sec_bid:
                # check stop time
                if self.stop_time:
                    if self.d_ask['is_today']:
                        if self.d_ask['priority_seconds'] > self.stop_time:
                            self.stop_iteration = True
                            return False
                # update the book
                l_msg = self.update(self.d_ask)
                # mark the ask as 'empty'
                if self.d_ask['action'] == 'history':
                    self.last_priority_id = self.d_ask['priority_indicator']
                    self.i_sec_ask = 9999999999999999
                    self.i_get_new_ask = True
                    self.f_time = self.d_ask['priority_seconds']
                    s_time = self.d_ask['priority_time']
                    s_date = self.d_ask['session_date']
                    self.s_time = '{} {}'.format(s_date, s_time)

            elif self.i_sec_bid < self.i_sec_ask:
                # check stop time
                if self.stop_time:
                    if self.d_bid['is_today']:
                        if self.d_bid['priority_seconds'] > self.stop_time:
                            self.stop_iteration = True
                            return False
                # update the book
                l_msg = self.update(self.d_bid)
                # mark the bid as 'empty'
                if self.d_bid['action'] == 'history':
                    self.last_priority_id = self.d_bid['priority_indicator']
                    self.i_sec_bid = 9999999999999999
                    self.i_get_new_bid = True
                    self.f_time = self.d_bid['priority_seconds']
                    s_time = self.d_bid['priority_time']
                    s_date = self.d_bid['session_date']
                    self.s_time = '{} {}'.format(s_date, s_time)
        # keep the best prices in a variable
        self.i_my_order_id += 1
        i_bid_count = self.book_bid.price_tree.count
        i_ask_count = self.book_ask.price_tree.count
        if i_bid_count > 0 and i_ask_count > 0:
            best_bid = self.book_bid.price_tree.max_item()
            self.obj_best_bid = best_bid[1]
            self.best_bid = (best_bid[0], best_bid[1].i_qty)
            self.book_bid.set_last_best_queue(self.best_bid)
            best_ask = self.book_ask.price_tree.min_item()
            self.obj_best_ask = best_ask[1]
            self.best_ask = (best_ask[0], best_ask[1].i_qty)
            self.book_ask.set_last_best_queue(self.best_ask)
        # return the messages processed
        return l_msg

    def update(self, d_data):
        '''
        Update the book based on the message passed. Return a list of relevant
        updates to be used by external agents

        :param d_data: dictionary. Last message from the Environment
        '''
        # check if should stop iteration
        l_msg_to_env = []
        self.i_last_order_id = max(self.i_last_order_id, d_data['order_id'])
        if d_data['order_side'] == 'Buy Order':
            l_msg = translator.translate_row(self.i_last_order_id,
                                             d_data,
                                             self,
                                             s_side='Buy Order')
            if d_data['action'] == 'history':
                self.d_bid = d_data.copy()
            if self.b_secure_changes:
                s_err = '{} messages on Bid side'
                assert len(l_msg) == 1, s_err.format(len(l_msg))
            for d_aux in l_msg:
                # if the order is not from history, keep it
                if d_aux['action'] == 'history':
                    self.last_priority_id = d_aux['priority_indicator']
                if d_aux['agent_id'] != 10:
                    l_msg_to_env.append(d_aux)
                self.book_bid.update(d_aux)
        elif d_data['order_side'] == 'Sell Order':
            l_msg = translator.translate_row(self.i_last_order_id,
                                             d_data,
                                             self,
                                             s_side='Sell Order')
            if d_data['action'] == 'history':
                self.d_ask = d_data.copy()
            if self.b_secure_changes:
                s_err = '{} messages on Ask side'
                assert len(l_msg) == 1, s_err.format(len(l_msg))
            for d_aux in l_msg:
                # if the order is not from history, keep it
                if d_aux['action'] == 'history':
                    self.last_priority_id = d_aux['priority_indicator']  # ?
                if d_aux['agent_id'] != 10:
                    l_msg_to_env.append(d_aux)
                self.book_ask.update(d_aux)

            if self.b_secure_changes:
                assert len(l_msg) == 1, s_err.format(len(l_msg))
        # check consistency
        b_correct, d_correct = translator.correct_books(self)
        if b_correct:
            l_aux = [self.book_bid, self.book_ask]
            for s_side, func_update in zip(['BID', 'ASK'], l_aux):
                # correct each side
                if d_correct[s_side]:
                    for d_aux in d_correct[s_side]:
                        func_update.update(d_aux)
                        if d_aux['agent_id'] != 10:
                            l_msg_to_env.append(d_aux)
        # update variables
        i_bid_count = self.book_bid.price_tree.count
        i_ask_count = self.book_ask.price_tree.count
        if i_bid_count > 0 and i_ask_count > 0:
            # used to account for ofi
            last_bid = self.best_bid
            last_ask = self.best_ask
            best_bid = self.book_bid.price_tree.max_item()
            best_ask = self.book_ask.price_tree.min_item()
            # update attributes
            self.obj_best_bid = best_bid[1]
            self.best_bid = (best_bid[0], best_bid[1].i_qty)
            self.obj_best_ask = best_ask[1]
            self.best_ask = (best_ask[0], best_ask[1].i_qty)
            # account OFI and other variables
            # NOTE: this single line increased 30 prc the simulation time
            self.stats.update(self, last_bid, last_ask)
        # return the modifications
        return l_msg_to_env

    def __iter__(self):
        '''
        Return the self as an iterator object. Use next() to check the rows
        '''
        return self

    def next(self):
        '''
        Return the next item from the fr_data in iter process. If there are no
        further items, raise the StopIteration exception
        '''
        # read line
        l_msg = self._readline()
        # check if should stop iteration
        if not self.i_read_bid and not self.i_read_ask:
            # close files
            for obj in [self.fr_bid, self.archive_bid, self.fr_ask,
                        self.archive_ask]:
                obj.close()
            # stop iteration
            raise StopIteration
        elif self.stop_iteration:
            # stop iteration due to time set
            self.stop_iteration = False
            self.stop_time = None
            raise StopIteration
        return l_msg
