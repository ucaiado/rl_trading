#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Library that simulates a limit order book from data given by BmfBovespa and use
it to wrangle and clean the dataset used

@author: ucaiado

Created on 07/05/2016
"""
# import libraries
from bintrees import FastRBTree
import numpy as np
import pandas as pd
from market_gym.lob import parser_data
import zipfile


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
        self.d_msg['org_total_qty_order'] = self.d_msg['total_qty_order']
        f_q1 = self.d_msg['total_qty_order']
        f_q2 = self.d_msg['traded_qty_order']
        self.d_msg['total_qty_order'] = f_q1 - f_q2
        self.order_id = int(d_msg['seq_order_number'])
        self.sec_order_id = int(d_msg['priority_indicator'])
        self.name = d_msg['seq_order_number']
        self.main_id = self.sec_order_id
        if self.sec_order_id == 0:
            self.main_id = self.order_id

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
        self.order_tree = FastRBTree()

    def add(self, order_aux):
        '''
        Insert the information in the tree using the info in order_aux. Return
        is should delete the Price level or not

        :param order_aux: Order Object. The Order message to be updated
        :param i_old_sec_order: Integer. The previous secondary order id
        :param i_old_qty: Integer. The previous order qty
        '''
        # check if the order_aux price is the same of the self
        s_status = order_aux['order_status']
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

        :param order_aux: Order Object. The Order message to be updated
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

    def how_many_rows_read(self):
        '''
        Return the number of rows processed
        '''
        return self._i_idx

    def update(self, d_data, s_last_ident):
        '''
        Update the state of the order book given the data pased

        :param d_data: dict. data from the last row
        :param s_last_ident: string. last identification
        '''
        # check if the information should be processed
        if s_last_ident != 'MSG':
            return False
        # check if should filter out member
        if not self._should_use_it(d_data):
            return False
        # update the book information
        order_aux = Order(d_data)
        s_status = order_aux['order_status']
        b_sould_update = True
        # treat Bovespa files at the begining f the day
        if s_status != 'New':
            try:
                i_old_id = self.d_order_map[order_aux]['main_id']
            except KeyError:
                if s_status == 'Canceled' or s_status == 'Filled':
                    b_sould_update = False
                    s_status = 'Invalid'
                elif s_status == 'Replaced':
                    s_status = 'New'
        # process
        if s_status == 'New':
            b_sould_update = self._new_order(order_aux)
        elif s_status != 'Invalid':
            i_old_id = self.d_order_map[order_aux]['main_id']
            f_old_pr = self.d_order_map[order_aux]['price']
            i_old_q = self.d_order_map[order_aux]['qty']
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
            self.d_order_map[order_aux] = {}
            self.d_order_map[order_aux]['price'] = d_data['order_price']
            self.d_order_map[order_aux]['sec_order'] = order_aux.sec_order_id
            self.d_order_map[order_aux]['qty'] = f_qty
            self.d_order_map[order_aux]['main_id'] = order_aux.main_id

        # return that the update was done
        return True

    def _should_use_it(self, d_data):
        '''
        Check if should use the passed row to update method

        :param d_data: dict. data from the last row
        '''
        if self.i_member:
            if d_data['member'] != self.i_member:
                return False
        return True

    def _canc_expr_filled_order(self, order_obj, i_old_id, f_old_pr, i_old_q):
        '''
        Update price_tree when passed canceled, expried or filled orders

        :param order_obj: Order Object. The last order in the file
        :param i_old_id: integer. Old id of the order_obj
        :param f_old_pr: float. Old price of the order_obj
        :param i_old_q: integer. Old qty of the order_obj
        '''
        this_price = self.price_tree.get(f_old_pr)
        if this_price.delete(i_old_id, i_old_q):
            self.price_tree.remove(f_old_pr)
        # remove from order map
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
    def __init__(self, fr_data, i_member=None):
        '''
        Initialize a BidSide object. Save all parameters as attributes

        :param fr_data: ZipExtFile object. data to read
        :param i_member*: integer. Member number to be used as a filter
        '''
        super(BidSide, self).__init__('BID', fr_data, i_member)

    def get_n_top_prices(self, n, b_return_dataframe=True):
        '''
        Return a dataframe with the N top price levels

        :param n: integer. Number of price levels desired
        :param b_return_dataframe*: boolean. If should return a dataframe
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
        :param b_return_dataframe*: boolean. If should return a dataframe
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
    def __init__(self, fr_data, i_member=None):
        '''
        Initialize a BidSide object. Save all parameters as attributes

        :param fr_data: ZipExtFile object. data to read
        :param i_member*: integer. Member number to be used as a filter
        '''
        super(AskSide, self).__init__('ASK', fr_data, i_member)

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
    def __init__(self, s_fbid, s_fask, fr_trades, i_member=None,
                 b_mount_blklist=False, d_ignore_ask=None, d_ignore_bid=None):
        '''
        Initialize a LimitOrderBook object. Save all parameters as attributes

        :param s_fbid: string. Path to the bid file
        :param s_fask: string. Path to the ask file
        :param s_ftrades: string. Path to the trades file
        :param i_member*: integer. Member number to be used as a filter
        :param b_mount_blklist*: boolean. If should mount the black list.
        :param d_ignore_ask*: dictionary.
        :param d_ignore_bid*: dictionary.
        '''
        # open files
        self.fr_bid, self.archive_bid = open_file(s_fbid)
        self.fr_ask, self.archive_ask = open_file(s_fask)
        # save opened data files
        self.book_bid = BidSide(self.fr_bid, i_member=i_member)
        self.book_ask = AskSide(self.fr_ask, i_member=i_member)
        self.fr_trades = fr_trades
        self.f_time = 0
        self.s_time = ''
        self.stop_iteration = False
        self.stop_time = None
        # initiate control variables
        self.i_sec_ask = 9999999999999999
        self.i_sec_bid = 9999999999999999
        self.last_ident_bid = ''
        self.last_ident_ask = ''
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
        # save if should mark messages as invalid
        self.b_mount_blklist = b_mount_blklist
        self.d_wrongbid = {}
        self.d_wrongask = {}
        self.d_warningbid = {}
        self.d_warningask = {}
        self.b_crossed = False
        # save orders to ignore
        if d_ignore_ask or d_ignore_bid:
            b_mount_blklist = False
        self.d_ignore_ask = d_ignore_ask
        self.d_ignore_bid = d_ignore_bid

    def get_top_five_prices(self):
        '''
        Return a dataframe with the n top prices of the current order book

        :param n: integer. Number of price levels desired
        '''
        t_rtn1 = self.book_bid.get_n_top_prices(5, b_return_dataframe=False)
        t_rtn2 = self.book_ask.get_n_top_prices(5, b_return_dataframe=False)

        return [t_rtn1, t_rtn2]

    def get_n_top_prices(self, n):
        '''
        Return a dataframe with the n top prices of the current order book

        :param n: integer. Number of price levels desired
        '''
        t_rtn1 = self.book_bid.get_n_top_prices(n, b_return_dataframe=False)
        t_rtn2 = self.book_ask.get_n_top_prices(n, b_return_dataframe=False)
        df1 = pd.DataFrame(t_rtn1, columns=['Bid', 'qBid'])
        df2 = pd.DataFrame(t_rtn2, columns=['Ask', 'qAsk'])
        # exclude zero rows
        df1.qBid = [x.i_qty for x in df1.qBid]
        df2.qAsk = [x.i_qty for x in df2.qAsk]
        df1 = df1.ix[df1.qBid != 0, :]
        df2 = df2.ix[df2.qAsk != 0, :]
        # filter the top prices
        df1 = df1.ix[df1.Bid <= self.f_top_bid, :]
        df2 = df2.ix[df2.Ask >= self.f_top_ask, :]
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        df_rtn = df1.join(df2)
        df_rtn = df_rtn.ix[:, ['qBid', 'Bid', 'Ask', 'qAsk']]

        return df_rtn

    def get_n_best_prices(self, n, f_err=None, b_excludezeros=False):
        '''
        Return a dataframe with the n best prices of the current order book

        :param n: integer. Number of price levels desired
        :param f_err*: float. error related to the last price, if is set
        :param b_excludezeros*: boolean. exclude zeros from dataframe
        '''
        t_rtn1 = self.book_bid.get_n_top_prices(n, b_return_dataframe=False)
        t_rtn2 = self.book_ask.get_n_top_prices(n, b_return_dataframe=False)
        df1 = pd.DataFrame(t_rtn1, columns=['Bid', 'qBid'])
        df2 = pd.DataFrame(t_rtn2, columns=['Ask', 'qAsk'])
        if b_excludezeros:
            # exclude zero rows
            df1.qBid = [x.i_qty for x in df1.qBid]
            df2.qAsk = [x.i_qty for x in df2.qAsk]
            df1 = df1.ix[df1.qBid != 0, :]
            df2 = df2.ix[df2.qAsk != 0, :]
        if f_err:
            df1 = df1.ix[df1.Bid <= (self.book_bid.last_price + f_err), :]
            df2 = df2.ix[df2.Ask >= (self.book_ask.last_price - f_err), :]
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        df_rtn = df1.join(df2)
        df_rtn = df_rtn.ix[:, ['qBid', 'Bid', 'Ask', 'qAsk']]

        return df_rtn

    def get_first_member_price(self, n, i_member, f_err, i_max=3):
        '''
        Return the best bid and offer from the i_member passed

        :param n: integer. Number of orders to filter
        :param i_member: integer. the member desired
        :param f_err: float. error related to the last price
        :param i_max*: integer. The maximum number of order in a unique price
        '''
        df_book = self.get_n_best_prices(50, f_err=f_err)
        l_bid = []
        l_ask = []
        # filter bid orders
        for f_price in df_book.Bid.values:
            if len(l_bid) == n:
                break
            x = self.book_bid.price_tree.get(f_price)
            i_count = 0
            for order_key in x.order_tree:
                if len(l_bid) == n:
                    break
                if i_count >= i_max:
                    break
                order_aux = x.order_tree.get(order_key)
                if order_aux['member'] == i_member:
                    i_count += 1
                    d_aux = order_aux.d_msg.copy()
                    d_aux['total_at_price'] = x.i_qty
                    d_aux['last_time'] = self.s_time
                    l_bid.append(d_aux)
        # filter bid orders
        for f_price in df_book.Ask.values:
            if len(l_ask) == n:
                break
            x = self.book_ask.price_tree.get(f_price)
            i_count = 0
            for order_key in x.order_tree:
                if len(l_ask) == n:
                    break
                if i_count >= i_max:
                    break
                order_aux = x.order_tree.get(order_key)
                if order_aux['member'] == i_member:
                    i_count += 1
                    d_aux = order_aux.d_msg.copy()
                    d_aux['total_at_price'] = x.i_qty
                    d_aux['last_time'] = self.s_time
                    l_ask.append(d_aux)
        return l_bid, l_ask

    def get_order_book(self, i_n_filter=30):
        '''
        Return the 5 best prices of the current order book

        :param i_n_filter*: integer. Number of queues in the best price filter
        '''
        df_book = self.get_n_best_prices(i_n_filter)
        na_ask = df_book['Ask'].values
        i_last_bid = -1
        f_best_bid = None
        f_best_ask = None
        for x in df_book.iterrows():
            i_now = (~(x[1]['Bid'] > na_ask)).sum()
            if i_now > i_last_bid:
                i_last_bid = i_now
                f_best_bid = x[1]['Bid']
                f_best_ask = na_ask[na_ask > f_best_bid]

        f_best_ask = f_best_ask[0]

        df1 = df_book.ix[:, ['qBid', 'Bid']]
        df2 = df_book.ix[:, ['qAsk', 'Ask']]
        df1 = df1[df1.Bid <= f_best_bid].reset_index(drop=True)
        df2 = df2[df2.Ask >= f_best_ask].reset_index(drop=True)
        df_book = (df1.join(df2)).ix[:, ['qBid', 'Bid', 'Ask', 'qAsk']]
        return df_book

    def get_basic_stats(self):
        '''
        Return the number of price levels and number of orders remain in the
        dictionaries and trees
        '''
        i_n_order_bid = len(self.book_bid.d_order_map.keys())
        i_n_order_ask = len(self.book_ask.d_order_map.keys())
        i_n_price_bid = len([x for x in self.book_bid.price_tree.keys()])
        i_n_price_ask = len([x for x in self.book_ask.price_tree.keys()])
        return i_n_order_bid, i_n_order_ask, i_n_price_bid, i_n_price_ask

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
        self.stop_time = f_stop_time

    def mount_crossed_prices_list(self, s_side):
        '''
        Rerturn if the prices have crossed each other and the inlcude the
        orders envolved to a list

        :param s_side: string. side to seacrh for
        '''
        b_rtn = False
        if not self.b_crossed:
            # if the prices is not crossed yet recover side information
            if s_side == 'ASK':
                row = self.d_ask.copy()
                d_invalid = self.d_warningask
                b_rtn = self.f_top_bid > row['order_price']
            else:
                row = self.d_bid.copy()
                d_invalid = self.d_warningbid
                b_rtn = row['order_price'] > self.f_top_ask
            # if the last message cause the prices to crossed
            # mask it in a blacklist
            if b_rtn:
                # count events
                # raise InvalidTypeException('Foo')
                s_key1 = row['seq_order_number']
                s_key2 = row['order_status']
                if s_key1 not in d_invalid:
                    d_invalid[s_key1] = {}
                if s_key2 not in d_invalid[s_key1]:
                    d_invalid[s_key1][s_key2] = 0
                d_invalid[s_key1][s_key2] += 1
        # return if the last message cause the prices to crossed
        return b_rtn

    def mount_blacklist_when_passive(self, s_side):
        '''
        Mark all orders before the order that was executed as invalid

        :param s_side: string. side to seacrh for
        '''
        # recover last information
        if s_side == 'ASK':
            row = self.d_ask.copy()
            d_invalid = self.d_wrongask
        else:
            row = self.d_bid.copy()
            d_invalid = self.d_wrongbid
        # check everytthing that is before the price of this message
        gen_bk = None
        if row['order_side'] == 'Buy Order':
            l_x = self.book_bid.get_n_top_prices(1, b_return_dataframe=False)
            if len(l_x) > 0:
                f_best_price = l_x[0][0]
                f_max = f_best_price + 0.01
                f_min = row['order_price']
                gen_bk = self.book_bid.price_tree.item_slice(f_min,
                                                             f_max,
                                                             reverse=True)
        else:
            l_x = self.book_ask.get_n_top_prices(1, b_return_dataframe=False)
            if len(l_x) > 0:
                f_best_price = l_x[0][0]
                f_max = row['order_price'] + 0.01
                f_min = f_best_price
                gen_bk = self.book_ask.price_tree.item_slice(f_min,
                                                             f_max,
                                                             reverse=False)
        # include messages in the blacklist
        b_stop = False
        if not gen_bk:
            return None
        for f_price, obj_price in gen_bk:
            # if f_price == row['order_price']:
            #     break
            # assert obj_price.order_tree.count <= 2, 'More than two offers'
            for idx_ord, obj_order in obj_price.order_tree.nsmallest(1000):
                # check if the order Id is different from the order last traded
                d_compare = obj_order.d_msg
                if row['seq_order_number'] != d_compare['seq_order_number']:
                    # and cancel them
                    # print row
                    # raise InvalidTypeException('Foo')
                    s_key1 = obj_order.d_msg['seq_order_number']
                    s_key2 = obj_order.d_msg['order_status']
                    if s_key1 not in d_invalid:
                        d_invalid[s_key1] = {}
                    if s_key2 not in d_invalid[s_key1]:
                        d_invalid[s_key1][s_key2] = 0
                    d_invalid[s_key1][s_key2] += 1
                else:
                    b_stop = True
                    break
            if b_stop:
                break

    def mount_blacklist_when_active(self, s_side):
        '''
        Mark all orders with better prices than the trade as invalid

        :param s_side: string. side to seacrh for
        '''
        # recover last information
        if s_side == 'ASK':
            row = self.d_ask.copy()
            d_invalid = self.d_wrongask
        else:
            row = self.d_bid.copy()
            d_invalid = self.d_wrongbid
        # check everytthing that is before the price of this message
        gen_bk = None
        if row['order_side'] == 'Buy Order':
            l_x = self.book_bid.get_n_top_prices(1, b_return_dataframe=False)
            # define a function to compare values

            def _compare(a, b):
                return a['order_price'] < b['order_price']
            # recover order book
            if len(l_x) > 0:
                f_best_price = l_x[0][0]
                f_max = f_best_price + 0.01
                f_min = row['order_price']
                gen_bk = self.book_bid.price_tree.item_slice(f_min,
                                                             f_max,
                                                             reverse=True)
        else:
            l_x = self.book_ask.get_n_top_prices(1, b_return_dataframe=False)

            def _compare(a, b):
                return a['order_price'] > b['order_price']

            if len(l_x) > 0:
                f_best_price = l_x[0][0]
                f_max = row['order_price'] + 0.01
                f_min = f_best_price
                gen_bk = self.book_ask.price_tree.item_slice(f_min,
                                                             f_max,
                                                             reverse=False)
        # include messages in the blacklist
        b_stop = False
        if not gen_bk:
            return None
        for f_price, obj_price in gen_bk:

            # if f_price == row['order_price']:
            #     break
            # assert obj_price.order_tree.count <= 2, 'More than two offers'
            for idx_ord, obj_order in obj_price.order_tree.nsmallest(1000):
                # check if the order Id is different from the order last traded
                d_compare = obj_order.d_msg
                if _compare(row, d_compare):
                    # and cancel them
                    # print row
                    # print
                    # print d_compare
                    # raise InvalidTypeException('Foo')
                    s_key1 = obj_order.d_msg['seq_order_number']
                    s_key2 = obj_order.d_msg['order_status']
                    if s_key1 not in d_invalid:
                        d_invalid[s_key1] = {}
                    if s_key2 not in d_invalid[s_key1]:
                        d_invalid[s_key1][s_key2] = 0
                    d_invalid[s_key1][s_key2] += 1
                else:
                    b_stop = True
                    break
            if b_stop:
                break

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
        if self.last_ident_bid == 'MSG' and self.last_ident_ask == 'MSG':
            if self.i_sec_ask < self.i_sec_bid:
                # check stop time
                if self.stop_time:
                    if self.d_ask['is_today']:
                        if self.d_ask['priority_seconds'] > self.stop_time:
                            self.stop_iteration = True
                            return False
                # update the book
                if self.b_mount_blklist:
                    if self.d_ask['execution_type'] == 'Trade':
                        if self.d_ask['agressor_indicator'] == 'Passive':
                            self.mount_blacklist_when_passive('ASK')
                        elif self.d_ask['execution_type'] == 'New':
                            # pass
                            self.mount_blacklist_when_active('ASK')
                    else:
                        # mark if the last message cause the book to cross
                        self.b_crossed = self.mount_crossed_prices_list('ASK')
                self.book_ask.update(self.d_ask, self.last_ident_ask)
                # mark the ask as 'empty'
                self.i_sec_ask = 9999999999999999
                self.i_get_new_ask = True
                self.f_time = self.d_ask['priority_seconds']
                self.s_time = self.d_ask['priority_time']

            elif self.i_sec_bid < self.i_sec_ask:
                # check stop time
                if self.stop_time:
                    if self.d_bid['is_today']:
                        if self.d_bid['priority_seconds'] > self.stop_time:
                            self.stop_iteration = True
                            return False
                # update the book
                if self.b_mount_blklist:
                    if self.d_bid['execution_type'] == 'Trade':
                        if self.d_bid['agressor_indicator'] == 'Passive':
                            self.mount_blacklist_when_passive('BID')
                        elif self.d_bid['execution_type'] == 'New':
                            # pass
                            self.mount_blacklist_when_active('BID')
                    else:
                        # mark if the last message cause the book to cross
                        self.b_crossed = self.mount_crossed_prices_list('BID')
                self.book_bid.update(self.d_bid, self.last_ident_bid)
                # mark the bid as 'empty'
                self.i_sec_bid = 9999999999999999
                self.i_get_new_bid = True
                self.f_time = self.d_bid['priority_seconds']
                self.s_time = self.d_bid['priority_time']

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
        self._readline()
        # update the best bid and best offer
        # TODO: Still need to make the price decreases. Now, just a trade will
        # make the price going down
        if self.book_bid.last_price != 0 and self.book_ask.last_price != 0:
            # if the last info is a trade and the agressor is passive, it is
            # the best price. Else, test the new and replaced orders on Ask
            if self.d_ask['execution_type'] == 'Trade':
                if self.d_ask['agressor_indicator'] == 'Passive':
                    self.f_top_ask = self.d_ask['order_price']
            elif self.d_ask['execution_type'] in ['New', 'Replaced']:
                if self.f_top_bid:
                    f_price = self.d_ask['order_price']
                    if f_price >= self.f_top_bid:
                        if f_price <= self.f_top_ask:
                            self.f_top_ask = f_price
            # if the last info is a trade and the agressor is passive, it is
            # the best price. Else, test the new and replaced orders on Bid
            if self.d_bid['execution_type'] == 'Trade':
                if self.d_bid['agressor_indicator'] == 'Passive':
                    self.f_top_bid = self.d_bid['order_price']
            elif self.d_ask['execution_type'] in ['New', 'Replaced']:
                if self.f_top_ask:
                    f_price = self.d_bid['order_price']
                    if f_price <= self.f_top_ask:
                        if f_price >= self.f_top_bid:
                            self.f_top_bid = f_price
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
        return True
