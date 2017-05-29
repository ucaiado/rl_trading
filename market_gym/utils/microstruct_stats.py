#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement an object to handle the computation of different microstructure stats

@author: ucaiado

Created on 04/24/2017
"""

from util import ElapsedList
from market_gym.config import WEIGHTING_TIME


'''
Begin help functions
'''

'''
End help functions
'''


class BookStats(object):
    '''
    '''
    def __init__(self, b_keep_data=True):
        '''
        ...

        :param b_keep_data: boolean.
        '''
        if b_keep_data:
            self.elapsed_10sec = ElapsedList(i_count=int(WEIGHTING_TIME)+1)
            self.elapsed_10min = ElapsedList(f_elapsed_time=15., i_count=41.)
            self.d_elapsed = {'10sec': self.elapsed_10sec,
                              '10min': self.elapsed_10min}
            self.d_keep = {'MAX': -9999999., 'MIN': 9999999.}
        self.b_keep_data = b_keep_data
        self.i_ofi = 0
        self.l_valid = ['all_ofi', 'delta_ofi', 'rel_price', 'delta_mid',
                        'high_low']

    def update(self, book_obj, last_bid, last_ask):
        '''
        ...

        :param book_obj: Book object.
        :param last_bid: Tuple.
        :param last_ask: Tuple.
        '''
        best_bid = book_obj.book_bid.price_tree.max_item()
        best_ask = book_obj.book_ask.price_tree.min_item()
        # account OFI and other variables
        if last_bid and last_ask:
            f_en = 0.
            if last_bid != book_obj.best_bid:
                if book_obj.best_bid[0] >= last_bid[0]:
                    f_en += book_obj.best_bid[1]
                if book_obj.best_bid[0] <= last_bid[0]:
                    f_en -= last_bid[1]
            if last_ask != book_obj.best_ask:
                if book_obj.best_ask[0] <= last_ask[0]:
                    f_en -= book_obj.best_ask[1]
                if book_obj.best_ask[0] >= last_ask[0]:
                        f_en += last_ask[1]
            self.i_ofi += f_en
            if not self.b_keep_data:
                return False
            # account other variables
            f_mid = (book_obj.best_bid[0] + book_obj.best_ask[0])/2.
            # f_thistime = d_data['priority_seconds']
            f_thistime = book_obj.f_time
            d_keep = {}
            d_keep['OFI'] = self.i_ofi
            d_keep['MAX'] = self.d_keep['MAX']
            d_keep['MIN'] = self.d_keep['MIN']
            d_keep['LAST'] = f_mid
            # accoutn for relative price (very slow)
            # f_tfilter = self.f_time - (10. * 60)
            # f_relall = 0.
            # l_aux = [v['relative_price'] * v['qty'] * 1. for k, v
            #          in self.book_bid.d_order_map.iteritems()
            #          if v['priority_seconds'] >= f_tfilter]
            # l_aux += [v['relative_price'] * v['qty'] * 1. for k, v
            #           in self.book_ask.d_order_map.iteritems()
            #           if v['priority_seconds'] >= f_tfilter]

            # l_aux2 = [v['qty'] * 1. for k, v
            #           in self.book_bid.d_order_map.iteritems()
            #           if v['priority_seconds'] >= f_tfilter]
            # l_aux2 += [v['qty'] * 1. for k, v
            #            in self.book_ask.d_order_map.iteritems()
            #            if v['priority_seconds'] >= f_tfilter]
            # f_relall = np.sum(l_aux)
            # f_relall /= np.sum(l_aux2)
            f_relall_v = (book_obj.book_bid.i_cum_rel +
                          book_obj.book_ask.i_cum_rel)
            f_rel_q = (book_obj.book_bid.i_qty_rel +
                       book_obj.book_ask.i_qty_rel)
            d_keep['RELALL_V'] = f_relall_v
            d_keep['RELALL_Q'] = f_rel_q
            d_keep['TIME'] = f_thistime
            # account for high low
            if f_mid > d_keep['MAX']:
                d_keep['MAX'] = f_mid
            if f_mid < d_keep['MIN']:
                d_keep['MIN'] = f_mid
            # keep values in memory
            self.elapsed_10sec.update(d_keep, f_thistime)
            if self.elapsed_10min.update(d_keep, f_thistime):
                self.d_keep = {'MAX': -9999999., 'MIN': 9999999.}

    def get_elapsed_data(self, s_type):
        '''
        ...

        :param s_type: string.
        :param s_period*: string.
        '''
        # assert s_type in self.l_valid, 'Invalid field required'
        f_rtn = 0.
        if s_type == 'all_ofi':
            f_rtn = self.i_ofi
        elif self.b_keep_data:
            if s_type == 'delta_ofi':
                f_now = self.elapsed_10sec.l[-1]['OFI']
                f_old = self.elapsed_10sec.l[0]['OFI']
                f_rtn = f_now - f_old
            elif s_type == 'rel_price':
                f_rel_now_v = self.elapsed_10min.l[-1]['RELALL_V']
                f_rel_old_v = self.elapsed_10min.l[0]['RELALL_V']
                f_rel_now_q = self.elapsed_10min.l[-1]['RELALL_Q']
                f_rel_old_q = self.elapsed_10min.l[0]['RELALL_Q']
                f_rtn = f_rel_now_v - f_rel_old_v
                f_rtn /= max(1., (f_rel_now_q - f_rel_old_q))
            elif s_type == 'delta_mid':
                # get delta mid
                f_mid_now = self.elapsed_10sec.l[-1]['LAST']
                f_mid_old = self.elapsed_10sec.l[0]['LAST']
                f_rtn = f_mid_now - f_mid_old
            elif s_type == 'high_low':
                # get high low
                f_max = max(d['MAX'] for d in self.elapsed_10min.l)
                f_min = min(d['MIN'] for d in self.elapsed_10min.l)
                f_rtn = float(int((f_max - f_min)/((f_max+f_min)/2.) * 10**4))
        return f_rtn
