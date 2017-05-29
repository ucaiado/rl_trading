#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement different methods to hedge positions and measure the risk of a Zero
cupon bond portfolio

REFERENCE: Nawalkha, S. K; Soto, G. M.; Beliaeva, N. A., "Interest Rate Risk
Modeling, the fixed Income Valuation course". Wiley, 2005


@author: ucaiado

Created on 12/22/2016
"""

import numpy as np
import math
import pandas as pd
import pprint

'''
Begin help functions
'''

'''
End help functions
'''


def update_maxmin(f_frice, a):
    '''
    Update maximum and minimum price observed by the agent while positioned

    :param f_frice: float.
    :param a: agent object.
    '''
    if f_frice > a.current_max_price:
        a.current_max_price = f_frice
    if f_frice < a.current_min_price:
        a.current_min_price = f_frice


class RiskModel(object):
    '''
    A basic risk model representation for a fixed income strategy that measures
    the loss potential and the immunization needs
    '''
    def __init__(self, env, f_portfolio_value=10**6):
        '''
        Initiate a RiskModel object. Save all parameters as attributes

        :param env: Environment object. the environment that uses this object
        :param f_portfolio_value*: float. The total
        '''
        self.env = env
        self.l_hedging_instr = env.l_hedge
        self.s_main = env.s_main_intrument
        self.l_ratios = []
        self.d_dv01 = {}
        self.na_pu = None
        self.na_du = None
        self.f_portfolio_value = f_portfolio_value
        self.s_risk_model = 'BasicModel'
        self.b_stop_trading = False
        self.price_stop_buy = None
        self.price_stop_sell = None

    def reset(self):
        '''
        reset risk model parameters to use in a new simulation
        '''
        self.current_price = None
        self.b_stop_trading = False
        self.price_stop_buy = None
        self.price_stop_sell = None
        self.l_ratios = []
        self.na_pu = None
        self.na_du = None

    def set_ratios(self):
        '''
        Set the DV01 ratios of the pair between the main instrument and the
        others avaiable to hedging
        '''
        # calculate the dv01 for each instrument
        d_aux = {}
        l_rtn = []
        l_du = []
        for s_key, idx in self.env.order_matching.d_map_book_list.iteritems():
            book_obj = self.env.order_matching.l_order_books[idx]
            f_du = self.env.l_du[self.env.order_matching.idx][idx]/252.
            f_price, f_qty = book_obj.best_bid
            f_dv01 = (f_du*10.)/(1. + f_price/100.)**(1. + f_du)
            d_aux[s_key] = f_dv01
            l_du.append(f_du)
        # calculate the ration in relation to the main instrument
        self.d_dv01 = d_aux
        for s_instr in self.l_hedging_instr:
            l_rtn.append(d_aux[s_instr]/d_aux[self.s_main])
        self.l_du = l_du
        return l_rtn

    def portfolio_duration(self, d_position):
        '''
        Return the duration of a portfolio

        :param d_position: dictionary. portfolio to be hedged
        '''
        l_pu = []
        l_pos = []
        l_du = []
        self.last_pu = {}
        self.last_pos = {}
        self.last_du = {}
        for s_key, idx in self.env.order_matching.d_map_book_list.iteritems():
            book_obj = self.env.order_matching.l_order_books[idx]
            f_du = self.env.l_du[self.env.order_matching.idx][idx]
            f_price, f_qty = book_obj.best_bid
            f_pu = 10.**5/(1. + f_price/100.)**(f_du/252.)
            f_pos = -d_position[s_key]['qBid']  # inverto para qty em PU ?
            f_pos -= -d_position[s_key]['qAsk']
            self.last_du[s_key] = f_du
            l_du.append(f_du)
            self.last_pos[s_key] = f_pos
            l_pos.append(f_pos)
            self.last_pu[s_key] = f_pu
            l_pu.append(f_pu)
        return self._get_duration(l_pu, l_du, l_pos)

    def _get_duration(self, l_pu, l_du, l_pos):
        '''
        Calculate the duration for a given position

        :param l_pu: list.
        :param l_du: list.
        :param l_pos: list. final position in each instrument traded
        '''
        na_weight = self._get_weights(l_pu, l_pos)
        return sum(np.array(l_du)/252. * na_weight)

    def _get_weights(self, l_pu, l_pos):
        '''
        Return the positions as portfolio weights

        :param l_pu: list. the PU of each instrument
        :param l_pos: list. final position in each instrument traded (in PU)
        '''
        na_weight = np.array(l_pu) * np.array(l_pos)
        na_weight /= self.f_portfolio_value
        return na_weight

    def get_instruments_to_hedge(self, agent):
        '''
        Return a list of tuples with the instruments and quantities that can be
        used to hedge a given portfolio

        :param agent: Agent object. agent that need to hedge
        '''
        d_position = agent.position
        return self._get_instruments_to_hedge(d_position)

    def _get_instruments_to_hedge(self, d_position):
        '''
        Return a list of tuples with the instruments and quantities that can be
        used to hedge a given portfolio

        :param d_position: dictionary. portfolio in qty of contracts
        '''
        # check the ratios just once
        if not self.l_ratios:
            self.l_ratios = self.set_ratios()
        f_current_duration = self.portfolio_duration(d_position)
        # check were should hedge and what quantity
        f_main_pos = -d_position[self.s_main]['qBid']
        f_main_pos -= -d_position[self.s_main]['qAsk']
        l_hedged_position = []
        l_pos = [f_main_pos]
        l_du = [self.last_du[self.s_main]]
        l_pu = [self.last_pu[self.s_main]]
        for s_instr, f_ratio in zip(self.l_hedging_instr, self.l_ratios):
            if s_instr == self.s_main:
                s_action = 'BUY'
                if f_main_pos < 0:
                    s_action = 'SELL'
                if f_main_pos == 0:
                    return []
                return [(s_action, s_instr, f_main_pos)]
            f_aux_pos = -d_position[s_instr]['qBid']
            f_aux_pos -= -d_position[s_instr]['qAsk']
            l_hedged_position.append(f_aux_pos*f_ratio)
            l_pos.append(f_aux_pos)
            l_du.append(self.last_du[s_instr])
            l_pu.append(self.last_pu[s_instr])
        f_main_position = f_main_pos + sum(np.array(l_hedged_position))
        na_to_hedge = np.array([f_main_position] * len(l_hedged_position))
        na_to_hedge /= np.array(self.l_ratios)
        na_sign = np.sign(na_to_hedge)
        na_mult = 5 * na_sign
        if sum((abs(na_to_hedge)/5) < 1) != 0:
            na_to_hedge = np.ceil(abs(na_to_hedge)/5).astype(int) * na_mult
        else:
            na_to_hedge = np.round(abs(na_to_hedge)/5).astype(int) * na_mult
        l_to_hedge = list(na_to_hedge)
        l_rtn = []
        for idx, s_instr in enumerate(self.l_hedging_instr):
            i_qty = -l_to_hedge[idx]
            if i_qty != 0:
                l_pos_aux = l_pos[:]
                l_pos_aux[idx+1] += i_qty
                f_future_duration = self._get_duration(l_pu, l_du, l_pos_aux)
                f_abs_dur = abs(f_future_duration)
                # if qty is not enough to dicrease the duration, increase it
                if f_abs_dur > 1.2 and f_abs_dur < 3.:
                    i_qty *= 2
                elif f_abs_dur >= 3.:
                    i_qty *= 3
                l_pos_aux = l_pos[:]
                l_pos_aux[idx+1] += i_qty
                f_future_duration = self._get_duration(l_pu, l_du, l_pos_aux)
                # recalculate all
                if abs(f_future_duration) < abs(f_current_duration):
                    # change to rate quantity
                    s_action = 'BUY'
                    if -i_qty < 0:
                        s_action = 'SELL'
                    l_rtn.append((s_action, s_instr, -i_qty))
        return l_rtn


class KRDModel(RiskModel):
    '''
    A key rate duration model representation that uses the KRDs selected to
    decide what instruments sould be used in the immunization of a portfolio

    '''
    def __init__(self, env, l_krd, f_portfolio_value=10**6, s_kind='trava'):
        '''
        Initiate a KRDModel object. Save all parameters as attributes

        :param env: Environment object. the environment that uses this object
        :param l_krd: list. maturity of the key rates used, in years
        :param f_portfolio_value*: float. The total
        '''
        super(KRDModel, self).__init__(env, f_portfolio_value)
        self.s_risk_model = 'KRDModel_{}'.format(s_kind)
        self.l_krd = l_krd
        self.df_ratios = None
        self.l_cmm_target = ['DI1F19', 'DI1F21', 'DI1F23']
        self.s_kind = s_kind

    def portfolio_krd(self, d_position):
        '''
        Return a tuple with the key rate durations of a portfolio and all
        information needed to recalculate it

        :param d_position: dictionary. portfolio to be hedged
        '''
        # recover variables
        f_facevalue = 10.**5
        l_rates = []
        l_pos = []
        l_maturity = []
        l_instrument = []
        for s_key, idx in self.env.order_matching.d_map_book_list.iteritems():
            book_obj = self.env.order_matching.l_order_books[idx]
            l_instrument.append(book_obj.s_instrument)
            f_du = self.env.l_du[self.env.order_matching.idx][idx]
            f_price, f_qty = book_obj.best_bid
            f_pos = -d_position[s_key]['qBid']  # inverto para qty em PU ?
            f_pos -= -d_position[s_key]['qAsk']
            l_maturity.append(f_du/252.)
            l_pos.append(f_pos)
            l_rates.append(f_price)
        # get the key rate duration matrix
        l_exp_pu = [f_facevalue * np.exp(-f_rate/100 * f_mat)
                    for f_rate, f_mat in zip(l_rates, l_maturity)]
        l_pu = [f_facevalue * (1.+f_rate/100)**(-f_mat)
                for f_rate, f_mat in zip(l_rates, l_maturity)]
        l_dPdYP = [f_facevalue * f_mat * np.exp(-f_rate/100 * f_mat)
                   for f_rate, f_mat in zip(l_rates, l_maturity)]
        df_krd = self.key_rates(l_dPdYP, l_exp_pu)

        na_weights = self._get_weights(l_pu, l_pos)
        df_exposure = self._get_krd_exposure(df_krd, na_weights)
        t_rtn = (df_krd, na_weights, df_exposure, l_maturity, l_pos, l_pu,
                 l_instrument)
        return t_rtn

    def _get_krd_exposure(self, df_krd, na_weights):
        '''
        Return the exposure in KRDs based on krds passed and weights

        :param df_krd: data frame. KRD of the instruments traded
        :param na_weights: numpy array. the weight in portfolio of eack KRD
        '''
        df_exposure = pd.Series(df_krd.T.dot(na_weights))
        df_exposure.index = self.l_krd
        return df_exposure

    def key_rates(self, l_dPdYP, l_pu):
        '''
        Return the matrix of key rates durations for the instruments traded
        in the environment

        :param l_dPdYP: list. $\frac{dP * P}{dY}$
        :param l_pu: list. PU of aeach contract
        '''
        # add up the linear contributions $s(t, t_i)\$ for $i=1, 2, ..., m$ to
        # obtain the change in the given zero-coupon rate $\Delta y(t)$
        if isinstance(self.df_ratios, type(None)):
            self._set_linear_contributions()
        df = self.df_ratios
        return df.apply(lambda x: x * np.array(l_dPdYP) / np.array(l_pu),
                        axis=0)

    def get_target_krds(self, l_cmm, d_data, df_krd, s_kind='fly'):
        '''
        Rerturn the target krds pandas serties to be the same of a buttlerfly.

        :param l_cmm: list. instruments used in the butterfly, ordered by matry
        :param d_data: dictionary. maturity and PU of each instrument
        :param s_kind*: string. the kind of target to return
        '''
        # calculate positions
        if s_kind == 'fly':
            f_Qm = 1.  # quantity at the middle of the structure
            f_alpha = (d_data[l_cmm[2]][1] * 1. - d_data[l_cmm[1]][1])
            f_alpha /= (d_data[l_cmm[2]][1] / 1. - d_data[l_cmm[0]][1])
            f_Qs = (f_Qm * f_alpha * d_data[l_cmm[1]][0]) / d_data[l_cmm[0]][0]
            f_Ql = (f_Qm * (1 - f_alpha) * d_data[l_cmm[1]][0])
            f_Ql /= d_data[l_cmm[2]][0]
            l_pos = [-f_Qs, f_Qm, -f_Ql]
        elif s_kind == 'trava':
            l_pu = [d_data[s_key][0] for s_key in l_cmm]
            l_mat = [d_data[s_key][1] for s_key in l_cmm]
            l_pos = [0., 10, 0.]
            na_weights = self._get_weights(l_pu, l_pos)
            f_curr_duration = sum(np.array(l_mat) * na_weights)
            l_pos_aux = []
            for s_key in self.l_hedging_instr:
                f_pu = d_data[s_key][0]
                f_matr = d_data[s_key][1]
                f_dur_aux = 5. * f_pu / self.f_portfolio_value * f_matr
                f_unt = -f_curr_duration / f_dur_aux * 5.
                l_pos_aux.append(f_unt)
            l_pos = [l_pos_aux[0]/20.] + [1.] + [l_pos_aux[1]/20.]
        # calculate targe
        l_p = [d_data[l_cmm[0]][0], d_data[l_cmm[1]][0], d_data[l_cmm[2]][0]]
        na_weights = self._get_weights(l_p, l_pos)
        df_target = pd.Series(df_krd.T.dot(na_weights))
        df_target.index = self.l_krd

        return df_target

    def _set_linear_contributions(self):
        '''
        Define the linear contribution $s(t, t_i)$ made by the change in the
        ith key rate, $\Delta y(t_i)$, to the change in a given zero-coupon
        rate $\Delta y(t)$, according to Nawalkha, 266
        '''
        l_maturity = []
        l_krd = self.l_krd
        # recover data from books
        for s_key, idx in self.env.order_matching.d_map_book_list.iteritems():
            f_du = self.env.l_du[self.env.order_matching.idx][idx]
            l_maturity.append(f_du/252.)
        # create the $s(t, t_i)$ matrix, according to Nawalkha, 266
        l = []
        i_last_idx = len(l_krd) - 1
        for i_list, f_mat in enumerate(l_maturity):
            l.append([])
            for idx in xrange(len(l_krd)):
                f_krd = l_krd[idx]
                if idx == 0:
                    f_krd1 = l_krd[idx+1]
                    if f_mat < f_krd:
                        l[i_list].append(1.)
                    elif f_mat > f_krd1:
                        l[i_list].append(0.)
                    else:
                        l[i_list].append((f_krd1 - f_mat)/(f_krd1-f_krd))
                elif idx == i_last_idx:
                    f_krd_1 = l_krd[idx-1]
                    if f_mat > f_krd:
                        l[i_list].append(1.)
                    elif f_mat < f_krd_1:
                        l[i_list].append(0.)
                    else:
                        l[i_list].append((f_mat - f_krd_1)/(f_krd-f_krd_1))
                else:
                    f_krd1 = l_krd[idx+1]
                    f_krd_1 = l_krd[idx-1]
                    if (f_mat >= f_krd_1) & (f_mat <= f_krd):
                        l[i_list].append((f_mat - f_krd_1)/(f_krd-f_krd_1))
                    elif (f_mat >= f_krd) & (f_mat <= f_krd1):
                        l[i_list].append((f_krd1 - f_mat)/(f_krd1-f_krd))
                    elif (f_mat < f_krd_1) | (f_mat > f_krd1):
                        l[i_list].append(0.)
                    else:
                        l[i_list].append(0.)
        self.df_ratios = pd.DataFrame(l)

    def _get_instruments_to_hedge(self, d_position):
        '''
        Return a list of tuples with the instruments and quantities that can be
        used to hedge a given portfolio (in rate, not PU)

        :param d_position: dictionary. portfolio in qty of contracts
        '''
        # measure the KRDs of the current portfolios
        f_portfolio_value = self.f_portfolio_value
        t_rtn = self.portfolio_krd(d_position)
        df_krd, na_weights, df_expos, l_mat, l_pos, l_pu, l_instr = t_rtn
        d_aux = dict(zip(l_instr, zip(l_pu, l_mat,
                                      np.cumsum(len(l_instr) * [1])-1)))
        df_target = self.get_target_krds(self.l_cmm_target, d_aux, df_krd,
                                         s_kind=self.s_kind)
        # NOTE: Why I am inverting the signal? I dont know
        # ... maybe something related to positions in PU and rates
        df_target *= (l_pos[d_aux[self.l_cmm_target[1]][2]])
        # calculate the current duration and distance for the target in
        # absolute percentage
        f_curr_duration = sum(np.array(l_mat) * na_weights)
        f_curr_abs_target = sum(abs((df_expos-df_target)/df_target))
        # check which hedge will drive the strategy closer to the target
        f_min_abs_target = f_curr_abs_target
        l_rtn = []
        for idx, s_key in enumerate(self.l_hedging_instr):
            f_pu = d_aux[s_key][0]
            f_matr = d_aux[s_key][1]
            f_dur_aux = 5. * f_pu / f_portfolio_value * f_matr
            f_unt = np.round(-f_curr_duration / f_dur_aux)
            if abs(f_unt) > 10e-6:
                s_debug = '\t{}: {:0.2f}, {:0.2f}'
                # limit the number of contracts that can be traded at each time
                i_qty = float(f_unt*5)
                if f_unt > 3.:
                    i_qty = 15.
                elif f_unt < -3.:
                    i_qty = -15.
                # simulate how would be the measures doing the hedge
                # recalculate all
                idx = d_aux[s_key][2]
                l_pos_aux = l_pos[:]
                l_pos_aux[idx] += i_qty
                na_weights_aux = self._get_weights(l_pu, l_pos_aux)
                f_aux_duration = sum(np.array(l_mat) * na_weights_aux)
                df_expos_aux = self._get_krd_exposure(df_krd, na_weights_aux)
                f_aux_abs_target = sum(abs((df_expos_aux-df_target)/df_target))
                # === DEBUG ===
                # print s_debug.format(s_key, f_aux_duration, f_aux_abs_target)
                # =============
                # check the hedge instrument that will drive down the krd most
                if abs(f_aux_duration) < abs(f_curr_duration):
                    if f_aux_abs_target < f_min_abs_target:
                        f_min_abs_target = f_aux_abs_target
                        # the quantity is in PU. So Convert to rate
                        s_action = 'BUY'
                        if -i_qty < 0:
                            s_action = 'SELL'
                        l_rtn = [(s_action, s_key, -i_qty)]

        return l_rtn


class SingleHedgeModel(RiskModel):
    '''
    A SingleHedgeModel model representation that immunize portfolio using just
    one instrument

    '''
    def __init__(self, env, f_portfolio_value=10**6, s_instrument='DI1F19'):
        '''
        Initiate a KRDModel object. Save all parameters as attributes

        :param env: Environment object. the environment that uses this object
        :param l_krd: list. maturity of the key rates used, in years
        :param f_portfolio_value*: float. The total
        '''
        super(SingleHedgeModel, self).__init__(env, f_portfolio_value)
        self.s_risk_model = 'SingleHedgeModel'
        self.l_hedging_instr = [s_instrument]


class GreedyHedgeModel(RiskModel):
    '''
    A GreedyHedgeModel checks if the the market is offering a good deal to
    hedge the agent's position. The immunization is done using a duration
    neutral strategy that used just one instrument. The 'good deal' notion
    should be implemented as something related to price, time or even
    fair-priceness quant struff

    '''
    def __init__(self, env, f_value=10**6, s_instrument='DI1F19',
                 s_fairness='spread'):
        '''
        Initiate a GreedyHedgeModel object. Save all parameters as attributes

        :param env: Environment object. the environment that uses this object
        :param s_fairness*: string. the fair price notion of the agent
        :param f_value*: float. The total value available
        '''
        super(GreedyHedgeModel, self).__init__(env, f_value)
        self.s_fairness = s_fairness
        if s_fairness == 'spread':
            self.func_fair_price = self._compare_to_spread
        elif s_fairness == 'closeout':
            # closeout also should include stoploss?
            self.func_fair_price = self._compare_to_closeout
            s_instrument = env.s_main_intrument
        self.s_risk_model = 'GreedyHedge_{}'.format(s_fairness)
        self.l_hedging_instr = [s_instrument]
        self.main_hedge = s_instrument
        self.f_target = 0.03  # could be smaller when closeout (2 bps?)
        self.f_stop = 0.03
        self.last_txt = ''
        self.current_price = None
        self.f_last_gain = None
        self.f_last_loss = None
        self.price_stop_buy = None
        self.price_stop_sell = None

    def set_gain_loss(self, f_gain, f_loss):
        '''
        Set a target to the agent stop trading on the session

        :param f_gain: float.
        :param f_loss: float.
        '''
        self.f_last_gain = f_gain
        self.f_last_loss = f_loss

    def can_open_position(self, s_side, agent):
        '''
        Check the positions limits of an agent

        :param s_side: string. Side of the trade to check the limit
        :param agent: Agent object. agent that need to hedge
        '''
        if not self.l_ratios:
            self.l_ratios = self.set_ratios()
        # recover position limits
        s_instr = self.env.s_main_intrument
        f_max_pos = agent.max_pos
        f_max_disclosed = agent.max_disclosed_pos
        # calculate the current position
        f_pos = agent.position[s_instr]['qBid']
        f_pos -= agent.position[s_instr]['qAsk']
        f_pos_discl = f_pos + agent.disclosed_position[s_instr]['qBid']
        f_pos_discl -= agent.disclosed_position[s_instr]['qAsk']
        f_pnlt = 0.
        # check if can open position to a specific side
        if s_side == 'ASK':
            if f_pos <= f_max_pos * -1:
                return False
            elif f_pos_discl <= f_max_disclosed * -1:
                return False
        elif s_side == 'BID':
            if f_pos >= f_max_pos:
                return False
            elif f_pos_discl >= f_max_disclosed:
                return False

        return True

    def should_open_at_current_price(self, s_side, agent):
        '''
        '''
        # recover position limits
        s_instr = self.env.s_main_intrument
        f_pnlt = 0.
        if agent.f_pnl < -1500.:
            f_pnlt = self.f_stop / 3. * 3.
        elif agent.f_pnl < -1000.:
            f_pnlt = self.f_stop / 3. * 2
        elif agent.f_pnl < -500.:
            f_pnlt = self.f_stop / 3. * 1.
        # calculate the current position
        f_pos = agent.position[s_instr]['qBid']
        f_pos -= agent.position[s_instr]['qAsk']
        f_pos_discl = f_pos + agent.disclosed_position[s_instr]['qBid']
        f_pos_discl -= agent.disclosed_position[s_instr]['qAsk']
        # recover prices
        book_obj = agent.env.get_order_book(s_instr)
        f_current_bid, i_qbid = book_obj.best_bid
        f_current_ask, i_qask = book_obj.best_ask
        f_bidask_spread = (f_current_ask - f_current_bid)
        # check if there is something wierd in the prices
        if (f_bidask_spread <= 0.005) or (f_bidask_spread > 0.04):
            # print 'wierd bid-ask spread', f_bidask_spread
            return False
        # check if can open position based on the last stop
        if self.price_stop_sell and s_side == 'ASK':
            f_check = self.price_stop_sell
            if f_current_ask >= f_check - f_pnlt:
                if f_current_ask <= f_check + f_pnlt:
                    # print 'last time of stop at ask', f_check
                    return False
        if self.price_stop_buy and s_side == 'BID':
            f_check = self.price_stop_buy
            if f_current_bid >= f_check - f_pnlt:
                if f_current_bid <= f_check + f_pnlt:
                    # print 'last time of stop at bid', f_check
                    return False
        # check if can open positions based on the last price traded
        if f_pos < 0 and s_side == 'ASK':
            l_agent_prices = [f_p for f_p, f_q, d_tob in
                              agent.d_trades[s_instr][s_side]]
            f_min = min(l_agent_prices) - f_pnlt
            f_max = max(l_agent_prices) + f_pnlt
            if f_current_ask >= f_min and f_current_ask <= f_max:
                # print 'same prices at ask', f_current_ask, f_max, f_min
                return False
        elif f_pos > 0 and s_side == 'BID':
            l_agent_prices = [f_p for f_p, f_q, d_tob in
                              agent.d_trades[s_instr][s_side]]
            f_min = min(l_agent_prices) - f_pnlt
            f_max = max(l_agent_prices) + f_pnlt
            if f_current_bid >= f_min and f_current_bid <= f_max:
                # print 'same prices at bid', f_current_bid, f_max, f_min
                return False
        elif f_pos_discl > 0 and s_side == 'ASK':
            f_agent_price = agent.current_open_price
            if abs(f_current_ask - f_agent_price) < 0.005:
                # print 'too low at ask', f_current_ask, f_agent_price
                return False
        elif f_pos_discl < 0 and s_side == 'BID':
            f_agent_price = agent.current_open_price
            if abs(f_current_bid - f_agent_price) < 0.005:
                # print 'too low at bid', f_current_bid, f_agent_price
                return False
        return True

    def should_hedge_open_position(self, agent):
        '''
        Check if the current open position should be hedged

        :param agent: Agent object. agent that need to hedge
        '''
        # recover position limits
        s_instr = self.env.s_main_intrument
        f_pos = agent.position[s_instr]['qBid']
        f_pos -= agent.position[s_instr]['qAsk']
        f_pos_discl = f_pos + agent.disclosed_position[s_instr]['qBid']
        f_pos_discl -= agent.disclosed_position[s_instr]['qAsk']
        # recover price from hedging instrument
        obj_book = self.env.get_order_book(self.main_hedge)
        if f_pos_discl < 0:
            f_price, f_qty = obj_book.best_ask
        elif f_pos_discl > 0:
            f_price, f_qty = obj_book.best_bid
        # check if is fair to mound a spread
        if f_pos_discl != 0 and f_pos != 0:
            s_side = 'ASK'
            if f_pos > 0:
                s_side = 'BID'
            if not self.func_fair_price(f_price, f_pos_discl, agent, s_side):
                return False
            print '.',
        # close out open positions by the current mid
        if s_instr != self.main_hedge:
            obj_book = self.env.get_order_book(s_instr)
            f_ask, f_qty = obj_book.best_ask
            f_bid, f_qty = obj_book.best_bid
            f_mid = (f_ask + f_bid)/2.
            if f_pos_discl < 0:
                f_qty = abs(f_pos_discl)
                f_vol = f_qty * f_mid
                agent.disclosed_position[s_instr]['qBid'] += f_qty
                agent.disclosed_position[s_instr]['Bid'] += f_vol
            elif f_pos_discl > 0:
                f_qty = abs(f_pos_discl)
                f_vol = f_qty * f_mid
                agent.disclosed_position[s_instr]['qAsk'] += f_qty
                agent.disclosed_position[s_instr]['Ask'] += f_vol
        return True

    def get_instruments_to_hedge(self, agent):
        '''
        Return a list of tuples with the instruments and quantities that can be
        used to hedge a given portfolio

        :param agent: Agent object. agent that need to hedge
        '''
        # TODO: if s_fairness==closeout, should "hedge" on the main instrument
        d_position = agent.position
        return self._get_instruments_to_hedge(d_position)

    def should_stop_disclosed(self, agent):
        '''
        Return if the agent should stop the current disclosed position or not

        :param agent: Agent object. agent that need to hedge
        '''
        s_instr = self.env.s_main_intrument
        # calculate the current position
        f_pos = agent.position[s_instr]['qBid']
        f_pos -= agent.position[s_instr]['qAsk']
        f_pos_discl = f_pos + agent.disclosed_position[s_instr]['qBid']
        f_pos_discl -= agent.disclosed_position[s_instr]['qAsk']
        f_agent_price = agent.current_open_price
        if not f_agent_price or f_pos_discl == 0.:
            if self.b_stop_trading:
                agent.done = True
            return False
        f_ref_price = f_agent_price
        # recover prices
        book_obj = agent.env.get_order_book(s_instr)
        f_current_bid, i_qbid = book_obj.best_bid
        f_current_ask, i_qask = book_obj.best_ask
        f_bidask_spread = (f_current_ask - f_current_bid)
        # check if there is something weird with the spread
        if (f_bidask_spread <= 0.005) or (f_bidask_spread > 0.03):
            return False
        # check if should stop to trade
        if self.b_stop_trading:
            return True
        if self.f_last_gain:
            f_pnl = agent.f_pnl - 40.  # due to MtM
            if f_pnl > self.f_last_gain:
                self.b_stop_trading = True
                return True
            elif f_pnl < self.f_last_loss:
                self.b_stop_trading = True
                return True
        # check if should execute the stop gain
        if f_pos_discl > 0:
            update_maxmin(f_current_bid, agent)
            f_ref_price = max(agent.current_max_price, f_ref_price)
            f_loss = f_ref_price - self.f_stop
            if f_current_bid < f_loss:
                if i_qbid <= 600.:
                    return True
            return f_current_bid < f_loss - self.f_stop/2.
        elif f_pos_discl < 0:
            update_maxmin(f_current_ask, agent)
            f_ref_price = min(agent.current_min_price, f_ref_price)
            f_loss = f_ref_price + self.f_stop
            if f_current_ask > f_loss:
                if i_qask <= 600.:
                    return True
            return f_current_ask > f_loss + self.f_stop/2.
        return False

    def _compare_to_spread(self, f_current_price, f_open_pos, agent, s_side):
        '''
        ...

        :param f_current_price: float. The current price in the hedging instr
        :param f_open_pos: float. the current disclosed position
        :param agent: Agent object. agent that need to hedge
        '''
        # short_current_price >= (long_avg_price-avg_spread_price + param)
        if f_open_pos > 0:
            f_param = self.f_target  # NOTE: hard coded
        elif f_open_pos < 0:
            f_param = -self.f_target  # NOTE: hard coded
        s_instr = self.env.s_main_intrument
        s_hedge = self.main_hedge
        # s_side = 'ASK'
        # if f_open_pos > 0:
        #     s_side = 'BID'
        # implement the prices accountability
        idx = int(abs(f_open_pos/agent.order_size))
        l_disclosed = agent.d_trades[s_instr][s_side][-idx:]
        if len(l_disclosed) == 0:
            print 'no disclosed position'
            print '--open'
            pprint.pprint(agent.d_trades)
            print '--position'
            pprint.pprint(agent.position)
            print '--disclosed'
            print agent.disclosed_position
            print '--param'
            print s_side, f_open_pos
            raise NotImplementedError
        f_long_avg_price = 0.
        f_avg_spread = 0.
        f_qtot = 0.
        for f_p, f_q, d_tob in l_disclosed:
            f_long_avg_price += f_p*f_q
            f_qtot += f_q
            f_aux = (d_tob[s_instr]['Ask'] + d_tob[s_instr]['Bid'])/2.
            f_aux -= (d_tob[s_hedge]['Ask'] + d_tob[s_hedge]['Bid'])/2.
            f_avg_spread += f_aux * f_q
        f_long_avg_price /= f_qtot
        f_avg_spread /= f_qtot
        f_fair_price = (f_long_avg_price - f_avg_spread + f_param)
        # keep the price into memory of the agent
        agent.current_open_price = f_long_avg_price

        s_err = 'PRICE: {}, DISCL: {}, AVG SPREAD: {}, MY PRICE: {}'
        s_err += ', CURRNT: {}'
        s_err = s_err.format(f_fair_price, f_open_pos, f_avg_spread,
                             f_long_avg_price, f_current_price)
        if self.last_txt != s_err:
            # print s_err
            self.last_txt = s_err
        if f_open_pos > 0:
            return f_current_price >= f_fair_price
        elif f_open_pos < 0:
            return f_current_price <= f_fair_price

    def _compare_to_closeout(self, f_current_price, f_open_pos, agent, s_side):
        '''
        '''
        # short_current_price >= (long_avg_price-avg_spread_price + param)
        s_instr = self.env.s_main_intrument
        idx = int(abs(f_open_pos/agent.order_size))
        l_disclosed = agent.d_trades[s_instr][s_side][-idx:]
        f_long_avg_price = 0.
        f_avg_spread = 0.
        f_qtot = 0.
        for f_p, f_q, d_tob in l_disclosed:
            f_long_avg_price += f_p*f_q
            f_qtot += f_q
        f_long_avg_price /= f_qtot
        f_avg_spread /= f_qtot
        f_fair_price = (f_long_avg_price + self.f_target)
        # keep the price into memory of the agent
        agent.current_open_price = f_long_avg_price

        s_err = 'POS: {}, MY PRICE: {}, CURRNT: {}, MAX: {}, MIN: {}'
        s_err = s_err.format(f_open_pos, f_long_avg_price, f_current_price,
                             agent.current_max_price, agent.current_min_price)
        if self.last_txt != s_err:
            # print s_err + '\n'
            self.last_txt = s_err
        # recover prices
        book_obj = agent.env.get_order_book(s_instr)
        f_current_bid, i_qbid = book_obj.best_bid
        f_current_ask, i_qask = book_obj.best_ask
        f_bidask_spread = (f_current_ask - f_current_bid)
        # check if there is something wierd in the prices
        if (f_bidask_spread <= 0.005) or (f_bidask_spread > 0.04):
            return False
        # check if should execute the stop gain
        if f_open_pos > 0:
            f_gain = f_long_avg_price + self.f_target
            if f_current_bid >= f_gain:
                if i_qbid <= 400.:
                    return True
            return f_current_bid > f_gain + self.f_target/2.
        elif f_open_pos < 0:
            f_gain = f_long_avg_price - self.f_target
            if f_current_ask <= f_gain:
                if i_qask <= 400.:
                    return True
            return f_current_ask < f_gain - self.f_target/2.
        return False
