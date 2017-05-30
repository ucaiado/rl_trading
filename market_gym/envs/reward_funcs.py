#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement the reward functions to be used by the Environment.

When implementing a new functiono, always include the name of the reward
function in the atribute "implemented", from RewardFunc class. Then, include it
in the set_func and as the others and  implement a method with an underscore
before is, as "_pnl" to the "pnl" reward function.

The RewardFunc class is intended to keep the Environment class less cluttered.

@author: ucaiado

Created on 03/05/2017
"""
import numpy as np
from market_gym import RewardWrapper


class RewardFunc(RewardWrapper):
    '''
    RewardFunc is a wrapper for different functions used by the environment to
    assess the value of the agent's actions
    '''
    implemented = ['pnl', 'ofi', 'pnl_pos', 'ofi_pnl', 'ofi_only',
                   'pnl_greedy', 'ofi_new', 'ofi_pnl_new']

    def __init__(self):
        '''
        Initiate a RewardFunc object. Save all parameters as attributes
        '''
        super(RewardFunc, self).__init__()

    def reset(self):
        '''
        '''
        self.log_info = {}
        self.last_time_closed = 0.
        self.last_pos = 0.
        if self.s_type in ['ofi_new', 'ofi_pnl_new']:
            l_main_keys = ['by_pnl', 'holding_pos', 'bb1', 'bb2', 'bb3', 'bid',
                           'ask', 'none_pos', 'none_no_pos1', 'none_no_pos2']
            self.log_info = {'rwd': dict(zip(l_main_keys, [0. for x in
                                                           l_main_keys])),
                             'num': dict(zip(l_main_keys, [0. for x in
                                                           l_main_keys]))}
        # reset any other variable here
        pass

    def set_func(self, s_type):
        '''
        Set the reward function used when the get_reward is called
        '''
        if s_type == 'pnl':
            self.s_type = s_type
            self.reward_fun = self._pnl
        elif s_type == 'ofi':
            self.s_type = s_type
            self.reward_fun = self._ofi
        elif s_type == 'pnl_pos':
            self.s_type = s_type
            self.reward_fun = self._pnl_pos
        elif s_type == 'ofi_pnl':
            self.s_type = s_type
            self.reward_fun = self._ofi_pnl
        elif s_type == 'ofi_only':
            self.s_type = s_type
            self.reward_fun = self._ofi_only
        elif s_type == 'ofi_new':
            self.s_type = s_type
            self.reward_fun = self._ofi_new
        elif s_type == 'pnl_greedy':
            self.s_type = s_type
            self.reward_fun = self._pnl_greedy
        elif s_type == 'ofi_pnl_new':
            self.s_type = s_type
            self.reward_fun = self._ofi_pnl_new

    def _pnl_greedy(self, e, s, a, pnl, inputs):
        '''
        Return the reward based on PnL from the last step marked to the
        mid-price of the instruments traded

        :param e: Environment object. Environment where the agent operates
        :param a: Agent object. the agent that will perform the action
        :param s: dictionary. The inputs from environment to the agent
        :param pnl: float. The current pnl of the agent
        :param inputs: dictionary. The inputs from environment to the agent
        '''
        # calculate the current PnL just in the main instrument
        reward = 0.
        s_cmm = a.env.s_main_intrument
        d_consolidate = {}
        for s_key in a.position[s_cmm]:
            d_consolidate[s_key] = a.position[s_cmm][s_key]
            d_consolidate[s_key] += a.disclosed_position[s_cmm][s_key]
        d_consolidate['Position'] = d_consolidate['qBid']
        d_consolidate['Position'] -= d_consolidate['qAsk']
        f_pnl_rwrd = 0.
        l_pu = [None for x in a.env.l_instrument]
        l_du = a.env.l_du[a.env.order_matching.idx]
        for s_instr, f_du, f_pu in zip(a.env.l_instrument, l_du, l_pu):
            if s_instr == s_cmm:
                f_pu_bid = 0.
                if d_consolidate['qBid'] != 0:
                    f_pu_bid = d_consolidate['Bid'] / d_consolidate['qBid']
                    f_pu_bid = 10**5 * (1+f_pu_bid/100.)**(-f_du/252.)
                f_pu_ask = 0.
                if d_consolidate['qAsk'] != 0:
                    f_pu_ask = d_consolidate['Ask'] / d_consolidate['qAsk']
                    f_pu_ask = 10**5 * (1+f_pu_ask/100.)**(-f_du/252.)
                f_pnl_rwrd += f_pu_bid * d_consolidate['qBid']
                f_pnl_rwrd -= f_pu_ask * d_consolidate['qAsk']
                f_qty = d_consolidate['qAsk'] + d_consolidate['qBid']
                f_pu_mid = 0.
                if f_pu:
                    f_pu_mid = f_pu
                elif inputs['midPrice'][s_instr] != 0:
                    f_pu_mid = inputs['midPrice'][s_instr]
                    f_pu_mid = 10**5 * (1+f_pu_mid/100.)**(-f_du/252.)
                # print d_consolidate
                f_pnl_rwrd += -d_consolidate['Position'] * f_pu_mid
                # include costs
                f_pnl_rwrd -= (f_qty * 0.80)

        f_old_pnl = s['Rwrd_Pnl']
        # first part of reward function: The monetized OFI
        reward = np.around(f_pnl_rwrd - f_old_pnl, 2)
        # update the OFI_Pnl state
        s['Rwrd_Pnl'] = f_pnl_rwrd

        return reward

    def _pnl_pos(self, e, s, a, pnl, inputs):
        '''
        Return the reward based on PnL from the last step marked to the
        mid-price of the instruments traded

        :param e: Environment object. Environment where the agent operates
        :param a: Agent object. the agent that will perform the action
        :param s: dictionary. The inputs from environment to the agent
        :param pnl: float. The current pnl of the agent
        :param inputs: dictionary. The inputs from environment to the agent
        '''
        reward = self._pnl(e, s, a, pnl, inputs)
        s_main = e.s_main_intrument
        if not a.logged_action:
            return reward
        f_penalty = abs(e.agent_states[a][s_main]['Position']) * 0.02
        f_penalty += abs(np.around(a.log_info['duration'])) * 0.30
        return reward - f_penalty

    def _ofi_pnl(self, e, s, a, pnl, inputs):
        '''
        Return the reward based on PnL from the last step marked to the
        mid-price of the instruments traded

        :param e: Environment object. Environment where the agent operates
        :param a: Agent object. the agent that will perform the action
        :param s: dictionary. The inputs from environment to the agent
        :param pnl: float. The current pnl of the agent
        :param inputs: dictionary. The inputs from environment to the agent
        '''
        reward = self._pnl(e, s, a, pnl, inputs)
        reward += self._ofi_only(e, s, a, pnl, inputs)
        return reward

    def _ofi_new(self, e, s, a, pnl, inputs, use_pnl=False):
        '''
        Return the reward based on scaled OFI from the last step of the main
        instruments traded

        :param e: Environment object. Environment where the agent operates
        :param a: Agent object. the agent that will perform the action
        :param s: dictionary. The inputs from environment to the agent
        :param pnl: float. The current pnl of the agent
        :param inputs: dictionary. The inputs from environment to the agent
        '''
        # make sure that it is possible to account ofi changes
        reward = 0.
        if 'to_delta' not in a.log_info:
            return reward
        # NOTE: as it is measuring how much OFI accumulated from the last
        # update give this reward just when the agent selects a new action
        if not a.b_new_reward:
            return reward
        # measure the change in OFI in the main instrument between each step
        # NOTE: remember that is the amount between each time step. It is not
        # related to time, actually
        s_cmm = a.env.s_main_intrument
        f_dv01 = a.risk_model.d_dv01[s_cmm] * 5.
        f_pos = self.get_agent_pos(e, s, a, pnl, inputs)
        f_ofi_now = float(inputs['OFI'][s_cmm])
        f_ofi_old = float(a.log_info['to_delta']['OFI'][s_cmm])
        f_delta = a.bound_values(f_ofi_now - f_ofi_old, 'ofi_new', s_cmm)
        # measure the reward by ofi
        s_last_action = a.last_action
        # f_when_keep_both_is_good = 0.40
        f_when_keep_both_is_good = 0.20
        f_when_keep_both_is_not_so_bad = 0.50
        if s_last_action == 'BEST_BOTH':
            if abs(f_delta) <= f_when_keep_both_is_good:
                # scaling between 0 and 0.5.
                f_this_rwd = abs(f_delta)/f_when_keep_both_is_good/2.
                reward += f_this_rwd
                self.log_info['rwd']['bb1'] += f_this_rwd
                self.log_info['num']['bb1'] += 1
            elif abs(f_delta) <= f_when_keep_both_is_not_so_bad:
                f_aux = f_when_keep_both_is_not_so_bad
                f_aux -= f_when_keep_both_is_good
                f_aux2 = abs(f_delta) - f_when_keep_both_is_good
                f_this_rwd = f_aux2/f_aux/2  # scaling between 0 and 0.5.
                reward += -f_this_rwd
                self.log_info['rwd']['bb2'] += -f_this_rwd
                self.log_info['num']['bb2'] += 1
            else:
                f_aux = 1. - f_when_keep_both_is_not_so_bad
                f_aux2 = abs(f_delta) - f_when_keep_both_is_not_so_bad
                f_this_rwd = f_aux2/f_aux/2 + 0.03
                reward += -f_this_rwd
                self.log_info['rwd']['bb3'] += -f_this_rwd
                self.log_info['num']['bb3'] += 1
        elif s_last_action == 'BEST_BID':
            reward += f_delta * 1.05
            self.log_info['rwd']['bid'] += f_delta * 1.05
            self.log_info['num']['bid'] += 1
        elif s_last_action == 'BEST_OFFER':
            reward += -f_delta * 1.05
            self.log_info['rwd']['ask'] += -f_delta * 1.05
            self.log_info['num']['ask'] += 1
        elif isinstance(s_last_action, type(None)):
            # give the agent a reward to not keep orders in the best prices?
            if f_pos == 0.:
                # it is just valid if the agent is not doing hedge
                self.last_pos = 0.
                self.last_time_closed = e.order_matching.f_time
                if abs(f_delta) >= 0.75:
                    f_mult = self.log_info['num']['none_no_pos1']
                    f_mult = max(1., np.exp(f_mult/200.))
                    f_aux = (abs(f_delta) - 0.75) / (1. - 0.75)
                    f_this_rwd = f_aux/f_mult
                    reward += f_this_rwd
                    self.log_info['rwd']['none_no_pos1'] += f_this_rwd
                    self.log_info['num']['none_no_pos1'] += 1
                elif abs(f_delta) > 0.3 and abs(f_delta) < 0.75:
                    reward += 0.
                else:
                    f_mult = self.log_info['num']['none_no_pos2']
                    f_mult = max(1., np.exp(f_mult/200.))
                    f_aux = (abs(f_delta) - 0.) / (0.15 - 0.) / 2.
                    f_this_rwd = f_aux/f_mult
                    reward += -f_this_rwd
                    self.log_info['rwd']['none_no_pos2'] += -f_this_rwd
                    self.log_info['num']['none_no_pos2'] += 1
            elif f_pos != 0:
                # additional penalty for not having an order, holding positions
                f_mult = self.log_info['num']['none_pos']
                f_mult = max(1., np.exp(f_mult/200.))
                f_aux = abs(f_delta)/5.
                if abs(f_delta) < 0.1:
                    f_aux = 0.
                if (f_pos > 0. and f_delta > 0.8):
                    f_mult *= -1
                elif (f_pos < 0. and f_delta < -0.8):
                    f_mult *= -1
                f_this_rwd = f_aux / f_mult
                reward += -f_this_rwd
                self.log_info['rwd']['none_pos'] += -f_this_rwd
                if abs(f_delta) >= 0.1:
                    self.log_info['num']['none_pos'] += 1
        # penalty by pnl side
        if use_pnl:
            f_delta_pnl = self._pnl_greedy(e, s, a, pnl, inputs)
            if f_delta_pnl != 0.:
                f_this_rwd = np.abs(f_delta_pnl)/f_dv01
                # f_this_rwd = 1. / (1+np.exp(-0.2 * (f_this_rwd/25. - 12.)))
                # f_this_rwd = (f_this_rwd - 0.0831726965) * 9.
                f_this_rwd *= np.sign(f_delta_pnl)
                # print 'pnl: {}, reward: {}'.format(f_delta_pnl, f_this_rwd)
                reward += f_this_rwd
                self.log_info['rwd']['by_pnl'] += f_this_rwd
                self.log_info['num']['by_pnl'] += 1
        # give a penalty by holding positions
        if f_pos != 0:
            f_opened_time = e.order_matching.f_time - self.last_time_closed
            f_mult = 0.015 / (1.+np.exp(-4.6 * (f_opened_time/600. - 0.5)))
            f_this_rwd = int(abs(f_pos)/5) * f_mult
            reward += -f_this_rwd
            self.log_info['rwd']['holding_pos'] += -f_this_rwd
            self.log_info['num']['holding_pos'] += 1

        e.i_COUNT += 1
        return reward

    def _ofi_pnl_new(self, e, s, a, pnl, inputs):
        '''
        Return the reward based on scaled OFI from the last step of the main
        instruments traded

        :param e: Environment object. Environment where the agent operates
        :param a: Agent object. the agent that will perform the action
        :param s: dictionary. The inputs from environment to the agent
        :param pnl: float. The current pnl of the agent
        :param inputs: dictionary. The inputs from environment to the agent
        '''
        return self._ofi_new(e, s, a, pnl, inputs, use_pnl=True)

    def _ofi_only(self, e, s, a, pnl, inputs):
        '''
        Return the reward based on PnL from the last step marked to the
        mid-price of the instruments traded

        :param e: Environment object. Environment where the agent operates
        :param a: Agent object. the agent that will perform the action
        :param s: dictionary. The inputs from environment to the agent
        :param pnl: float. The current pnl of the agent
        :param inputs: dictionary. The inputs from environment to the agent
        '''
        # first part: pnl
        d_input = inputs
        reward = 0.
        # make sure that it is possible to account ofi changes
        if 'to_delta' not in a.log_info:
            return reward
        if not a.older_code:
            return reward
        # second part of the reward: The good placement
        # NOTE: as it is measuring how much OFI accumulated from the last
        # update give this reward just when the agent selects a new action
        if not a.b_new_reward:
            return reward
        s_compare = a.older_code
        # measure the reward per step according to the time between them
        f_tupdates = max(a.f_min_time, a.delta_toupdate)
        f_tot_sec = 19800.  # number of second in the day
        f_n_steps = f_tot_sec / f_tupdates  # num of rewards given
        f_w_right_ofi = 1.
        f_w_right_price_change = 1.5
        f_number_ofbets_per_step = len(a.l_instr_to_bet) * 1.
        f_w_tot = (f_w_right_price_change + f_w_right_ofi)
        f_w_tot *= f_number_ofbets_per_step
        f_ntotbets = f_n_steps * f_w_tot
        f_max_reward_in_a_day = 10.**4 * 3  # max reward in perfect behvior
        f_val = f_max_reward_in_a_day/f_ntotbets
        # measure the reward by ofi
        l_cmm = a.l_instr_to_bet
        for i_idx, s_cmm in zip(xrange(len(s_compare)), l_cmm):
            # check the side of the bet
            i_bet = a.d_bets[s_compare[i_idx]]
            # measure the change in OFI
            i_ofi_now = d_input['OFI'][s_cmm]
            i_ofi_old = a.log_info['to_delta']['OFI'][s_cmm]
            i_delta = i_ofi_now - i_ofi_old
            # if the agent is right
            f_aux = ((i_delta < 0) * (i_bet < 0)) * 1.
            f_aux += ((i_delta > 0) * (i_bet > 0)) * 1.
            f_aux += ((i_delta == 0) * (i_bet == 0)) * 1.
            # if the agent is wrong
            f_aux += ((i_delta < 0) * (i_bet > 0)) * -1.
            f_aux += ((i_delta > 0) * (i_bet < 0)) * -1.
            f_aux += ((i_delta == 0) * (i_bet != 0)) * -1.
            # give the reward
            if f_aux > 0:
                reward += 1. * f_val * f_w_right_ofi
            elif f_aux < 0:
                reward += -0.7 * f_val * f_w_right_ofi
            # measure the reward by side of mid-price
            if d_input['midPrice'][s_cmm] != 0:
                f_current_mid = d_input['midPrice'][s_cmm]
                f_mid_new = a.log_info['to_delta']['TOB'][s_cmm]['Ask']
                f_mid_new += a.log_info['to_delta']['TOB'][s_cmm]['Bid']
                f_delta_mid = f_current_mid - f_mid_new/2.
                i_delta_mid = int(f_delta_mid * 100)
                # if the agent is right
                f_aux = ((i_delta_mid < 0) * (i_bet < 0)) * 1.
                f_aux += ((i_delta_mid > 0) * (i_bet > 0)) * 1.
                f_aux += ((i_delta_mid == 0) * (i_bet == 0)) * 1.
                # if the agent is wrong
                f_aux += ((i_delta_mid < 0) * (i_bet > 0)) * -1.
                f_aux += ((i_delta_mid > 0) * (i_bet < 0)) * -1.
                f_aux += ((i_delta_mid == 0) * (i_bet != 0)) * -1.
                # give the reward
                if f_aux > 0:
                    reward += 1. * f_val * f_w_right_price_change
                elif f_aux < 0:
                    reward += -0.7 * f_val * f_w_right_price_change
        e.i_COUNT += 1
        # print 'rward step: {} {}'.format(self.i_COUNT, a.b_new_reward)
        return reward

    def _ofi(self, e, s, a, pnl, inputs):
        '''
        Return the reward based on PnL from the last step marked to the
        mid-price of the instruments traded

        :param e: Environment object. Environment where the agent operates
        :param a: Agent object. the agent that will perform the action
        :param s: dictionary. The inputs from environment to the agent
        :param pnl: float. The current pnl of the agent
        :param inputs: dictionary. The inputs from environment to the agent
        '''
        d_input = inputs
        # make sure that it is possible to account ofi changes
        if 'to_delta' not in a.log_info:
            return 0.
        if not a.older_code:
            return 0.
        # account the ofi by position and changes
        reward = 0.
        f_pnl_ofi = 0.
        # NOTE: HARD CODED
        l_cmm = a.l_instr_to_bet
        d_deltas = {}
        for s_key in l_cmm:
            i_ofi_now = d_input['OFI'][s_key]
            i_ofi_old = a.log_info['to_delta']['OFI'][s_key]
            d_deltas[s_key] = i_ofi_now - i_ofi_old
            f_cum_ofi = a.ofi_acc[s_key]['Ask']
            f_cum_ofi -= a.ofi_acc[s_key]['Bid']
            f_pos = a.ofi_acc[s_key]['qBid'] - a.ofi_acc[s_key]['qAsk']
            f_aux = f_cum_ofi + i_ofi_now * f_pos
            f_aux /= a.d_ofi_scale[s_key]
            f_pnl_ofi += f_aux
        f_old_ofi = s['OFI_Pnl']
        # first part of reward function: The monetized OFI
        reward = np.around(f_pnl_ofi - f_old_ofi, 2)
        # update the OFI_Pnl state
        s['OFI_Pnl'] = f_pnl_ofi
        # second part of the reward: The good placement
        # NOTE: as it is measuring how much OFI accumulated from the last
        # update give this reward just when the agent selects a new action
        if not a.b_new_reward:
            return reward
        s_compare = a.older_code
        for i_idx, s_cmm in zip(xrange(len(s_compare)), l_cmm):
            i_bet = a.d_bets[s_compare[i_idx]]
            f_aux = d_deltas[s_cmm] * i_bet
            f_aux /= a.d_ofi_scale[s_cmm]
            reward += f_aux * 5  # number of contracts by order
        return reward
