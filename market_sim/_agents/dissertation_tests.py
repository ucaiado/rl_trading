#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Different implementations of Rl agent to be tested in my dissertation

@author: ucaiado

Created on 11/21/2016
"""

from agent_rl import QLearningAgent

'''
Begin help functions
'''

'''
End help functions
'''


class QLearningAgent1(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    actions_to_open = [None]

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent1, self).__init__(env, i_id, d_normalizers,
                                              d_ofi_scale, f_min_time,
                                              f_gamma, f_alpha, i_numOfTilings,
                                              s_decay_fun, f_ttoupdate,
                                              d_initial_pos, s_hedging_on,
                                              b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent1'
        # Initialize any additional variables here
        self.features_names = ['position']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    fun(inputs['spread'][s_crt], 'spread_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent2(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent2, self).__init__(env, i_id, d_normalizers,
                                              d_ofi_scale, f_min_time,
                                              f_gamma, f_alpha, i_numOfTilings,
                                              s_decay_fun, f_ttoupdate,
                                              d_initial_pos, s_hedging_on,
                                              b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent2'
        # Initialize any additional variables here
        self.features_names = ['position', 'ofi_new']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    fun(inputs['spread'][s_crt], 'spread_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    # fun(f_hour, 'hour'),
                    fun(f_last_ofi, 'ofi_new', s_main)]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent3(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent3, self).__init__(env, i_id, d_normalizers,
                                              d_ofi_scale, f_min_time,
                                              f_gamma, f_alpha, i_numOfTilings,
                                              s_decay_fun, f_ttoupdate,
                                              d_initial_pos, s_hedging_on,
                                              b_hedging,
                                              b_keep_pos)
        self.s_agent_name = 'QLearningAgent3'
        # Initialize any additional variables here
        self.features_names = ['position', 'ofi_new', 'ratio_longo']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    # fun(f_hour, 'hour'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent4(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent4, self).__init__(env, i_id, d_normalizers,
                                              d_ofi_scale, f_min_time,
                                              f_gamma, f_alpha, i_numOfTilings,
                                              s_decay_fun, f_ttoupdate,
                                              d_initial_pos, s_hedging_on,
                                              b_hedging,
                                              b_keep_pos)
        self.s_agent_name = 'QLearningAgent4'
        # Initialize any additional variables here
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    # fun(f_hour, 'hour'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent5(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent5, self).__init__(env, i_id, d_normalizers,
                                              d_ofi_scale, f_min_time,
                                              f_gamma, f_alpha, i_numOfTilings,
                                              s_decay_fun, f_ttoupdate,
                                              d_initial_pos, s_hedging_on,
                                              b_hedging,
                                              b_keep_pos)
        self.s_agent_name = 'QLearningAgent5'
        # Initialize any additional variables here
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    # fun(f_hour, 'hour'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent6(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent6, self).__init__(env, i_id, d_normalizers,
                                              d_ofi_scale, f_min_time,
                                              f_gamma, f_alpha, i_numOfTilings,
                                              s_decay_fun, f_ttoupdate,
                                              d_initial_pos, s_hedging_on,
                                              b_hedging,
                                              b_keep_pos)
        self.s_agent_name = 'QLearningAgent6'
        # Initialize any additional variables here
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo', 'spread_curto']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    # fun(f_hour, 'hour'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['spread'][s_crt], 'spread_curto')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent7(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent7, self).__init__(env, i_id, d_normalizers,
                                              d_ofi_scale, f_min_time,
                                              f_gamma, f_alpha, i_numOfTilings,
                                              s_decay_fun, f_ttoupdate,
                                              d_initial_pos, s_hedging_on,
                                              b_hedging,
                                              b_keep_pos)
        self.s_agent_name = 'QLearningAgent7'
        # Initialize any additional variables here
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo', 'spread_curto',
                               'high_low']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    # fun(f_hour, 'hour'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['spread'][s_crt], 'spread_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent8(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent8, self).__init__(env, i_id, d_normalizers,
                                              d_ofi_scale, f_min_time,
                                              f_gamma, f_alpha, i_numOfTilings,
                                              s_decay_fun, f_ttoupdate,
                                              d_initial_pos, s_hedging_on,
                                              b_hedging,
                                              b_keep_pos)
        self.s_agent_name = 'QLearningAgent8'
        # Initialize any additional variables here
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo', 'spread_curto',
                               'high_low', 'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    # fun(f_hour, 'hour'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['spread'][s_crt], 'spread_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent9(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent9, self).__init__(env, i_id, d_normalizers,
                                              d_ofi_scale, f_min_time,
                                              f_gamma, f_alpha, i_numOfTilings,
                                              s_decay_fun, f_ttoupdate,
                                              d_initial_pos, s_hedging_on,
                                              b_hedging,
                                              b_keep_pos)
        self.s_agent_name = 'QLearningAgent9'
        # Initialize any additional variables here
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo', 'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent10(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent10, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent10'
        # Initialize any additional variables here
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent11(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent11, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent11'
        # Initialize any additional variables here
        self.features_names = ['position', 'ratio_longo',
                               'ratio_curto', 'spread_longo']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent12(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent12, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent12'
        # Initialize any additional variables here
        self.features_names = ['position', 'ratio_longo',
                               'ratio_curto', 'spread_longo', 'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgentA(QLearningAgent):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgentA, self).__init__(env, i_id, d_normalizers,
                                              d_ofi_scale, f_min_time,
                                              f_gamma, f_alpha,
                                              i_numOfTilings, s_decay_fun,
                                              f_ttoupdate, d_initial_pos,
                                              s_hedging_on, b_hedging,
                                              b_keep_pos)
        self.s_agent_name = 'QLearningAgentA'
        # Initialize any additional variables here
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto']
        self.f_gamma = f_gamma
        self.f_alpha = f_alpha

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        # experience
        f_hour = float(self.env.order_matching.s_time[11:13])
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent14(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent14, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.1,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent14'
        # Initialize any additional variables here
        self.decayfun = 'linear'
        self.f_gamma = f_gamma
        self.f_alpha = 0.1


class QLearningAgent15(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent15, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.3,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent15'
        # Initialize any additional variables here
        self.decayfun = 'linear'
        self.f_gamma = f_gamma
        self.f_alpha = 0.3


class QLearningAgent16(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent16, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent16'
        # Initialize any additional variables here
        self.decayfun = 'linear'
        self.f_gamma = f_gamma
        self.f_alpha = 0.5


class QLearningAgent17(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent17, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.7,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent17'
        # Initialize any additional variables here
        self.decayfun = 'linear'
        self.f_gamma = f_gamma
        self.f_alpha = 0.7


class QLearningAgent18(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent18, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.9,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent18'
        # Initialize any additional variables here
        self.decayfun = 'linear'
        self.f_gamma = f_gamma
        self.f_alpha = 0.9


class QLearningAgent19(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent19, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent19'
        # Initialize any additional variables here
        self.f_gamma = 0.1
        self.f_alpha = 0.7


class QLearningAgent20(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent20, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent20'
        # Initialize any additional variables here
        self.f_gamma = 0.3
        self.f_alpha = 0.7


class QLearningAgent21(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent21, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent21'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.7


class QLearningAgent22(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent22, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent22'
        # Initialize any additional variables here
        self.f_gamma = 0.7
        self.f_alpha = 0.7


class QLearningAgent23(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent23, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent23'
        # Initialize any additional variables here
        self.f_gamma = 0.9
        self.f_alpha = 0.7


class QLearningAgent24(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent24, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, 'linear',
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent24'
        # Initialize any additional variables here
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'


class QLearningAgent25(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent25, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, 'tpower',
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent25'
        # Initialize any additional variables here
        # 'tpower'
        # 'trig'
        self.decayfun = 'tpower'


class QLearningAgent26(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent26, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, 'trig',
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent26'
        # Initialize any additional variables here
        self.decayfun = 'trig'


class QLearningAgent27(QLearningAgentA):
    '''
    A representation of an agent that learns using Q-learning with linear
    parametrization and e-greedy exploration described at p.60 ~ p.61 form
    Busoniu at al., 2010. The approximator used is the implementation of tile
    coding, described at Sutton and Barto, 2016 (draft).
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent27, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               f_gamma, f_alpha,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent27'
        # Initialize any additional variables here
        self.f_gamma = 0.9
        self.f_alpha = 0.7
        # 'tpower'
        # 'trig'
        self.decayfun = 'tpower'


class QLearningAgent28(QLearningAgentA):
    '''
    The initial version of the learning agent. Used by the baseline test and by
    the last test
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent28, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent28'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.5
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo', 'spread_curto',
                               'size_bid_longo', 'size_bid_curto', 'high_low',
                               'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['spread'][s_crt], 'spread_curto'),
                    fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent29(QLearningAgentA):
    '''
    The initial version of the learning agent. Used by the baseline test. It is
    the same of the 28, besides the reward function, set in agent.py
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent29, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent29'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.5
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo', 'spread_curto',
                               'size_bid_longo', 'size_bid_curto', 'high_low',
                               'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['spread'][s_crt], 'spread_curto'),
                    fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent30(QLearningAgentA):
    '''
    The initial version of the learning agent. Used by the baseline test. It is
    the same of the 28, besides the reward function, set in agent.py
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent30, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent30'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.5
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo', 'spread_curto',
                               'size_bid_longo', 'size_bid_curto', 'high_low',
                               'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['spread'][s_crt], 'spread_curto'),
                    fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent31(QLearningAgentA):
    '''
    The initial version of the learning agent. Used by the baseline test. It is
    the same of the 28, besides the reward function, set in agent.py
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent31, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent31'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.5
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo', 'spread_curto',
                               'high_low', 'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['spread'][s_crt], 'spread_curto'),
                    # fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    # fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent32(QLearningAgentA):
    '''
    The initial version of the learning agent. Used by the baseline test. It is
    the same of the 28, besides the reward function, set in agent.py
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent32, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent32'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.5
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'
        self.features_names = ['position', 'ratio_longo',
                               'ratio_curto', 'spread_longo', 'spread_curto',
                               'high_low', 'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    # fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    fun(inputs['spread'][s_crt], 'spread_curto'),
                    # fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    # fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent33(QLearningAgentA):
    '''
    The initial version of the learning agent. Used by the baseline test. It is
    the same of the 28, besides the reward function, set in agent.py
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent33, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent33'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.5
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'spread_longo',
                               'high_low', 'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    # fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    # fun(inputs['spread'][s_crt], 'spread_curto'),
                    # fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    # fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent34(QLearningAgentA):
    '''
    The initial version of the learning agent. Used by the baseline test. It is
    the same of the 28, besides the reward function, set in agent.py
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent34, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent34'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.5
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo',
                               'high_low', 'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    # fun(inputs['spread'][s_crt], 'spread_curto'),
                    # fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    # fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent35(QLearningAgentA):
    '''
    The initial version of the learning agent. Used by the baseline test. It is
    the same of the 28, besides the reward function, set in agent.py
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent35, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent35'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.5
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto',
                               'high_low', 'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    # fun(inputs['spread'][s_lng], 'spread_longo'),
                    # fun(inputs['spread'][s_crt], 'spread_curto'),
                    # fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    # fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent36(QLearningAgentA):
    '''
    The initial version of the learning agent. Used by the baseline test. It is
    the same of the 28, besides the reward function, set in agent.py
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent36, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent36'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.5
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'
        self.features_names = ['position', 'ofi_new',
                               'ratio_curto', 'spread_longo',
                               'high_low', 'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    # fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    # fun(inputs['spread'][s_crt], 'spread_curto'),
                    # fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    # fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent37(QLearningAgentA):
    '''
    The initial version of the learning agent. Used by the baseline test. It is
    the same of the 28, besides the reward function, set in agent.py
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent37, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent37'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.5
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo',
                               'rel_price']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo'),
                    # fun(inputs['spread'][s_crt], 'spread_curto'),
                    # fun(inputs['size'][s_lng]['BID'], 'size_bid_longo'),
                    # fun(inputs['size'][s_crt]['BID'], 'size_bid_curto'),
                    # fun(inputs['HighLow'][s_lng], 'high_low'),
                    fun(inputs['reallAll'][s_lng], 'rel_price')]
        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn


class QLearningAgent38(QLearningAgentA):
    '''
    The initial version of the learning agent. Used by the baseline test. It is
    the same of the 28, besides the reward function, set in agent.py
    '''

    def __init__(self, env, i_id, d_normalizers, d_ofi_scale, f_min_time=3600.,
                 f_gamma=0.5, f_alpha=0.5, i_numOfTilings=16, s_decay_fun=None,
                 f_ttoupdate=5., d_initial_pos={}, s_hedging_on='DI1F19',
                 b_hedging=True, b_keep_pos=True):
        '''
        Initialize a QLearningAgent. Save all parameters as attributes

        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        :param d_normalizers: dictionary. The maximum range of each feature
        :param f_min_time*: float. Minimum time in seconds to the agent react
        :param f_gamma*: float. weight of delayed versus immediate rewards
        :param f_alpha*: the initial learning rate used
        :param i_numOfTilings*: unmber of tiling desired
        :param s_decay_fun*: string. The exploration factor decay function
        :param f_ttoupdate*. float. time in seconds to choose a diferent action
        '''
        super(QLearningAgent38, self).__init__(env, i_id, d_normalizers,
                                               d_ofi_scale, f_min_time,
                                               0.5, 0.5,
                                               i_numOfTilings, s_decay_fun,
                                               f_ttoupdate, d_initial_pos,
                                               s_hedging_on, b_hedging,
                                               b_keep_pos)
        self.s_agent_name = 'QLearningAgent38'
        # Initialize any additional variables here
        self.f_gamma = 0.5
        self.f_alpha = 0.5
        # 'tpower'
        # 'trig'
        self.decayfun = 'linear'
        self.features_names = ['position', 'ofi_new', 'ratio_longo',
                               'ratio_curto', 'spread_longo']

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
        d_data['BOOK_RATIO'] = 0.
        d_data['LOG_RET'] = inputs['logret']
        d_rtn = {}
        d_rtn['cluster'] = 0
        d_rtn['Position'] = float(state[s_main]['Position'])
        d_rtn['best_bid'] = state['best_bid']
        d_rtn['best_offer'] = state['best_offer']
        # calculate the current position in the main instrument
        f_pos = self.position[s_main]['qBid']
        f_pos -= self.position[s_main]['qAsk']
        f_pos += self.disclosed_position[s_main]['qBid']
        f_pos -= self.disclosed_position[s_main]['qAsk']
        # calculate the duration exposure
        # f_duration = self.risk_model.portfolio_duration(self.position)
        # measure the OFI index
        f_last_ofi = 0.
        f_last_ofi = inputs['dOFI'][s_main]
        # for the list to be used as features
        fun = self.bound_values
        s_lng = self.env.l_instrument[1]
        s_crt = self.env.l_instrument[0]
        l_values = [fun(f_pos * 1., 'position'),
                    fun(f_last_ofi, 'ofi_new', s_main),
                    fun(inputs['ratio'][s_lng]['BID'], 'ratio_longo'),
                    fun(inputs['ratio'][s_crt]['BID'], 'ratio_curto'),
                    fun(inputs['spread'][s_lng], 'spread_longo')]

        d_rtn['features'] = dict(zip(self.features_names, l_values))

        return d_rtn
