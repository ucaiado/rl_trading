#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement a simulator to mimic a dynamic order book environment

@author: ucaiado

Created on 10/24/2016
"""

from market_gym.envs.interest_rate import YieldCrvEnv
from market_gym.envs.stockn_option import StcknOptEnv


'''
Begin help functions
'''

IMPLEMENTED_ENVS = {'YieldCurve': YieldCrvEnv,
                    'StockAndOption': StcknOptEnv}


class NotFoundEnvException(Exception):
    """
    NotFoundEnvException is raised by the make() method when the env name is
    not presented in clobal dict IMPLEMENTED_ENVS
    """
    pass

'''
Begin help functions
'''


def make(s_env, **kwargs):
    '''
    Return the environment object identified

    :param s_env: string. The key of env in IMPLEMENTED_ENVS
    '''
    try:
        cls = IMPLEMENTED_ENVS[s_env]
        return cls
    except KeyError:
        s_err = 'Environment specified was not implemented yet'
        raise NotFoundEnvException(s_err)
