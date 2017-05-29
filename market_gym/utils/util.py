#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement different functions that can be used by different modules

@author: ucaiado

Created on 04/18/2017
"""

from collections import deque
import itertools

'''
Begin help functions
'''

'''
End help functions
'''


class ElapsedList(object):
    '''
    ElapsedList is a list that has a maximum lenght and is updated
    from time to time
    '''
    def __init__(self, f_elapsed_time=1., i_count=11):
        '''
        Initialize an ElapsedList object. Save all parameters as attributes

        :param f_elapsed_time: float. time resolution in seconds. It is how
            much information you lose at the begining of each time bucket
        :param i_count: the maximum size of the list kept in memory. i_count
            times f_elapsed_time is the time you are interested in
        '''
        self.i_count = i_count
        self.f_elapsed_time = max(1., f_elapsed_time)
        self.last_time = 0.
        self.l = deque([0], maxlen=i_count)
        self.i_used = 0

    def update(self, obj_value, f_current_time):
        '''
        Update the list according to the time passed. Return if it change the
        bucket time

        :param f_current_time: float. the cuurent time
        :param obj_value: Python object. Any python structure to be inserted
        '''
        self.i_used += 1
        f_elapsed_time = self.f_elapsed_time
        f_time = self.last_time + f_elapsed_time
        if f_current_time >= f_time or self.i_used == 1:
            f_aux = int(self.last_time/f_elapsed_time+1)
            f_aux *= f_elapsed_time
            f_aux = int((f_current_time - f_aux)/f_elapsed_time)
            i_include = int(min(f_aux, self.i_count))
            # i_include = f_aux
            if i_include > 0 and not self.i_used == 1:
                self.l.extend(itertools.repeat(self.l[-1], i_include))
            self.l.append(obj_value)
            f_aux = int(f_current_time/f_elapsed_time) * f_elapsed_time
            self.last_time = f_aux
            return True
        else:
            self.l[-1] = obj_value
            return False

    def get_value(self, i_id):
        '''
        Return and item from the list according to the id passed
        '''
        if i_id >= len(self.l):
            i_id = len(self.l)-1
        return self.l[i_id]

    @property
    def count(self):
        '''
        Return and item from the list according to the id passed
        '''
        return len(self.l)


class ElapsedList_old(object):
    '''
    ElapsedList is a list that has a maximum lenght and is updated
    from time to time
    '''
    def __init__(self, f_elapsed_time=1., i_count=11):
        '''
        Initialize an ElapsedList object. Save all parameters as attributes

        :param f_elapsed_time: float. time resolution in seconds. It is how
            much information you lose at the begining of each time bucket
        :param i_count: the maximum size of the list kept in memory. i_count
            times f_elapsed_time is the time you are interested in
        '''
        self.i_count = i_count
        self.f_elapsed_time = f_elapsed_time
        self.last_time = 0.
        self.l = deque([], maxlen=i_count)
        self.i_used = 0

    def update(self, obj_value, f_current_time):
        '''
        Update the list according to the time passed. Return if it change the
        bucket time

        :param f_current_time: float. the cuurent time
        :param obj_value: Python object. Any python structure to be inserted
        '''
        self.i_used += 1
        if f_current_time >= self.last_time or self.i_used == 1:
            self.l.append(obj_value)
            self.last_time = f_current_time + self.f_elapsed_time
            return True
        else:
            self.l[-1] = obj_value
            return False

    def get_value(self, i_id):
        '''
        Return and item from the list according to the id passed
        '''
        if i_id >= len(self.l):
            i_id = len(self.l)-1
        return self.l[i_id]

    @property
    def count(self):
        '''
        Return and item from the list according to the id passed
        '''
        return len(self.l)
