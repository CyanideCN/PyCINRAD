# -*- coding: utf-8 -*-
#Author: Du puyuan

def check_radartype(accept_list):
    r'''Check if the decorated function is supported for this type of radar.'''
    def check(func):
        def inner(self, *args, **kwargs):
            if self.radartype not in accept_list:
                raise RadarError('{} radar is not supported for fuction {}'.format(self.radartype, func.__name__))
            return func(self, *args, **kwargs)
        return inner
    return check