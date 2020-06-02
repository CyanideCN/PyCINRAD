# -*- coding: utf-8 -*-
# Author: Puyuan Du

import warnings


class Deprecated(object):
    def __init__(self, obj, warn_msg):
        self.obj = obj
        self.msg = warn_msg

    def __getattr__(self, attr):
        warnings.warn(self.msg)
        return getattr(self.obj, attr)

    def __call__(self, *args, **kwargs):
        warnings.warn(self.msg)
        return self.obj(*args, **kwargs)
