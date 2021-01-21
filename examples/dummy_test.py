#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:23:29 2021

@author: andrew
"""

import shelve
import numpy as np
import warnings
warnings.filterwarnings("error")
y=1
z = "tester"
try:
    x = np.log(0/1)
except RuntimeWarning:
    filename='shelve.out'
    my_shelf = shelve.open(filename,'n') # 'n' for new

    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()

print(x)


my_shelf = shelve.open(filename)
for key in my_shelf:
    #globals()[key]=my_shelf[key]
    print(key)
    print(my_shelf[key])
my_shelf.close()