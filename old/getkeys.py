# Citation: Box Of Hats (https://github.com/Box-Of-Hats )
# -*- coding: utf-8 -*-

import win32api as wapi
import time

keyList = [38, 40, ord("T")]

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(key):
            keys.append(key)
    return keys
 
