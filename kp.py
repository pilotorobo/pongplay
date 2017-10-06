# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

#import win32api as wapi
#import time
#
#keyList = ["\b"]
#for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789":
#    keyList.append(char)
#
#def key_check():
#    keys = []
#    for key in keyList:
#        if wapi.GetAsyncKeyState(ord(key)):
#            keys.append(key)
#    return keys
#
#key_check()

from directkeys import PressKey,ReleaseKey, W, A, S, D

UP = 0x48
DOWN = 0X50



def go_up():
    ReleaseKey(DOWN)
    PressKey(UP)
    
    
def go_down():
    ReleaseKey(UP)
    PressKey(DOWN)
    
    
    
if __name__ == "__main__":
    import time

    for i in list(range(4))[::-1]:
            print(i+1)
            time.sleep(1)

    #go_up()    
    #while True:
        #pass
    ReleaseKey(DOWN)
    ReleaseKey(UP)

#import time

#while True:
    #ReleaseKey(UP)

#time.sleep(5)