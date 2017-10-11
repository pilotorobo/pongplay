# -*- coding: utf-8 -*-

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

from get_pong_data import get_objects_locations


#non_key = [1, 0, 0]
#up_key = [0, 1, 0]
#down_key = [0, 0, 1]
#
#starting_value = 1

#while True:
#    file_name = 'train_data/training_data-{}.npy'.format(starting_value)
#
#    if os.path.isfile(file_name):
#        print('File exists, moving along',starting_value)
#        starting_value += 1
#    else:
#        print('File does not exist, starting fresh!',starting_value)
#        
#        break


def keys_to_output(keys):
    
    if len(keys) == 0 or len(keys) >= 2:
        return 0
    
    if 38 in keys:
        return 1
    
    if 40 in keys:
        return 2
    
    raise Exception("This should not be here.")



def main():
    
    train_data_buffer = list()
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    #last_time = time.time()
    paused = False
    print('STARTING!!!')
    #while(True):
    
    for i in range(20000):
        
        print(i)
        
        screen = grab_screen(region=(10,140,820,650))
        screen = cv2.resize(screen, (480,270))
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        keys = key_check()
        key_value = keys_to_output(keys)

        obj_locations = get_objects_locations(screen)

        train_data_point = np.append(obj_locations, key_value)

        train_data_buffer.append(train_data_point)
        
        
    print(np.array(train_data_buffer).shape)
    
    np.save("traindata_v2.npy", np.array(train_data_buffer))
    print('SAVED')                    


if __name__ == "__main__":
    main()
