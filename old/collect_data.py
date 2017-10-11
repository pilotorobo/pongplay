# -*- coding: utf-8 -*-

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os


non_key = [1, 0, 0]
up_key = [0, 1, 0]
down_key = [0, 0, 1]

starting_value = 1

while True:
    file_name = 'train_data/training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)
        
        break


def keys_to_output(keys):
    
    if len(keys) == 0 or len(keys) >= 2:
        return non_key
    
    if 38 in keys:
        return up_key
    
    if 40 in keys:
        return down_key
    
    raise Exception("This should not be here.")



def main(file_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    #last_time = time.time()
    paused = False
    print('STARTING!!!')
    while(True):
        
        if not paused:
            #screen = grab_screen(region=(0,40,1920,1120))
            screen = grab_screen(region=(150,85,1920-150,1120-85))
            #last_time = time.time()
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (480,270))
            # run a color convert:
            #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
            #print(screen.shape)
            
            keys = key_check()
            output = keys_to_output(keys)
            #print(output)
            training_data.append([screen,output])

            #print('loop took {} seconds'.format(time.time()-last_time))
            #last_time = time.time()
##            cv2.imshow('window',cv2.resize(screen,(640,360)))
##            if cv2.waitKey(25) & 0xFF == ord('q'):
##                cv2.destroyAllWindows()
##                break

#            cv2.imshow("lu", screen)
#            cv2.waitKey()
            

            if len(training_data) % 100 == 0:
                print(len(training_data))
#                break
                
                if len(training_data) == 1000:
                    np.save(file_name,training_data)
                    print('SAVED')
                    training_data = []
                    starting_value += 1
                    file_name = 'train_data/training_data-{}.npy'.format(starting_value)
#                    break

                    
        keys = key_check()
        if ord("T") in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main(file_name, starting_value)
