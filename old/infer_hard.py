#Module that infer actions based on hand coded rules

from grabscreen import grab_screen

from get_pong_data import get_objects_locations

import time

import cv2

from directkeys import PressKey, ReleaseKey

import kp

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)
    
    while(True):
        
        #screen = grab_screen(region=(0,40,1920,1120))
        screen = grab_screen(region=(150,85,1920-150,1120-85))
    
        screen = cv2.resize(screen, (480,270))
        # run a color convert:
        #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    
        #ball_center, left_bar_center, right_bar_center = get_objects_locations(screen)
        obj_locations = get_objects_locations(screen)
        ball_center, left_bar_center, right_bar_center = obj_locations[0:2], obj_locations[2:4], obj_locations[4:6]
        
        
        delta_pos = ball_center[1] - right_bar_center[1]
        
        #print(delta_pos)
        #continue
        
        if delta_pos < -1:
            kp.go_up()
            print("UP", delta_pos)
        elif delta_pos > 1:
            print("DOWN", delta_pos)
            kp.go_down()
        else:
            print("NOTHING")
            ReleaseKey(0x48)
            ReleaseKey(0x50)
    
        #print(get_objects_locations(screen))
    
        #print(screen.shape)
        
        #