# -*- coding: utf-8 -*-

import numpy as np
import cv2

from screen_features import get_screen_features
from pressed_keys import  get_key_pressed



def main():
    
    print("Fit pong screen into the window")
    print("Press 'up' or 'down' to start gathering data.")
    print("Press 'q' to stop and save.")
    
    saving_data = False

    train_data_buffer = list()
    
    #Call function to clear buffer of pressed keys
    get_key_pressed()
    
    #Keeping getting track of the object locations and keys pressed
    while True:

        screen, obj_locations = get_screen_features()
        
        key_pressed = get_key_pressed()
        
        #Show frame
        cv2.imshow("PilotoRobo - PythonJogaPong", screen)
    
        #Pass next frame every 10ms
        #Exit when 'q' is pressed
        if cv2.waitKey(10) == ord('q') or key_pressed == -1:
            cv2.destroyAllWindows()
            break
        
        
        #Check whether we are already saving data
        if saving_data:

            #Join locations with the key pressed
            train_data_point = np.append(obj_locations, key_pressed)

            #Add to the data points buffer
            train_data_buffer.append(train_data_point)
            
            print("Datapoint: {}".format(len(train_data_buffer)))
        
        elif key_pressed > 0:
            saving_data = True
        
        



    #Save data        
    
    #Convert datapoint list to a numpy array
    train_data_buffer = np.array(train_data_buffer)
    
    print(train_data_buffer.shape)
    print(train_data_buffer[:100])

    np.save("traindata_v1.npy", train_data_buffer)
    print('Data saved.')                    


if __name__ == "__main__":
    main()
