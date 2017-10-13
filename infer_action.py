# -*- coding: utf-8 -*-

import numpy as np
import cv2

from screen_features import get_screen_features
from pressed_keys import  get_key_pressed

from directkeys import PressKey, ReleaseKey

from neural_net import NeuralNetwork

def nayve_prediction(obj_locations):
                    #Perform action prediction based on data
    #            prediction = 0
    #            delta_v_pos = obj_locations[1] - obj_locations[5]
    #            if delta_v_pos < 0:
    #                prediction = 1
    #            elif delta_v_pos > 0:
    #                prediction = 2
    return 0

def get_offset_and_range_vectors(frame_data_buffer):
    """
    Function to get vectors to set origin of the game to 0,0 and range it from [0,1] in the horizontal and vertical axis.
    """
    
    #Get max and min feature values of the entire dataset
    min_values = np.amin(frame_data_buffer, axis=0)
    max_values = np.amax(frame_data_buffer, axis=0)
    
    left_offset = min_values[2] #Get the min value of the first bar horizontal position
    top_offset = min_values[1] #Get the min value of the ball vertical position
    
    #Subtract the second bar maximum h position from the first bar minimum h position
    range_h = max_values[4] - min_values[2]
    #Subtract the ball maximum v position from the ball minimum v position
    range_v = max_values[1] - min_values[1]
    
    offset_vector = np.array([left_offset, top_offset, left_offset, top_offset, left_offset, top_offset])
    range_vector = np.array([range_h, range_v, range_h, range_v, range_h, range_v])

    return offset_vector, range_vector

def main():
    
    print("Fit pong screen into the window")
    print("Press 'up' or 'down' to start infering actions.")
    print("Press 'q' for quit.")
    
    infering = False
    
    
    #Load neural network
    nn = NeuralNetwork()
    nn.load()
    
    #Save the last n frames to compute data features
    frame_data_buffer = None    
    
    #Call function to clear buffer of pressed keys
    get_key_pressed()
    
    last_pos_h = 0
    last_pos_v = 0
    
    #Keeping getting track of the object locations and keys pressed
    while True:

        screen, obj_locations = get_screen_features()
        
        key_pressed = get_key_pressed()
        
        cv2.imshow("PilotoRobo - PythonJogaPong", screen)
    
        #Pass next frame every 10ms
        #Exit when 'q' is pressed
        if cv2.waitKey(10) == ord('q') or key_pressed == -1:
            cv2.destroyAllWindows()
            break
            
        #Calculate speed
        h_speed = obj_locations[0] - last_pos_h
        v_speed = obj_locations[1] - last_pos_v

        last_pos_h = obj_locations[0]
        last_pos_v = obj_locations[1]

        screen_features = np.insert(obj_locations, 2, [h_speed, v_speed])
        
        #Check whether we are already saving data
        if infering:
            
            #prediction = 0
            
            prediction_probs = nn.predict([screen_features])[0] 
            prediction = np.argmax(prediction_probs)
            print(prediction)
            
            
            
            
#            #Ensure we have history of frames to work with
#            if len(frame_data_buffer) > 10:
#            
#                #Normalize screen data
##                offset_vector, range_vector = get_offset_and_range_vectors(frame_data_buffer)
##                datapoints = obj_locations - offset_vector
##                datapoints = datapoints / range_vector
##
#                
#                datapoints = obj_locations
#    
#                #Get ball speed
#                speed_datapoints = frame_data_buffer[-1, :2] - datapoints[:2]
#                #Insert new datapoints after the ball position features
#                datapoints = np.concatenate((datapoints[:2], speed_datapoints, datapoints[2:]))
#            
#                
#        
#            #Add frame data to frame_data_buffer
#            frame_data_buffer = np.append(frame_data_buffer, [obj_locations], axis=0)
        
            if prediction == 0:
                print("Nothing")
                ReleaseKey(0x48)
                ReleaseKey(0x50)
            elif prediction == 1:
                print("Up")
                ReleaseKey(0x50)
                PressKey(0x48)
            elif prediction == 2:
                print("Down")
                ReleaseKey(0x48)
                PressKey(0x50)
        
        elif key_pressed > 0:
            print("Infering")
            infering = True
            
            #Once start infering, init frame_data_buffer array
            #frame_data_buffer = np.array([obj_locations])
            
if __name__ == "__main__":
    main()
