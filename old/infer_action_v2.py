from grabscreen import grab_screen
import cv2
from get_pong_data import get_objects_locations
import numpy as np
from directkeys import PressKey, ReleaseKey


import tensorflow as tf


tf.reset_default_graph()

n_features = 8

x_mean = tf.Variable(initial_value=[0]*n_features, trainable=False, dtype=tf.float32)
x_std = tf.Variable(initial_value=[1]*n_features, trainable=False, dtype=tf.float32)

x = tf.placeholder(dtype=tf.float32, shape=[None, n_features])

x_norm = (x-x_mean)/x_std
    
y = tf.placeholder(dtype=tf.int32, shape=[None])
y_onehot = tf.one_hot(y, depth=3)
    
h1 = tf.layers.dense(x, 20, activation=tf.nn.relu)
h2 = tf.layers.dense(h1, 20, activation=tf.nn.relu)
#h3 = tf.layers.dense(h2, 10, activation=tf.nn.relu)
    
logits = tf.layers.dense(h2, 3, activation=None)

saver = tf.train.Saver()

with tf.Session() as sess:
    
    saver.restore(sess, "./model/model_v21.ckpt")
    
    print(x_mean.eval())
            
    last_pos_h = 0
    last_pos_v = 0

    last_action = 0
    
    
    #Values to compute position invariate features
    max_ball_v_pos = 0
    min_ball_v_pos = 999999999
    
    max_bar_h_pos = 0
    min_bar_h_pos = 99999999
    
    while True:

        #screen = grab_screen(region=(10,140,820,650))
        #screen = cv2.resize(screen, (480,270))
        #screen = grab_screen(region=(200,200,480+200,270+200))
        screen = grab_screen(region=(200,200,720+200,405+200))
        #screen = grab_screen(region=(200,200,960+200,540+200))

        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        obj_locations = get_objects_locations(screen)
        
        #Draw object locations
        ball_center = tuple(np.round(obj_locations[:2]).astype(int))
        bar1_center = tuple(np.round(obj_locations[2:4]).astype(int))
        bar2_center = tuple(np.round(obj_locations[4:6]).astype(int))

        cv2.circle(screen, ball_center, 5, 128, -1)
        cv2.circle(screen, bar1_center, 5, 128, -1)
        cv2.circle(screen, bar2_center, 5, 128, -1)
        
        #Update pos invariant features
        #[ballh, ballv, bar1h, bar1v, bar2h, bar2v, label]
        max_ball_v_pos = max(max_ball_v_pos, obj_locations[1] + 000.1)
        min_ball_v_pos = min(min_ball_v_pos, obj_locations[1]) 
        max_bar_h_pos = max(max_bar_h_pos, obj_locations[4])
        min_bar_h_pos = min(min_bar_h_pos, obj_locations[2])
        
        gwindow_height = max_ball_v_pos - min_ball_v_pos
        gwindow_width = max_bar_h_pos - min_bar_h_pos
        
        print(gwindow_height, gwindow_width)
        
        pos_inv_vec = np.array([gwindow_width, gwindow_height, gwindow_width, gwindow_height, gwindow_width, gwindow_height])
    
        #Transform features
        obj_locations = obj_locations / pos_inv_vec
        
        #Calculate speed
        h_speed = obj_locations[0] - last_pos_h
        v_speed = obj_locations[1] - last_pos_v

        last_pos_h = obj_locations[0]
        last_pos_v = obj_locations[1]
        
        screen_features = np.insert(obj_locations, 2, [h_speed, v_speed])
        
        result = sess.run(logits, feed_dict={
            x: [screen_features]
        })
        
        ind = np.argmax(result[0])
        if ind == 0:
            print(result, "nothing")
            ReleaseKey(0x48)
            ReleaseKey(0x50)
        elif ind == 1:
            print(result, "up")
            ReleaseKey(0x50)
            PressKey(0x48)
        elif ind == 2:
            print(result, "down")
            ReleaseKey(0x48)
            PressKey(0x50)
            
        
            
            


        

        cv2.imshow("window", screen)

        if cv2.waitKey(25) & 0xFF ==  ord('q'):
            cv2.destroyAllWindows()
            break