import numpy as np
from grabscreen import grab_screen
import cv2

import tensorflow as tf

#from pong_model import pong_model

import time

from get_pong_data import get_objects_locations

from directkeys import PressKey, ReleaseKey



def pong_dense_model():
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 8])
    
    y = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    
    h1 = tf.layers.dense(inputs, 20, activation=tf.nn.relu)
    h2 = tf.layers.dense(h1, 20, activation=tf.nn.relu)
    h3 = tf.layers.dense(h2, 10, activation=tf.nn.relu)
    
    logits = tf.layers.dense(h3, 3, activation=None)

    sc = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    cost = tf.reduce_mean(sc)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
    return inputs, y, logits, cost, accuracy


tf.reset_default_graph()

inputs, y, logits, cost, accuracy = pong_dense_model()



saver = tf.train.Saver()

with tf.Session() as sess:
    
    saver.restore(sess, "./model/model2.ckpt")
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    last_pos_h = 0
    last_pos_v = 0
    
    data_input_mean = np.array([ 257.28282598,  115.4207842 ,    1.81251011,    0.74357972,
         29.37375706,  124.76625135,  449.49676704,  113.94621712])
    
    data_input_std = np.array([  1.14266196e+02,   8.66662041e+01,   9.56141500e+01,
         7.88366735e+01,   1.57539844e+01,   3.75195882e+01,
         6.26986331e-02,   5.45067298e+01])
    
    while(True):
        
        #screen = grab_screen(region=(0,40,1920,1120))
        screen = grab_screen(region=(150,85,1920-150,1120-85))
        #last_time = time.time()
        # resize to something a bit more acceptable for a CNN
        screen = cv2.resize(screen, (480,270))
        # run a color convert:
        #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        
        screen_features = get_objects_locations(screen)
        
        #Calculate speed
        h_speed = screen_features[0] - last_pos_h
        v_speed = screen_features[1] - last_pos_v

        last_pos_h = screen_features[0]
        last_pos_v = screen_features[1]
        
        
        
        
        screen_features = (np.insert(screen_features, 2, [h_speed, v_speed]) - data_input_mean) / data_input_std

        
        #print(screen.shape)
        
        result = sess.run(logits, feed_dict={
            inputs: [screen_features]
        })
        
        #print(result)
        #continue
        
        ind = np.argmax(result[0])
        if ind == 0:
            print(result, "nothing")
        elif ind == 1:
            print(result, "up")
        elif ind == 2:
            print(result, "down")
        
#        break