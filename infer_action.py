import numpy as np
from grabscreen import grab_screen
import cv2

import tensorflow as tf

from pong_model import pong_model

import time



tf.reset_default_graph()

inputs, y, logits, cost, accuracy = pong_model()


saver = tf.train.Saver()

with tf.Session() as sess:
    
    saver.restore(sess, "./model/model.ckpt")
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    while(True):
        
        #screen = grab_screen(region=(0,40,1920,1120))
        screen = grab_screen(region=(150,85,1920-150,1120-85))
        #last_time = time.time()
        # resize to something a bit more acceptable for a CNN
        screen = cv2.resize(screen, (480,270))
        # run a color convert:
        #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        
        #print(screen.shape)
        
        result = sess.run(logits, feed_dict={
            inputs: screen.reshape(1, screen.shape[0], screen.shape[1], 1)
        })
        
        ind = np.argmax(result[0])
        if ind == 0:
            print("nothing")
        elif ind == 1:
            print("up")
        elif ind == 2:
            print("down")
        
#        break
    
    