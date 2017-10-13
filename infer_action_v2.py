import cv2
import numpy as np

from directkeys import PressKey, ReleaseKey

from screen_features import get_screen_features

from neural_net import NeuralNetwork


#import tensorflow as tf
#
#
#tf.reset_default_graph()
#
#n_features = 8
#
#x_mean = tf.Variable(initial_value=[0]*n_features, trainable=False, dtype=tf.float32)
#x_std = tf.Variable(initial_value=[1]*n_features, trainable=False, dtype=tf.float32)
#
#x = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
#
#keep_prob = tf.placeholder(dtype=tf.float32)
#
#x_norm = (x-x_mean)/x_std
#    
#y = tf.placeholder(dtype=tf.int32, shape=[None])
#y_onehot = tf.one_hot(y, depth=3)
#    
#h1 = tf.layers.dense(x, 20, activation=tf.nn.relu)
#h2 = tf.layers.dense(h1, 20, activation=tf.nn.relu)
#h3 = tf.layers.dense(h2, 10, activation=tf.nn.relu)
#    
#logits = tf.layers.dense(h2, 3, activation=None)
#
#saver = tf.train.Saver()

#sess = tf.Session()
#    
#saver.restore(sess, "./model/model_v21_nodrop.ckpt")

#Init neural net
neural_network = NeuralNetwork()
neural_network.load()

last_pos_h = 0
last_pos_v = 0

while True:

    screen, obj_locations = get_screen_features()

    cv2.imshow("PilotoRobo - PythonJogaPong", screen)

    if cv2.waitKey(25) & 0xFF ==  ord('q'):
        cv2.destroyAllWindows()
        break

    #Calculate speed
    h_speed = obj_locations[0] - last_pos_h
    v_speed = obj_locations[1] - last_pos_v

    last_pos_h = obj_locations[0]
    last_pos_v = obj_locations[1]

    screen_features = np.insert(obj_locations, 2, [h_speed, v_speed])

#    result = sess.run(logits, feed_dict={
#        x: [screen_features]
#    })

    result = neural_network.predict([screen_features])

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
            

