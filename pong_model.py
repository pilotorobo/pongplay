import tensorflow as tf

def conv2d(inputs, num_outputs, kernel_size, stride, padding):
    return tf.contrib.layers.conv2d(inputs=inputs, num_outputs=num_outputs, 
                                    kernel_size=kernel_size, stride=stride, padding=padding)

def dense(inputs, units, activation=None):
    return tf.layers.dense(inputs=inputs, units=units, activation=activation)

def pong_conv_model():
    
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 270, 480, 1])

    y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

    #Divide inputs by 255 for normalization
    x = conv2d(inputs/255, 10, 5, 2, 'SAME')
    x = conv2d(x, 20, 5, 2, 'SAME')
    x = conv2d(x, 40, 5, 2, 'SAME')
    x = conv2d(x, 80, 5, 2, 'SAME')

    x = tf.contrib.layers.flatten(x)

    x = dense(x,5000,activation=tf.nn.relu)
    x = dense(x,500,activation=tf.nn.relu)
    x = dense(x,100,activation=tf.nn.relu)
    x = dense(x,50,activation=tf.nn.relu)

    logits = dense(x,3,activation=None)


    sc = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    cost = tf.reduce_mean(sc)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
    return inputs, y, logits, cost, accuracy

def pong_dense_model():
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 6])
    
    y = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    
    h1 = dense(inputs, 20, activation=tf.nn.relu)
    h1 = dense(inputs, 3, activation=None)
    
    
    
    