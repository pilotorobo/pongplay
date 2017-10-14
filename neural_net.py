import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import train_test_split



class NeuralNetwork():
    
    def __init__(self):
                
        tf.reset_default_graph()

        n_features = 8

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_features])    
        self.y = tf.placeholder(dtype=tf.int32, shape=[None])
        y_onehot = tf.one_hot(self.y, depth=3)
    
        h1 = tf.layers.dense(self.x, 20, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 20, activation=tf.nn.relu)
        #h3 = tf.layers.dense(h2, 10, activation=tf.nn.relu)
    
        self.logits = tf.layers.dense(h2, 3, activation=None)

        sc = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_onehot)

        self.cost = tf.reduce_mean(sc)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(y_onehot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #Optimizer
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cost)
        #self.optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(self.cost)
            
    def fit(self, X, Y):
        
        #Split train, test and validation set
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
        x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)        
        
        epochs = 25000

        self.sess = tf.Session()
        
        # Initializing the variables
        self.sess.run(tf.global_variables_initializer())

        
        for e in range(epochs):
        
            #Run optimizer and compute cost                
            cost_value, _ = self.sess.run([self.cost, self.optimizer], feed_dict={
                self.x: x_train,
                self.y: y_train
            })
            
            if e % 500 == 0:
                print("Epoch: {} Cost: {}".format(e, cost_value))

            #Run accuracy and compute its value        
            acc_value = self.sess.run(self.accuracy, feed_dict={
                self.x: x_valid,
                self.y: y_valid
            })

            if e % 500 == 0:
                print("Accuracy: {}".format(acc_value))
                print("")
                
                
        #Calculate final accuracy    
        final_acc = self.sess.run(self.accuracy, feed_dict={
            self.x: x_test,
            self.y: y_test
        })
    
        print("Final accuracy: {}".format(final_acc))
            

    def predict(self, X):
        
        prediction = self.sess.run(self.logits, feed_dict={
            self.x: X
        })
        
        return prediction
    
    def save(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "./model/model_v15.ckpt")
    
    def load(self):
        self.sess = tf.Session()
        
        saver = tf.train.Saver()
        saver.restore(self.sess, "./model/model_v15.ckpt")
        
        
        
def set_ball_speed(dataset):
    """
    Function to get ball speed based on the difference of horizontal and vertical positions in subsequent frames.
    """
    
    #Since we don't know the previous position of the first sample, we will have one less datapoint
    #Velocity(T) = Position(T) - Position(T-1)
    speed_datapoints = dataset[1:, :2] - dataset[0:-1, :2]
    #Insert new datapoints after the ball position features
    new_dataset = np.concatenate((dataset[1:, :2], speed_datapoints, dataset[1:, 2:]), axis=1)
    return new_dataset       
    
if __name__ == "__main__":
    
    #Load dataset
    dataset = np.load(file="traindata_v15.npy", encoding='bytes')

    #Inspect dataset
    labels_counter = Counter(dataset[:,6].tolist())
    print(labels_counter)
    print("Prediction must be higher than: {}".format(labels_counter[0.0]/dataset.shape[0]))
    
    dataset = set_ball_speed(dataset)
    
    neural_network = NeuralNetwork()
    
    X_data = dataset[:, :-1]
    Y_data = dataset[:, -1]

    neural_network.fit(X_data, Y_data)
    neural_network.save()
    
    
    
    
    
    
    
        