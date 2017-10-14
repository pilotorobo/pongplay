import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from collections import Counter

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

class DecisionTree():
    
    def __init__(self):
        self.dt = DecisionTreeClassifier()
        
    def load(self):
        
        dataset = np.load(file="traindata_v15.npy", encoding='bytes')

        #Inspect dataset
        labels_counter = Counter(dataset[:,6].tolist())
        print(labels_counter)
        print("Prediction must be higher than: {}".format(labels_counter[0.0]/dataset.shape[0]))

        dataset = set_ball_speed(dataset)

        X_data = dataset[:, :-1]
        Y_data = dataset[:, -1]
        
        x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
        
        self.dt.fit(x_train, y_train)
        
    def predict(self, X):
        return self.dt.predict(X)