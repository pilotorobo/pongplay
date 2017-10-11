import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

train_data = np.load("train_data/training_data-1.npy")


for i in range(50):

    #data_index = random.randint(0,490)
    data_index = i

    img = train_data[data_index][0]
    label = train_data[data_index][1]
    
    print(label)

    cv2.imshow("lucas", img)
    cv2.waitKey()

