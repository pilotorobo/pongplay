import cv2
import numpy as np

def get_objects_locations(img):
    #Simplify image using threshold (maybe not necessary)
    #_, cent_img = cv2.threshold(cent_img,10,255,cv2.THRESH_BINARY)
    
    #Get connected compontents in the image
    n_elem, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    
    #Calculate features to determine which elements are ball and bars 
    calc_features = list()    
    for i, (x0,y0,width,height,area) in enumerate(stats):
        #Calc ball feature
        ball_feature = abs(width/height - 1 + area - width*height)#abs(width + height - 8 - 8 + area - 64)
        #Bar feature
        bar_feature = abs(height/width - 4.6 + area - width*height)#abs(width + height - 10 - 46 + area - 460)
        
        calc_features.append((i, ball_feature, bar_feature))
    
       
    #Sort values to get the most probable indexes of ball and bars
    ball_ind = sorted(calc_features, key=lambda a: a[1])[0][0]
    bars_ind = [bar_data[0] for bar_data in sorted(calc_features, key=lambda a: a[2])[0:2]]
    
    
    #Get the centroids with the indexes
    ball_center = centroids[ball_ind]
    bars_center = centroids[bars_ind]
    
    #Get left bar and right bar based on sorted value of the x position
    sorted_bars = sorted(bars_center, key=lambda a: a[0])
    
    left_bar_cent, right_bar_cent = sorted_bars[0], sorted_bars[1]
    
    return np.array([ball_center, left_bar_cent, right_bar_cent]).reshape(-1)