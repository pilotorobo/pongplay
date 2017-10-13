from grabscreen import grab_screen
import cv2
import numpy as np


def get_objects_locations(img):
    
    try:
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
    
    except:
        return np.array([0,0,0,0,0,0])


def get_screen():
    
    #screen = grab_screen(region=(200,200,480+200,270+200))
    screen = grab_screen(region=(200,200,720+200,405+200))
    #screen = grab_screen(region=(200,200,960+200,540+200))
    #print(screen.shape)
    #screen = cv2.resize(screen, (480,270))

    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

    return screen

def get_screen_features():
    
    screen = get_screen()
    
    obj_locations = get_objects_locations(screen)

    #Get center tuples 
    ball_center = tuple(np.round(obj_locations[:2]).astype(int))
    bar1_center = tuple(np.round(obj_locations[2:4]).astype(int))
    bar2_center = tuple(np.round(obj_locations[4:6]).astype(int))

    #Draw object locations on the image
    cv2.circle(screen, ball_center, 5, 128, -1)
    cv2.circle(screen, bar1_center, 5, 128, -1)
    cv2.circle(screen, bar2_center, 5, 128, -1)
    
    return screen, obj_locations


if __name__ == "__main__":
    print("Fit pong screen into the window:")

    while True:

        screen, obj_locations = get_screen_features()

        cv2.imshow("PilotoRobo - PythonJogaPong", screen)
    
        #Pass next frame every 25ms
        #Exit when 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            cv2.destroyAllWindows()
            break