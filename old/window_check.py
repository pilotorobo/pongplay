from grabscreen import grab_screen
import cv2
from get_pong_data import get_objects_locations
import numpy as np


while True:

    #screen = grab_screen(region=(200,200,480+200,270+200))
    screen = grab_screen(region=(200,200,720+200,405+200))
    #screen = grab_screen(region=(200,200,960+200,540+200))
    #print(screen.shape)
    #screen = cv2.resize(screen, (480,270))

    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

    obj_locations = get_objects_locations(screen)
    
    #print(obj_locations[2] - obj_locations[4])

    ball_center = tuple(np.round(obj_locations[:2]).astype(int))
    bar1_center = tuple(np.round(obj_locations[2:4]).astype(int))
    bar2_center = tuple(np.round(obj_locations[4:6]).astype(int))

    #print(ball_center)

    cv2.circle(screen, ball_center, 5, 128, -1)
    cv2.circle(screen, bar1_center, 5, 128, -1)
    cv2.circle(screen, bar2_center, 5, 128, -1)

    #print(data)

    cv2.imshow("window", screen)
    
    if cv2.waitKey(25) & 0xFF ==  ord('q'):
        cv2.destroyAllWindows()
        break