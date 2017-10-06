import pyscreenshot as ImageGrab
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # fullscreen
#    for i in range(100):
    
    
    im=ImageGrab.grab(bbox=(10,20,510,510))
#    im.show()
    plt.imshow(im)
    #plt.show()
    