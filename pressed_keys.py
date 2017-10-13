import win32api as wapi

def get_key_pressed():
    #Since for pong we can only go up and down, these are the only keys we will verify
    #Code for upkey is 38 and for down is 40

    up_pressed = wapi.GetAsyncKeyState(38)
    down_pressed = wapi.GetAsyncKeyState(40)
    exit_pressed = wapi.GetAsyncKeyState(ord("Q"))
    
    #if exit was pressed, return -1
    if exit_pressed:
        return -1
    
    #if the both keys are not pressed, or both pressed, return 0
    if down_pressed == up_pressed:
        return 0
    
    if up_pressed:
        return 1
    
    if down_pressed:
        return 2


 
if __name__ == "__main__":
    while True:
        print(get_key_pressed())