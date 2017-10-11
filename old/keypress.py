import win64con, ctypes, ctypes.wintypes

def esc_pressed():
    print("Escape was pressed.")

ctypes.windll.user32.RegisterHotKey(None, 1, 0, win32con.VK_ESCAPE)

try:
    msg = ctypes.wintypes.MSG()
    while ctypes.windll.user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
        if msg.message == win32con.WM_HOTKEY:
            esc_pressed()
        ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
        ctypes.windll.user32.DispatchMessageA(ctypes.byref(msg))
finally:
    ctypes.windll.user32.UnregisterHotKey(None, 1)