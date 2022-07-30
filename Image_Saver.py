import time
from PIL import ImageGrab
import winsound



# Windows beep settings
frequency = 700  # Set Frequency To 2500 Hertz
duration = 80  # Set Duration To 1000 ms == 1 second


time.sleep(3)

winsound.Beep(frequency, duration)

for i in range(10):
    screenshot = ImageGrab.grab()
    
    screenshot.save('./Images/Screenshots/image-{}.jpg'.format(i))
    
    time.sleep(0.5)
    winsound.Beep(frequency, 40)
    
winsound.Beep(frequency-100, duration)