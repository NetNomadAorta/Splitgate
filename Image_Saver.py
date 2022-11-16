import time
from PIL import ImageGrab
import winsound
import sys



# Windows beep settings
frequency = 700  # Set Frequency To 2500 Hertz
duration = 80  # Set Duration To 1000 ms == 1 second


time.sleep(3)

winsound.Beep(frequency, duration)

# Start FPS timer
fps_start_time = time.time()

for i in range(100):
    screenshot = ImageGrab.grab()
    
    screenshot.save('./Images/Screenshots/image-{}.jpg'.format(i))
    
    time.sleep(0.5)
    winsound.Beep(frequency, 40)
    
    ten_scale = 5
    
    i += 1
    if i % ten_scale == 0:
        fps_end_time = time.time()
        fps_time_lapsed = fps_end_time - fps_start_time
        fps = round(ten_scale/fps_time_lapsed, 2)
        
        sys.stdout.write('\033[2K\033[1G')
        print("  ", fps, "FPS -",
              end="\r"
              )
        fps_start_time = time.time()
    
winsound.Beep(frequency-100, duration)