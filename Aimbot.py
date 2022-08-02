import cv2
import numpy as np
from PIL import ImageGrab
import os
import torch
from torchvision import models
import albumentations as A  # our data augmentation library
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2
from pynput import mouse
from torchvision.utils import save_image
import win32con
from win32api import mouse_event as Cursor_2
from math import sqrt
import winsound
import win32gui, win32ui, win32con


# User parameters
SAVE_NAME_OD = "./Models/Splitgate-1.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split("-",1)[0] +"/"
MIN_SCORE    = 0.6
IMAGE_REDUCER_SCALER = 4
MOUSE_TO_PIXEL_SCALER = 2/3
GAME_MOUSE_SCALER_X = 1 # Default: 2.40
GAME_MOUSE_SCALER_Y = 1 # Default: 2.55
squared_value = 0.65
cont_value_1 = 12
cont_value_2 = 1.1
# squared_value = 0.75
# cont_value_1 = 11
# cont_value_2 = 0


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def click_and_coordinates():
    points = []
    buttons = []
    
    def on_click(x, y, button, pressed):
        if (pressed or not pressed) and str(button) == "Button.x2":
            points.append([x, y])
            buttons.append(button)
            print(button)
            # Stop listener
            return False
        
        if not pressed:
            # Stop listener
            return False
    
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    
    if len(points) > 0:
        return points[0], buttons[0]
    else:
        return "null", "null"


def window_capture():
    w = 5120
    h = 1440
    
    # hwnd = win32gui.FindWindow(None, windowname)
    hwnd = None
    
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)
    
    # Save the screenshot
    # dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (h, w, 4)
    
    # Free Resource
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    
    # Drop the alpha channel, or cv.matchTemplate()
    img = img[...,:3]
    
    img = np.ascontiguousarray(img)
    
    return img
    
    



dataset_path = DATASET_PATH

#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes_1 = len(categories.keys())
categories

classes_1 = [i[1]['name'] for i in categories.items()]
classes_1


# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)


# TESTING TO LOAD MODEL
if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD)
    model_1.load_state_dict(checkpoint)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU to train
model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

color_list = ['green', 'magenta', 'turquoise', 'red', 'green', 'orange', 'yellow', 'cyan', 'lime']

# Windows beep settings
frequency = 700  # Set Frequency To 2500 Hertz
duration = 80  # Set Duration To 1000 ms == 1 second

screenshot = ImageGrab.grab()
screenshot_cv2 = np.array(screenshot)

center_screen_x = (screenshot_cv2.shape[1]/2)/IMAGE_REDUCER_SCALER
center_screen_y = (screenshot_cv2.shape[0]/2)/IMAGE_REDUCER_SCALER

transforms_1 = A.Compose([
    A.Resize(int(screenshot_cv2.shape[0]/IMAGE_REDUCER_SCALER), 
             int(screenshot_cv2.shape[1]/IMAGE_REDUCER_SCALER)),
    ToTensorV2()
])

# Start FPS timer
fps_start_time = time.time()
ii = 0
tenScale = 50

while True:
    # Collects events from mouse
    positions, button = click_and_coordinates()
    
    screenshot = ImageGrab.grab()
    # screenshot = window_capture()
    
    screenshot_cv2 = np.array(screenshot)
    screenshot_cv2 = cv2.cvtColor(screenshot_cv2, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms_1(image=screenshot_cv2)
    transformed_image = transformed_image["image"]
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE]
    
    enemy_coordinates_list = dieCoordinates[die_class_indexes > 0].tolist() # SHOULD "== 1].tolist()" FOR ENEMY
    
    die_class_indexes = die_class_indexes.tolist()
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = die_scores.tolist()
    
    
    predicted_image = draw_bounding_boxes(transformed_image,
        boxes = dieCoordinates,
        # labels = [classes_1[i] for i in die_class_indexes], 
        labels = [str(round(i,2)) for i in die_scores], # SHOWS SCORE IN LABEL
        width = 3,
        colors = [color_list[i] for i in die_class_indexes]
        )
    
    # if len(enemy_coordinates_list) > 0:
    if len(enemy_coordinates_list) > 0 and str(button) == "Button.x2":
        center_to_enemy_x_len_list = []
        center_to_enemy_y_len_list = []
        for enemy_coordinates in enemy_coordinates_list:
            center_enemy_x = int(enemy_coordinates[0]
                                +(enemy_coordinates[2]-enemy_coordinates[0])/2
                                )
            center_enemy_y = int(enemy_coordinates[1]
                                +(enemy_coordinates[3]-enemy_coordinates[1])/4
                                )
            center_to_enemy_x_len_list.append(center_enemy_x - center_screen_x)
            center_to_enemy_y_len_list.append(center_enemy_y - center_screen_y)
        
        hypotenuse_list = []
        most_centered_hypotenuse = 100000
        for index, enemy_coordinates in enumerate(enemy_coordinates_list):
            hypotenuse = sqrt(center_to_enemy_y_len_list[index]**2 + center_to_enemy_x_len_list[index]**2)
            if hypotenuse < most_centered_hypotenuse:
                most_centered_hypotenuse = hypotenuse
                most_centered_to_enemy_x = center_to_enemy_x_len_list[index]
                most_centered_to_enemy_y = center_to_enemy_y_len_list[index]
        
        x_move = most_centered_to_enemy_x * IMAGE_REDUCER_SCALER * MOUSE_TO_PIXEL_SCALER * GAME_MOUSE_SCALER_X
        y_move = most_centered_to_enemy_y * IMAGE_REDUCER_SCALER * MOUSE_TO_PIXEL_SCALER * GAME_MOUSE_SCALER_Y
        
        if x_move < 0:
            x_move = -cont_value_1*abs(x_move)**squared_value+cont_value_2*x_move
        else:
            x_move = cont_value_1*x_move**squared_value+cont_value_2*x_move
        if y_move < 0:
            y_move = -cont_value_1*abs(y_move)**squared_value+cont_value_2*y_move
        else:
            y_move = cont_value_1*y_move**squared_value+cont_value_2*y_move
        
        winsound.Beep(frequency, duration)
        
        # Moves cursor
        for i in range(5):
            Cursor_2(win32con.MOUSEEVENTF_MOVE, 
                      int(x_move/5), 
                      int(y_move/5), 
                      0, 0) 
            time.sleep(0.0001)
        
        # Saves full image with bounding boxes
        save_image((predicted_image/255), "test.jpg")
    
    
    
    ii += 1
    if ii % tenScale == 0:
        fps_end_time = time.time()
        fps_time_lapsed = fps_end_time - fps_start_time
        print("  ", round(tenScale/fps_time_lapsed, 2), "FPS")
        fps_start_time = time.time()

cv2.destroyAllWindows()