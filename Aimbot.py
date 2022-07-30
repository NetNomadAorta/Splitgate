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
import matplotlib.pyplot as plt
import win32api, win32con
from win32api import SetCursorPos as Cursor


# User parameters
SAVE_NAME_OD = "./Models/Splitgate-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split("-",1)[0] +"/"
MIN_SCORE    = 0.7


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    
    if pressed and button == "Button.x2":
        print(button)
        # Stop listener
        return False
    
    if not pressed and button == "Button.x2":
        # Stop listener
        return False


def click(x,y):
    win32api.SetCursorPos((x,y))
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)



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

transforms_1 = A.Compose([
    # A.Resize(int(frame.shape[0]/2), int(frame.shape[1]/2)),
    ToTensorV2()
])

# Start FPS timer
fps_start_time = time.time()
ii = 0
tenScale = 100

while True:
    # Collect events until released
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    
    screenshot = ImageGrab.grab()
    
    screenshot_x1 = 950
    screenshot_y1 = 200
    screenshot_x2 = 1970
    screenshot_y2 = 1190
    
    screenshot_cropped = screenshot.crop((screenshot_x1, screenshot_y1, screenshot_x2, screenshot_y2))
    screenshot_cropped_cv2 = np.array(screenshot_cropped)
    screenshot_cropped_cv2 = cv2.cvtColor(screenshot_cropped_cv2, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms_1(image=screenshot_cropped_cv2)
    transformed_image = transformed_image["image"]
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE].tolist()
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE].tolist()
    
    predicted_image = draw_bounding_boxes(transformed_image,
        boxes = dieCoordinates,
        # labels = [classes_1[i] for i in die_class_indexes], 
        labels = [str(round(i,2)) for i in die_scores], # SHOWS SCORE IN LABEL
        width = 3,
        colors = [color_list[i] for i in die_class_indexes]
        )
    
    frame = predicted_image.permute(1,2,0).contiguous().numpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # window_name = 'image'
    # cv2.imshow(window_name, frame)
    # cv2.waitKey(1) 
    
    # # Saves full image with bounding boxes
    # save_image((predicted_image/255), "test.jpg")
    
    plt.imshow(predicted_image.permute(1, 2, 0))
    
    
    ii += 1
    if ii % tenScale == 0:
        fps_end_time = time.time()
        fps_time_lapsed = fps_end_time - fps_start_time
        print("  ", round(tenScale/fps_time_lapsed, 2), "FPS")
        fps_start_time = time.time()

cv2.destroyAllWindows()