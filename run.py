import glob
import torchvision.transforms as transforms
import torch
import cv2
import os
import numpy as np
from detector import TextDetector
from utils import draw_label

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True # setting to make the prediction deteminstic

ratios = [0.5,1,2,4]
img_paths = glob.glob('./*.png')
PATH = './model'
output_folder = './output_images'
if not os.path.exists(output_folder):
        os.mkdir(output_folder)
transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
detector = TextDetector (PATH, transform=transform)

for i, img_name in enumerate(img_paths):
    img = cv2.imread(img_name)

    scores, labels, roi_detections = detector.detection(img)
    
    if roi_detections is None:
        continue
    roi_array = np.array(roi_detections)

    # Sort the predictions
    centers = (roi_array[:, 0] + roi_array[:, 2]) // 2
    inds = np.argsort(centers).tolist()
    sorted_label = [(labels[ind]) for ind in inds]
    sorted_score = [(scores[ind]) for ind in inds]

    # Draw rois
    for roi in roi_detections:
        cv2.rectangle(img, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (0,0,0), 1)
    strings = []
    for integer in sorted_label:
        if integer == 10:
            strings.append(str(0))
        else:
            strings.append(str(integer))
    
    print(strings)
    img = draw_label(img, ''.join(strings))
    cv2.imwrite(os.path.join(output_folder, img_name.split('_')[-1]), img)