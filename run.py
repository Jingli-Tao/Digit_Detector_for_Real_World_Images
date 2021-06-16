import glob
from detector import TextDetector
import torchvision.transforms as transforms
import torch
import cv2
import json
import os
import numpy as np

# setting to make the prediction deteminstic
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
ratios = [0.5,1,2,4]
image_paths = glob.glob('./*.png')
PATH = './model'
output_folder = './graded_images'
if not os.path.exists(output_folder):
        os.mkdir(output_folder)
transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
detector = TextDetector (PATH,transform=transform)

def draw_label(img,text_to_write):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (int(img.shape[1]*0.7), int(img.shape[0]*0.9)) 
    fontScale = 0.5
    color = (0, 255, 0) 
    thickness = 1
    image = cv2.putText(img, text_to_write, org, font, fontScale, color, thickness, cv2.LINE_AA) 
    return image

for i,image_name in enumerate(image_paths):
    I = cv2.imread(image_name)

    scores,labels,roi_detections = detector.detection(I)
    
    if roi_detections is None:
        continue
    roi_array = np.array(roi_detections)
    # sorting the predictions from left to right
    centers = (roi_array[:,0]+roi_array[:,2])//2
    inds  = np.argsort(centers).tolist()
    sorted_label = [(labels[ind]) for ind in inds]
    sorted_score = [(scores[ind]) for ind in inds]
    # drawing rois
    for roi in roi_detections:
        cv2.rectangle(I,(int(roi[0]),int(roi[1])),(int(roi[2]),int(roi[3])),(0,0,0),1)
    strings = []
    for integer in sorted_label:
        if integer==10:
            strings.append(str(0))
        else:
            strings.append(str(integer))
    
    print(strings)
    I = draw_label(I,''.join(strings))
    cv2.imwrite(os.path.join(output_folder,image_name.split('_')[-1]),I)
