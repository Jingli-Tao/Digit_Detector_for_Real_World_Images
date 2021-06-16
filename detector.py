import cv2
import glob
from preprocessing_utils import pyramid_roi_generate,expand_roi,get_patch,get_transform,nms
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import torchvision.transforms as transforms
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
# Text detector: detect digits in the natural images 

# model_path: the digit classification cnn model
# transform: preprocessing transform used during training
# ratios: the scales applied during pyramid mser roi generatio

class TextDetector:
    def __init__(self,model_path, transform, ratios=[0.5,1,2,4]):
        self.model = torch.load(model_path,map_location=torch.device('cpu'))
        self.transform = transform
        self.ratios = ratios
        self.split_component_nms_window_size = 7
        self.split_component_pad_window_size = 16
        self.threshold = 0.5
    def get_label_and_score_from_tensor(self,results):
        label = results.max(1)[1].numpy()
        score = (1-nn.Softmax(dim=-1)(results).detach().numpy()[:,0])
        return label,score
    def get_tensor_from_rois(self,rois,I):
        img_crops = []
        for roi in rois:
            patch = get_transform(get_patch(I,roi),self.transform)
            img_crops.append(patch)
        
        input_tensor = torch.cat(img_crops, 0).to('cpu')
        return input_tensor

    def split_error_connected_component(self,roi, img):
        #pad the original oi and reize the patch to 32xX
        pad = self.split_component_pad_window_size
        nms_window = self.split_component_nms_window_size
        threshold= self.threshold
        pad_roi = max(0,roi[0]-pad),roi[1], min(img.shape[1], roi[2]+pad),roi[3]
        aspect_ratio = (pad_roi[2]-pad_roi[0])/(pad_roi[3]-pad_roi[1])
        target_width_X = int(aspect_ratio*32)
        patch = img[pad_roi[1]:pad_roi[3],pad_roi[0]:pad_roi[2],::-1]
        resize_patch = cv2.resize(patch,(target_width_X,32),interpolation=cv2.INTER_CUBIC)
        
        scale = (pad_roi[3]-pad_roi[1])/32
        sliding_step = 2
        img_crops,rois_splited = [],[]
        #create patches from sliding window
        for start_x in range(0,target_width_X-32,sliding_step):
            patch_window = resize_patch[:,start_x:start_x+32,:]
            patch_window_transformed = get_transform(patch_window,self.transform)
            img_crops.append(patch_window_transformed)
            p1 = pad_roi[0]+start_x*scale,pad_roi[1]
            p2 = p1[0]+32*scale,p1[1]+32*scale
            center = (p1[0]+p2[0])//2,(p1[1]+p2[1])//2
            roi_width = int((roi[3]-roi[1])*0.5)# use fix apect ratio for splitted component
            p1 = center[0]-roi_width//2,p1[1]
            p2 = center[0]+roi_width//2,p2[1]
            rois_splited.append([p1[0],p1[1],p2[0],p2[1]])
            
        #run prediction
        input_tensor = torch.cat(img_crops, 0).to('cpu')
        results = self.model(input_tensor)
        
        label,score = self.get_label_and_score_from_tensor(results)
        #nms and threshold
        selected = np.where(np.bitwise_and(score>threshold,score==signal.order_filter(score, np.ones((nms_window,)), nms_window-1)))
        roi_selected = np.array(rois_splited)[selected[0].tolist(),:]
        return roi_selected, score[selected],label[selected]

    def detection(self,I):
        
        rois_raw = pyramid_roi_generate(I, self.ratios)
        rois = expand_roi(rois_raw,I.shape[1],I.shape[0])
        if len(rois)==0:
            return None,None,None
        input_tensor = self.get_tensor_from_rois(rois,I)
        results = self.model(input_tensor)
        self.rois_raw = rois_raw
        label_raw, score_raw = self.get_label_and_score_from_tensor(results)
        roi_detections_candidate, scores_candidates,labels_candidates = [],[],[]
        
        
        for i,roi in enumerate(rois):
            if label_raw[i]>0 and score_raw[i]>0:
                aspect_ratio = (rois_raw[i][2]-rois_raw[i][0])/(rois_raw[i][3]-rois_raw[i][1])
                if aspect_ratio > 0.7:
                    rois_split,score_split,label_split = self.split_error_connected_component(roi,I)
                    roi_detections_candidate.extend(rois_split.tolist())
                    scores_candidates.extend(score_split.tolist())
                    labels_candidates.extend(label_split.tolist())
                else:
                    roi_detections_candidate.extend([rois_raw[i]])
                    scores_candidates.extend([score_raw[i]])
                    labels_candidates.extend([label_raw[i]])
        labels=[]
        if roi_detections_candidate:
            roi_detections,scores = nms(roi_detections_candidate,min_score=0.5,nms_threshold=0.1,scores=scores_candidates)
            for roi in roi_detections:
                ind=roi_detections_candidate.index(roi)
                labels.append(labels_candidates[ind])
                cv2.rectangle(I,(int(roi[0]),int(roi[1])),(int(roi[2]),int(roi[3])),(0,0,0),1)
            # print(scores,labels)
            return scores,labels,roi_detections
        else:
            return None,None,None
