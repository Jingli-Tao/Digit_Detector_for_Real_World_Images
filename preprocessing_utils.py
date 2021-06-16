import cv2
import numpy as np
import torch
# generate regio of interest using mser: the regio with aspect ratio>3 and width <3 will be filter out
def mser_roi(img,max_aspect_ratio = 3, min_width = 5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height,width = gray.shape[0:2]
    mser = cv2.MSER_create(_delta=2)
    regions, _ = mser.detectRegions(gray)  
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  
    rois = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_width:
            continue
        
        if 1.0*w/h < max_aspect_ratio:
            rois.append([x, y, x + w, y + h])
            
    return rois
# for region of intest with aspect ratio less than 1, width will be set the same as height to avoid aspect ratio distortion
def expand_roi(rois,image_width,image_height):
    expanded_rois=[]
    for roi in rois:
        x,y,w,h = roi[0],roi[1],roi[2]-roi[0],roi[3]-roi[1]
        center_x,center_y = x+w//2, y+h//2
        if w < h:
            w = h
            x = max(0,center_x- w//2)
            expanded_rois.append([x, y, min(x + w,image_width), y + h])
        else:
            expanded_rois.append(roi)
    return expanded_rois
        
#calculate IOU between target roi and other rois
def calculate_IOU(target_roi,other_roi):
    
    top_left_x,top_left_y = np.maximum(target_roi[0],other_roi[:,0]),np.maximum(target_roi[1],other_roi[:,1])
    bottom_right_x,bottom_right_y=np.minimum(target_roi[2],other_roi[:,2]),np.minimum(target_roi[3],other_roi[:,3])
    inter_w,inter_h=np.maximum(bottom_right_x-top_left_x,0),np.maximum(bottom_right_y-top_left_y,0)
    target_area=(target_roi[2]-target_roi[0])*(target_roi[3]-target_roi[1])
    other_area=(other_roi[:,2]-other_roi[:,0])*(other_roi[:,3]-other_roi[:,1])
    inter_area=inter_w*inter_h
    return inter_area/(target_area+other_area-inter_area)

#non maximum suppression: if scores are missing, area of roi will be used as scores
def nms(rois,min_score,nms_threshold,scores=None):
    roi_array=np.array(rois)
    if scores is None:
        scores=(roi_array[:,2]-roi_array[:,0])*(roi_array[:,3]-roi_array[:,1])
    else:
        scores=np.array(scores)

    roi_array=roi_array[scores>min_score,:]
    scores=scores[scores>min_score]
    roi_sorted=roi_array[np.argsort (scores)[::-1]].tolist()
    scores_sorted=scores[np.argsort (scores)[::-1]].tolist()
    remaining_rois=roi_sorted.copy()
    remaining_scores=scores_sorted.copy()
    selected_rois=[]
    selected_scores=[]
    
    while remaining_rois:
        top=remaining_rois.pop(0)
        selected_rois.append(top)
        selected_scores.append(remaining_scores.pop(0))
        if remaining_rois:
            ious=calculate_IOU(top,np.array(remaining_rois))
            remaining_rois = np.array(remaining_rois)[ious<nms_threshold].tolist()
            remaining_scores = np.array(remaining_scores)[ious<nms_threshold].tolist()
        else:
            break
    return selected_rois,selected_scores

#generate roi at different image pyramid and apply nms (threshold >0.7) to reduce hightly overlapped regions   
def pyramid_roi_generate(img,ratios):
    roi_all_level = []
    for ratio in ratios:
        dest_size = int(img.shape[1]*ratio),int(img.shape[0]*ratio)
        resized_img = cv2.resize(img,dest_size,interpolation=cv2.INTER_CUBIC)
        rois = mser_roi(resized_img)
        roi_all_level.append(rois)
    rois_res =[]
    for level,rois in enumerate(roi_all_level):
        for roi in rois:
            roi = int(roi[0]/ratios[level]),int(roi[1]/ratios[level]),int(roi[2]/ratios[level]),int(roi[3]/ratios[level])
            rois_res.append(roi)
    if rois_res:
        rois_res,_ = nms(rois_res,min_score=60,nms_threshold=0.7)

    return rois_res
# get patch from image in roi and resize it to 32x32
def get_patch(I,roi):
    patch=cv2.resize(I[roi[1]:roi[3],roi[0]:roi[2],::-1],(32,32))
    return patch

#apply the transform as a preprocessing step of cnn classification
def get_transform(patch,transform):
    patch=(patch - np.min(patch)) / (np.max(patch) - np.min(patch))
    if transform:
        patch=transform(torch.from_numpy(patch.transpose([2,0,1]))).unsqueeze(0)
    else:
        patch= torch.from_numpy(patch.transpose([2,0,1])).unsqueeze(0)
    return patch


