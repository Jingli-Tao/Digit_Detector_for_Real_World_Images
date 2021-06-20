import cv2
import numpy as np
import torch

def mser_roi(img, max_aspect_ratio=3, min_width=5):
    """
    Generate ROIs using MSERs: the regions  
    with aspect ratio > 3 and width < 5
    will be eliminated.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # height, width = gray.shape[0:2]
    mser = cv2.MSER_create(_delta=2) # delta: MSERs margin
    regions, _ = mser.detectRegions(gray)  
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  
    rois = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_width:
            continue
        
        if (1.0 * w / h) < max_aspect_ratio:
            rois.append([x, y, x + w, y + h])     
    return rois

def expand_roi(rois, img_width):
    """
    Expand tall ROIs (aspect ratio < 1):  
    set width to be the same as height 
    to avoid aspect ratio distortion.
    """
    expanded_rois=[]
    for roi in rois:
        x, y, w, h = roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]
        center_x, center_y = x + w // 2, y + h // 2
        if w < h:
            w = h
            x = max(0, center_x - w // 2)
            expanded_rois.append([x, y, min(x + w, img_width), y + h])
        else:
            expanded_rois.append(roi)
    return expanded_rois
        
#calculate IOU between target roi and other rois
def calculate_IOU(target_roi, other_roi):
    """
    Calculate IOU between target roi 
    and other rois.
    """
    top_left_x = np.maximum(target_roi[0], other_roi[:,0])
    top_left_y = np.maximum(target_roi[1], other_roi[:,1])

    bottom_right_x = np.minimum(target_roi[2], other_roi[:,2])
    bottom_right_y = np.minimum(target_roi[3], other_roi[:,3])

    inter_w = np.maximum(bottom_right_x - top_left_x, 0)
    inter_h = np.maximum(bottom_right_y - top_left_y, 0)

    target_area = (target_roi[2] - target_roi[0]) * (target_roi[3] - target_roi[1])
    other_area = (other_roi[:,2] - other_roi[:,0]) * (other_roi[:,3] - other_roi[:,1])
    inter_area = inter_w * inter_h

    return inter_area / (target_area + other_area - inter_area)

def nms(rois, min_score, nms_threshold, scores=None):
    """
    Non maximum suppression: if scores 
    are missing, area of ROI is used 
    as scores.
    """
    roi_array = np.array(rois)
    if scores is None:
        scores = (roi_array[:,2] - roi_array[:,0]) * (roi_array[:,3] - roi_array[:,1])
    else:
        scores = np.array(scores)

    roi_array = roi_array[scores > min_score,:]
    scores = scores[scores > min_score]
    roi_sorted = roi_array[np.argsort(scores)[::-1]].tolist()
    scores_sorted = scores[np.argsort(scores)[::-1]].tolist()
    remaining_rois = roi_sorted.copy()
    remaining_scores = scores_sorted.copy()
    selected_rois = []
    selected_scores = []
    
    while remaining_rois:
        top = remaining_rois.pop(0)
        selected_rois.append(top)
        selected_scores.append(remaining_scores.pop(0))
        if remaining_rois:
            ious = calculate_IOU(top, np.array(remaining_rois))
            remaining_rois = np.array(remaining_rois)[ious < nms_threshold].tolist()
            remaining_scores = np.array(remaining_scores)[ious < nms_threshold].tolist()
        else:
            break
    return selected_rois, selected_scores
 
def generate_roi_pyramid(img, ratios):
    """
    Generate MSERs at different image
    scale and apply nms (threshold >0.7) 
    to reduce highly overlapped regions.
    """
    roi_all_level = []
    for ratio in ratios:
        dsize = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
        resized_img = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
        rois = mser_roi(resized_img)
        roi_all_level.append(rois)
    
    rois_res =[]
    for level, rois in enumerate(roi_all_level):
        for roi in rois:
            roi = (int(roi[0] / ratios[level]), int(roi[1] / ratios[level]), 
                    int(roi[2] / ratios[level]), int(roi[3] / ratios[level]))
            rois_res.append(roi)
    if rois_res:
        rois_res, _ = nms(rois_res, min_score=60, nms_threshold=0.7)

    return rois_res

# get patch from image in roi and resize it to 32x32
def get_patch(img, roi):
    """
    Get patch from image in roi and 
    resize it to 32x32
    """
    patch = cv2.resize(img[roi[1]:roi[3], roi[0]:roi[2], ::-1], (32,32))
    return patch

def get_transform(patch, transform):
    """
    Apply the transformation as a preprocessing
    step of CNN classification.
    """
    patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
    if transform:
        patch = transform(torch.from_numpy(patch.transpose([2,0,1]))).unsqueeze(0)
    else:
        patch = torch.from_numpy(patch.transpose([2,0,1])).unsqueeze(0)
    return patch

def draw_label(img, text_to_write):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (int(img.shape[1] * 0.7), int(img.shape[0] * 0.9)) 
    fontScale = 0.5
    color = (0, 255, 0) 
    thickness = 1
    image = cv2.putText(img, text_to_write, org, font, fontScale, color, thickness, cv2.LINE_AA) 
    return image