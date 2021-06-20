import numpy as np
import cv2
import json
from scipy.io import loadmat, savemat
from os.path import join
from sklearn.model_selection import train_test_split
from utils import *
from tqdm import tqdm

def calculate_gt_stats(dataset_json='digitStruct.json'):
    dataset = json.load(open(dataset_json))
    ws, hs = [], []
    for img_item in tqdm(dataset):
        filename, bbox = img_item['filename'], img_item['boxes']
        gts = [[bb_item['left'], bb_item['top'], bb_item['left'] + bb_item['width'], bb_item['top'] + bb_item['height']] for bb_item in bbox]
        for gt in gts:
            x, y, x2, y2 = int(gt[0]), int(gt[1]), int(gt[2]), int(gt[3])
            w, h = x2 - x, y2 - y
            ws.append(w)
            hs.append(h)
    ws, hs = np.array(ws), np.array(hs)
    w_h_ratio = ws / hs
    return [ws.min(), ws.max()], [hs.min(), hs.max()], np.percentile(w_h_ratio, [5,95])

def generate_neg_samples(dataset_json='digitStruct.json', dataset_folder ='./train', resize_ratios=[0.5, 1, 2, 4]):
    dataset = json.load(open(dataset_json))
    
    neg_samples_all, neg_samples_box_all = {}, {}

    print("Generating negative examples!")
    for img_item in tqdm(dataset):
        filename, bbox = img_item['filename'], img_item['boxes']

        gts = [[bb_item['left'], bb_item['top'], bb_item['left'] + bb_item['width'], bb_item['top'] + bb_item['height']] for bb_item in bbox]
        
        img = cv2.imread(join(dataset_folder, filename))
        w, h = img.shape[1],img.shape[0]
        gts = expand_roi(gts, w)
        pred = generate_roi_pyramid(img, resize_ratios)
        pred = expand_roi(pred, w)
        # neg_sample_no = len(gts)
        neg_samples_box, neg_samples = [], []
        while pred:
            selected_box = pred.pop()
            if np.max(calculate_IOU(selected_box, np.array(gts))) < 0.3:
                neg_samples_box.append(selected_box)
                neg_samples.append(
                    cv2.resize(img[selected_box[1]:selected_box[3], selected_box[0]:selected_box[2], ::-1], (32,32), cv2.INTER_LINEAR)
                    )
        if len(neg_samples_box):
            neg_samples_all[filename], neg_samples_box_all[filename] = neg_samples, neg_samples_box
        
    return neg_samples_box_all, neg_samples_all

def create_train_val_test_data(train_mat, train_json, train_folder, test_mat, test_json, test_folder, split=0.8):
    neg_samples_box, neg_samples_all = generate_neg_samples(dataset_json=train_json, dataset_folder=train_folder)
    data = loadmat(train_mat)
    X_neg = np.array(np.concatenate(list(neg_samples_all.values())))
    y_neg = np.zeros((X_neg.shape[0], 1)).astype('int')
    X_pos, y_pos = data['X'], data['y']
    X_pos = np.transpose(X_pos, (3,0,1,2))
    X, y = np.concatenate([X_pos, X_neg]), np.concatenate([y_pos, y_neg])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1 - split), random_state=42)
    savemat('train_data.mat', {'X': X_train, 'y': y_train})
    savemat('val_data.mat', {'X': X_val, 'y': y_val})

    neg_samples_box, neg_samples_all = generate_neg_samples(dataset_json=test_json, dataset_folder=test_folder)
    X_neg = np.array(np.concatenate(list(neg_samples_all.values())))
    y_neg = np.zeros((X_neg.shape[0], 1)).astype('int')
    data = loadmat(test_mat)      
    X_pos, y_pos = data['X'], data['y']
    X_pos = np.transpose(X_pos, (3,0,1,2))
    X, y = np.concatenate([X_pos, X_neg]), np.concatenate([y_pos, y_neg])
    savemat('test_data.mat', {'X': X, 'y': y})

if __name__ == "__main__":
    create_train_val_test_data('train_32x32.mat', 'digitStruct.json', './train', 'test_32x32.mat', 'digitStruct_test.json', './test')