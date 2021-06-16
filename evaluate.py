import torch
from data_pipeline import SVHNDataset
from model import step_single
from sklearn.metrics import confusion_matrix
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from preprocessing_utils import calculate_IOU
import glob
import json
from detector import TextDetector
import cv2
transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# all the model trained in train.py
model_paths=['./tb_bs_16lr_0.001usevgg_0freeze_0epoches_20_b/model',
'./tb_bs_16lr_0.001usevgg_1freeze_0epoches_20/model',
'./tb_bs_16lr_0.001usevgg_1freeze_1epoches_20/model',
'./tb_bs_16lr_0.005usevgg_1freeze_0epoches_20/model',
'./tb_bs_16lr_0.01usevgg_1freeze_0epoches_20/model',
'./tb_bs_32lr_0.001usevgg_1freeze_0epoches_40/model',
'./tb_bs_64lr_0.001usevgg_1freeze_0epoches_80/model',
'./tb_bs_8lr_0.001usevgg_1freeze_0epoches_10/model',
'./tb_multitask_bs_16lr_0.001usevgg_0freeze_0epoches_20/model'
]
model_notes=['TextCNN','VGG-baseline','VGG-freeze','VGG-LR0.005','VGG-LR0.01','VGG-BS32','VGG-BS64','VGG-BS8','TextCNNmultitask']
def evaluate_model(valset,model_path):
    net =torch.load(model_path)
    device = "cuda:0"
    net.to(device)
    valloader = torch.utils.data.DataLoader(valset, batch_size=32,shuffle=True, num_workers=2)
    val_acc=0
    labels,preds=[],[]
    for val_i, val_data in enumerate(valloader, 0):
        val_inputs, val_labels = val_data[0].to(device),val_data[1].to(device)
        results= net(val_inputs)
        if isinstance(results,tuple):
            results=results[0]
        preds.extend(results.max(1)[1].to('cpu').numpy().tolist())
        labels.extend(val_data[1].flatten().numpy().tolist())
        del val_inputs,val_labels,results
    return np.sum(np.array(labels)-np.array(preds)==0)/len(labels)

# evaluate all the models in training, validation and test data
valset = SVHNDataset('val_data.mat',transform=transform)
print('mean Accuracy on validation set')
for i in range (len(model_paths)):
    mean_accuarcy= evaluate_model(valset,model_paths[i])
    print(model_notes[i],mean_accuarcy)
testset = SVHNDataset('test_data.mat',transform=transform)
print('mean Accuracy on Test set')
for i in range (len(model_paths)):
    mean_accuarcy= evaluate_model(testset,model_paths[i])
    print(model_notes[i],mean_accuarcy)
traiingset = SVHNDataset('train_data.mat',transform=transform)
print('mean Accuracy on training set')
for i in range (len(model_paths)):
    mean_accuarcy= evaluate_model(traiingset,model_paths[i])
    print(model_notes[i],mean_accuarcy)

#calculate F1 score of the best chosen model
image_paths=glob.glob('./test/*.png')
PATH=model_paths[1]
output_folder='./results'
transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
detector = TextDetector (PATH,transform=transform)
gt_dict=json.load(open('./digitStruct_test.json'))
tps,fps,fns=[],[],[]

for i,image_name in enumerate(image_paths):
    tp,fp,fn=0,0,0
    image_no=int(image_name.split('/')[-1][:-4])
    bbox=gt_dict[image_no-1]['boxes']
    gts = [[bb_item['left'],bb_item['top'], bb_item['left']+bb_item['width'], bb_item['top']+bb_item['height']] for bb_item in bbox]
    gts_label=[bb_item['label'] for bb_item in bbox]
    I = cv2.imread(image_name)
    results=detector.detection(I)
    scores,labels,roi_detections = results
    
    I = cv2.imread(image_name)
    if roi_detections is None:
        tp,fp=0,0
        fn=len(gts)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        continue
    roi_array=np.array(roi_detections)
    for gt_i in range(len(gts)):
        iou_array=calculate_IOU(gts[gt_i], roi_array)
        if np.max(iou_array)>0.3 and gts_label[gt_i] ==labels[np.argmax(iou_array)]:
            tp+=1
        else:
            fn+=1
    fp+=len(labels)-tp
    tps.append(tp)
    fps.append(fp)
    fns.append(fn)
   
tp_fp_fn_array=(np.sum(np.hstack([np.array(tps).reshape(-1,1),np.array(fps).reshape(-1,1),np.array(fns).reshape(-1,1)]),axis=0))
print('precision is {}, recall is {}'.format(tp_fp_fn_array[0]/(tp_fp_fn_array[0]+tp_fp_fn_array[1]),tp_fp_fn_array[0]/(tp_fp_fn_array[0]+tp_fp_fn_array[2])))