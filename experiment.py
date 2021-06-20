import torch
import torchvision.transforms as transforms
import os
from data_pipeline import SVHNDataset
from train import train_model

def batch_size_experiment(trainset, valset, batch_size_params):
    for batch_size_param in batch_size_params:
        train_model(
            trainset, 
            valset, 
            use_VGG=True, 
            train_batch_size=batch_size_param[0], 
            lr=0.001, 
            epoches=batch_size_param[1], 
            momentum=0.9
            )

def learning_rate_experiment(trainset, valset, lrs):
    for lr in lrs:
        train_model(
            trainset, 
            valset, 
            use_VGG=True, 
            train_batch_size=16,
            lr=lr, 
            epoches=20, 
            momentum=0.9
            )

def network_variation_experiment(trainset, valset, use_VGG):
    train_model(
        trainset,
        valset,
        use_VGG=use_VGG,
        train_batch_size=16,
        lr=0.001,
        epoches=20,
        momentum=0.9
        )

def free_weights_experiement(trainset, valset, freeze):
    train_model(
        trainset, 
        valset, 
        use_VGG=True,
        train_batch_size=16, 
        lr=0.001, 
        freeze=freeze, 
        epoches=20, 
        momentum=0.9
        )

def multi_task_experiment(trainset, valset, multitask):
    train_model(
        trainset,
        valset,
        use_VGG=False,
        multitask=multitask,
        train_batch_size=16,
        lr=0.001,
        epoches=20,
        momentum=0.9
        )

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    torch.backends.cudnn.benchmark = True

    transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # numbers are from VGG
    
    trainset = SVHNDataset('train_data.mat', transform=transform)
    valset = SVHNDataset('val_data.mat', transform=transform)

    # batch size experiment
    batch_sizes = [8, 16, 32, 64]
    epoches = [10, 20, 40, 80]
    batch_size_experiment(trainset, valset, zip(batch_sizes, epoches))

    # learning rate experiment
    lrs = [0.001, 0.005, 0.01]
    learning_rate_experiment(trainset, valset, lrs)

    # compare Text-Attentional CNN and VGG
    use_VGG = False
    network_variation_experiment(trainset, valset, use_VGG)

    # compare pretrained and retrained weights of VGG
    freeze = True
    free_weights_experiement(trainset, valset, freeze)

    # compare single-task mode and multi-task mode of Text-Attentional CNN
    multitask = True
    multi_task_experiment(trainset, valset, multitask)

if __name__ == "__main__":
    main()