import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from os.path import join
from model import TextAttentionCNN, FineTuneVGG, step_single, step_multitask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(trainset, valset, use_VGG, train_batch_size, lr, epoches, multitask=False, freeze=False, momentum=0.9):
    val_batch_size = 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory='True')
    valloader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size, shuffle=True, num_workers=2)
    if multitask:
        log_dir = './tb_multitask_bs_{}lr_{}usevgg_{}freeze_{}epoches_{}'.format(
            str(train_batch_size), str(lr), 
            str(int(use_VGG)), str(int(freeze)), str(epoches)
            )
    else:
        log_dir = './tb_bs_{}lr_{}usevgg_{}freeze_{}epoches_{}'.format(
            str(train_batch_size), str(lr), 
            str(int(use_VGG)), str(int(freeze)), str(epoches)
            )
    
    # use tensorboard to monitor training
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_scalar('Loss/train', 0, 0)
    writer.add_scalar('Loss/val', 0, 0)
    writer.add_scalar('Acc/train', 0, 0)
    writer.add_scalar('Acc/val', 0, 0)

    # network definition 
    if use_VGG:
        net = FineTuneVGG(freeze=freeze)
    else:
        net = TextAttentionCNN(multitask=multitask)
    net.to(device)

    # optimizer definition
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    val_acc_value_max = 0.0
    previous_train_step = 0
    
    for epoch in range(epoches):
        training_loss = 0.0
        training_acc = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            if multitask:
                loss, acc = step_multitask(inputs, labels, net, optimizer, True)
            else:
                loss, acc = step_single(inputs, labels, net, optimizer, True)

            # compute the EMA of training loss and accuracy (alpha = 0.01)
            training_acc = training_acc * 0.99 + 0.01 * acc
            training_loss = training_loss * 0.99 + 0.01 * loss.item()
            del inputs,labels,loss,acc

            # save the log for every 1000 iteration
            if (i % 1000 == 1) and i > 1000:
                print(i, training_loss / i, float(training_acc))
                writer.add_scalar('Loss/train', training_loss, i + previous_train_step)
                writer.add_scalar('Acc/train', float(training_acc), i + previous_train_step)
        previous_train_step += i

        # validation after each epoch of training
        val_loss = 0
        val_acc = 0.0
        for val_i, val_data in enumerate(valloader, 0):
            val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)

            if multitask:
                val_loss_item, val_acc_item = step_multitask(val_inputs, val_labels, net, optimizer, False)
            else:
                val_loss_item, val_acc_item = step_single(val_inputs, val_labels, net, optimizer, False)
            
            val_loss += val_loss_item.item()
            val_acc += val_acc_item
            del val_inputs, val_labels,val_loss_item,val_acc_item
        
        # compute the validation loss and accuracy
        val_acc_value = float(val_acc) / val_i
        writer.add_scalar('Loss/val', val_loss/val_i, epoch)
        writer.add_scalar('Acc/val', val_acc_value, epoch)
        print('val accuracy:')
        print(val_acc_value)

        #save the best model
        if val_acc_value > val_acc_value_max:
            val_acc_value_max = val_acc_value
            PATH = join(log_dir, 'model')
            torch.save(net, PATH)