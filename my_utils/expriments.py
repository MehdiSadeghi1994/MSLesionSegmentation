import os
import time
import torch
import copy
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import nibabel as nib
from torch.nn import functional as F

from my_utils.evaluation_metrics import dice_coefficients


def train_model(model, dataloaders, criterion, optimizer, device, num_data, input_shape, output_shape, n_classes=2, val_step=1, scheduler=None, model_name='Best_Model', max_epoch=50):
    if torch.cuda.is_available():
        print('\n-------training mode is on CUDA-------\n')
    else:
        print('\n-------training mode is on CPU-------\n')

    if os.path.exists(f'Checkpoint_Files/{model_name}_checkpoint.tar'):
        checkpoint = torch.load(f'Checkpoint_Files/{model_name}_checkpoint.tar', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_model_state = checkpoint['model_state_dict']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        last_epoch = checkpoint['epoch']
        best_dice = checkpoint['best_dice']
        losses = checkpoint['losses']
        dices = checkpoint['dices']

    else:
        last_epoch = 0
        best_dice = 0.0
        losses= {'train':[], 'validation':[]}
        dices= {'train':[], 'validation':[]}

    slices_in = [slice(None), slice(None)] + [
    slice(int((len_in - len_out) / 2), int(len_in - (len_in - len_out) / 2))
    for len_out, len_in in zip(output_shape, input_shape)
    ]   
    
    
    model.to(device)
    model.train()
    for epoch in range(last_epoch, max_epoch):

        running_loss = 0
        running_dice = 0
        with tqdm(total=num_data['train'], desc=f'Epoch {epoch + 1}/{max_epoch}', unit='img') as pbar:
            for image, label in dataloaders['train']:
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(mode=True):
                    logit = model(image)
                    logit = logit[slices_in]
                    logit = F.log_softmax(logit, 1)

                    _, predic = torch.max(logit, 1)

                    loss = criterion(logit, label)

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * image.shape[0]

                if label.ndim == 5:
                    dice_1 = dice_coefficients(predic.cpu().numpy(), label[:,:,:,:, 0].cpu().numpy())
                    dice_2 = dice_coefficients(predic.cpu().numpy(), label[:,:,:,:, 1].cpu().numpy())
                    temp = [(dice_1[i] + dice_2[i])/2 for i in range(len(dice_1))] 
                    running_dice += ((temp[0]+temp[1])/2) * image.shape[0]

                else:
                    dice_temp = dice_coefficients(predic.cpu().numpy(), label.cpu().numpy())
                    running_dice += ((dice_temp[0]+dice_temp[1])/2) * image.shape[0]
                
                
                
                pbar.update(image.shape[0])


        epoch_loss = running_loss / num_data['train']
        epoch_dice = running_dice / num_data['train']
        
        print(f'Train: Loss= {epoch_loss}, Dice= {epoch_dice}')
            
        if scheduler:
            scheduler.step()
        # Validate model
        if (epoch+1) % val_step == 0:
            val_loss, val_dice = validate_model(model,
                                                dataloaders['validation'],
                                                criterion,
                                                device,
                                                num_data['validation'],
                                                n_classes,
                                                input_shape,
                                                output_shape
                                                )
            losses['validation'].append(val_loss)
            losses['train'].append(epoch_loss)

            dices['train'].append(epoch_dice)
            dices['validation'].append(val_dice)
            print(f'      Validation: Loss= {val_loss}, Dice= {val_dice}\n\n ')

        if val_dice > best_dice:
            best_dice = val_dice
            best_model_state = copy.deepcopy(model.state_dict())



        torch.save({'epoch': epoch+1,
                    'model_state_dict': best_model_state,
                    'best_dice': best_dice,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                    'dices' : dices,
        }, f'Checkpoint_Files/{model_name}_checkpoint.tar')

    print('Best validation Dice: {:4f}'.format(best_dice))
    
    plt.plot(losses['train'], label='training loss')
    plt.plot(losses['validation'], label='validation loss')
    idx = np.argmin(losses['validation'])
    plt.plot([idx,idx], [0,losses['validation'][idx]],'--', color= 'coral' , label = 'Best' )
    plt.plot([0,idx], [losses['validation'][idx],losses['validation'][idx]],'--', color= 'coral' )
    plt.plot(idx, losses['validation'][idx] , 'o' ,color = 'coral', markersize = 8, markerfacecolor = "None" )
    plt.legend()
    plt.ylabel('Loss', fontsize= 12)
    plt.xlabel('Epoch',fontsize= 12)
    plt.tight_layout()
    plt.title(f'{model_name}_train-validation loss ',fontsize= 12)
    plt.savefig(f'loss_{model_name}.png',bbox_inches='tight')
    plt.show()
    

    model.load_state_dict(best_model_state)
    return model


def validate_model(model, dataloader, criterion, device, num_data, n_classes, input_shape, output_shape):
    print('   validation is in process....')
    model = model.eval()
    model = model.to(device)
    running_dice = 0
    running_loss = 0
    with tqdm(total=num_data) as bar:
        bar.set_description('       Validation: ')
        for data in dataloader:
            grand_truth = data['label']
      
            if grand_truth.ndim == 5:
                image_shape = grand_truth.shape[1:-1]
            else:
                image_shape = grand_truth.shape[1:]

            result = feedforward(model, data, n_classes, input_shape, output_shape, device, image_shape)
            result = result.unsqueeze(dim=0)
            grand_truth = grand_truth.to(device)
            
            loss = criterion(result, grand_truth)
            running_loss += loss.item()

            result = result.to('cpu')
            result = result.numpy()
            result = np.int32(np.argmax(result, axis=1))
            grand_truth = grand_truth.cpu().numpy()
            if grand_truth.ndim == 5:
                dice_1 = dice_coefficients(result, grand_truth[:,:,:,:, 0])
                dice_2 = dice_coefficients(result, grand_truth[:,:,:,:, 1])
                temp = [(dice_1[i] + dice_2[i])/2 for i in range(len(dice_1))] 
                
                
                running_dice += ((temp[0]+temp[1])/2) 

            else:
                temp = dice_coefficients(result, grand_truth)
                running_dice += ((temp[0]+temp[1])/2)


            bar.update(n=1)
        

    val_loss = running_loss / num_data
    val_dice = running_dice / num_data
    return val_loss, val_dice



def test_model(model, dataloder, num_data, n_classes, input_shape, output_shape, device, image_suffix='FUMResult.nii.gz',  result_path=None):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model.eval()
    with tqdm(total=num_data) as bar:
        bar.set_description('Test: ')
        for data in dataloder:
            
            image_path = data['image_path']
            
            if result_path:
                base_path = result_path

            else:
                base_path = os.path.split(image_path[0])[0]

            image_name = ''
            for item in os.path.split(image_path[0])[1].split('_')[:-1]:
                image_name += item+'_'
            image_name = image_name + image_suffix

            image_shape = tuple(i.numpy()[0] for i in data['image_shape'][0:3])
            result = feedforward(model, data, n_classes, input_shape, output_shape, device, image_shape)
            result = result.to('cpu')
            result = result.numpy()
            result = np.int32(np.argmax(result, axis=0))

            
            img = nib.Nifti1Image(result.astype(np.int32), np.eye(4))
            nib.save(img, os.path.join(base_path,image_name))

            bar.update(n=1)




def feedforward(model, data, n_classes, input_shape, output_shape, device, image_shape):
    model.eval()
    model = model.to(device)
    with torch.set_grad_enabled(False):
        result = torch.zeros((n_classes,) + image_shape).to(device)
        for patch, cordinates in zip(data['patches'], data['cordinates']):

            slices_out = [slice(None)] + [
                slice(int(center - len_out // 2), int(center + len_out // 2))
                for len_out, center in zip(output_shape, cordinates)
            ]
            
            slices_in = [0, slice(None)] + [
                slice(int((len_in - len_out) // 2), int((len_out - len_in) // 2))
                for len_out, len_in, in zip(output_shape, input_shape)
            ]

            patch = patch.to(device)
            predict = model(patch)
            result[slices_out] += predict[slices_in]
        
    return result         







       