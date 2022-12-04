from matplotlib import pyplot as plt
import torch
import numpy as np

def show_prediction(model, data_loader, device, num):
    fig, axs = plt.subplots(nrows=num, ncols=3, figsize=(5,10), squeeze=False)
    # plt.subplots_adjust(hspace = 0.3)
    # axs = axs.ravel()

    model.eval()
    image_so_far = 0

    for data_info in data_loader:
        with torch.no_grad():
            image = data_info['image'].to(device)
            label = data_info['mask'].to(device)
            
            output = model(image)[0]
            _, predic = torch.max(output, 1)
            
            image = image.cpu().numpy().squeeze() # out_size = c*h*W
            image = image.transpose(1,2,0).astype(np.uint8) # out_size = h*w*c 

            label = label.cpu().numpy().squeeze()

            predic = predic.cpu().numpy().squeeze()

            axs[image_so_far,0].imshow(image)
            axs[image_so_far,0].axis('off') 

            axs[image_so_far,1].imshow(label)
            axs[image_so_far,1].axis('off')

            axs[image_so_far,2].imshow(predic)
            axs[image_so_far,2].axis('off')

            
            image_so_far+=1

            if image_so_far==num :
                plt.savefig('show_predication.png')
                plt.show()
                return