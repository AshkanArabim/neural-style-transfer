import torch
import torchvision
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = std=[0.229, 0.224, 0.225]


def vgg_preprocessing(image, device):
    # note: image should already be a pytorch tensor
    # image = image[:, :, ::-1].double() / 255
    image = torch.flip(image, [-1]).float().to(device) / 255
    
    # move channels first (so it works with conv2d)
    # convert tims to [ch, h, w]
    image = torch.moveaxis(image, -1, 0)

    # resize
    image = torchvision.transforms.Resize((256, 256))(image)

    # central crop to 224 by 224
    # image = image[:, 16:-16, 16:-16]
    image = torchvision.transforms.functional.crop(image, 16, 16, 224, 224)

    # normalize
    imagenet_mean_tensor = torch.FloatTensor(imagenet_mean).view(3, 1, 1).to(device)
    imagenet_std_tensor = torch.FloatTensor(imagenet_std).view(3, 1, 1).to(device)
    image = torch.sub(image, imagenet_mean_tensor)
    image = torch.div(image, imagenet_std_tensor)
    
    return image


def prepare_image(image_folder, image_name, device):
    image_path = os.path.join(image_folder, image_name)
    # reverse BGR to RGB and compress between 0 and 1
    # dims: [h, w, BGR]
    image = cv.imread(image_path)
    
    # save image dims
    dims = image.shape
    
    image = torch.from_numpy(image)
    
    # based on: https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
    image = vgg_preprocessing(image, device)
    
    return (image, dims)


def calculate_loss(output_image, out_style_rep, out_content_rep, style_rep, content_rep, configs):
    # get the content loss
    content_loss = torch.nn.MSELoss(reduction='mean')(out_content_rep, content_rep)
    
    # STOP AND THINK vv
    # the style loss
    style_loss = 0
    for i in range(len(out_style_rep)):
        # don't forget gram matrix
        out_style_gram = gram_matrix(out_style_rep[i])
        src_style_gram = gram_matrix(style_rep[i])
        
        # show images of style and content reps
        if (configs['save_reps']):
            with torch.no_grad():
                draw_matrix(src_style_gram, f"src_style_gram_{i}.jpg")
                draw_matrix(out_style_gram, f"out_style_gram_{i}.jpg")
        
        style_loss += torch.nn.MSELoss(reduction='mean')(out_style_gram, src_style_gram)
    style_loss /= len(style_rep)
    
    # variance loss: absolutely needed
    variance_loss = 0
    variance_loss += torch.nn.MSELoss(reduction='mean')(
        output_image[1:, :], output_image[:-1, :] # vertical diff
    )
    variance_loss += torch.nn.MSELoss(reduction='mean')(
        output_image[:, 1:], output_image[:, :-1] # horizontal diff
    )
    
    # add together and apply style / content weights
    total_loss = content_loss * configs["content_loss_weight"] + \
        style_loss * configs["style_loss_weight"] + \
        variance_loss * configs["variance_loss_weight"]
    
    return (variance_loss, content_loss, style_loss, total_loss)

    
def gram_matrix(matrix):
    # flatten to [ch, h * w]
    ch, h, w = matrix.size()
    matrix = matrix.view(ch, -1)
    gram = torch.matmul(matrix, torch.transpose(matrix, 0, 1))
    gram /= h * w
    
    return gram


def draw_matrix(style_rep, name):
    # style_rep = [np.asarray(x.to('cpu')) for x in style_rep]
    style_rep = np.asarray(style_rep.to('cpu'))
    
    # for now I don't care which one
    plt.imshow(style_rep, interpolation='none')
    plt.savefig(name)
