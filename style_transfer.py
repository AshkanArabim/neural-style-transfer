import os
import torch
import torchvision
import cv2 as cv
import utils
import numpy as np
from vgg19 import Vgg19


def style_transfer(configs):
    # image preprocessing
    style_image, _ = utils.prepare_image(configs["style_images_dir"], configs["style_image_name"], configs["device"])
    content_image, content_dims = utils.prepare_image(configs["content_images_dir"], configs["content_image_name"], configs["device"])
    
    # get style and content representations
    neural_net = Vgg19().to(configs["device"])
    style_reps, _ = neural_net(style_image)
    _, content_rep = neural_net(content_image)
    
    # initialize output image (random pixels) --> we will tune this
    output_image = torch.from_numpy(np.random.rand(*content_dims))
    output_image = torch.autograd.Variable(output_image, requires_grad=True)
    
    # train the output image
    optimizer = torch.optim.Adam([output_image], lr = 1e-1)
    for i in range(configs["max_iters"]):
        optimizer.zero_grad()
        output_image_preprocessed = utils.vgg_preprocessing(output_image, configs["device"])
        out_style_reps, out_content_rep = neural_net(output_image_preprocessed)
        content_loss, style_loss, loss = utils.calculate_loss(out_style_reps, out_content_rep, style_reps, content_rep, configs)
        # print("iteration:", i, "\tloss:", loss)
        print(f"iter {i}/{configs['max_iters']} | loss: {loss}\t\tcontent_loss: {content_loss}\t\tstyle_loss: {style_loss}")
        loss.backward(retain_graph=True)
        optimizer.step()
    
    # convert to pixel values
    output_image = (output_image.detach().numpy() * 255).astype(int)
    # save output
    cv.imwrite(
        # os.path.join(configs["output_dir"], f"combined_{configs['style_image_name']}_{configs['content_image_name']}"),
        configs["output_name"],
        output_image
    )
    print('saved to:', configs["output_name"])


if __name__ == "__main__":
    # you can tweak these vv
    configs = {
        "save_reps": False, # breaks algorithm, but you'll get initial pics
        "style_image_name": "vg_self.jpg",
        "content_image_name": "my_pic.jpg",
        "style_loss_weight": 1,
        "content_loss_weight": 1,
        "variance_loss_weight": 1,
        "max_iters": 1000,
        # don't tweak below this line -------------
        "output_name": "out.jpg",
        # "output_dir": os.path.join(os.path.dirname(__file__), "output"),
        "style_images_dir": os.path.join(os.path.dirname(__file__), "style_images"),
        "content_images_dir": os.path.join(os.path.dirname(__file__), "content_images"),
        "device": (
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.mps.is_availabe()
            else "cpu"
        )
    }
    
    style_transfer(configs)
