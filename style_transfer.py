import os
import torch
import argparse


def style_transfer():
    # image preprocessing
    
    # get style and content representations
    
    # initialize output image (random pixels)
    
    # train the output image
    
    # save


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    style_images_dir = os.path.join(os.path.dirname(__file__), "style_images")
    content_images_dir = os.path.join(os.path.dirname(__file__), "content_images")
    
    # you can tweak these vv
    configs = {
        "style_image_name": "vg_self.jpg",
        "content_image_name": "my_pic.jpg",
        "style_loss_weight": 1,
        "content_loss_weight": 1,
        "variance_loss_weight": 1,
        "output_dir": output_dir,
        "style_images_dir": style_images_dir,
        "content_images_dir": content_images_dir
    }
    
    # TODO: select device based on availability
    
    style_transfer(configs)
