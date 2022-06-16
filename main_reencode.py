import os
import sys
sys.path.append(os.path.join(os.path.realpath(os.curdir), "encoder4editing"))
sys.path.append(os.path.join(os.path.realpath(os.curdir), "face_parsing"))
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from argparse import Namespace
from PIL import Image
from torchvision import transforms
from face_parsing import infer
from encoder4editing.models.psp import pSp



parsing_net = infer.load_model()
print("Parsing model sucessfully loaded!")


MODEL_PATH = "encoder4editing/pretrained_models/e4e_ffhq_encode.pt"
resize_dims = (256, 256)
# Setup required image transformations
img_transforms = transforms.Compose([
    transforms.Resize(resize_dims),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

ckpt = torch.load(MODEL_PATH, map_location='cpu')
for u in list(ckpt.keys()):
    if "opts" not in u:
        ckpt.pop(u)
opts = ckpt['opts']
# pprint.pprint(opts)  # Display full options used
# update the training options
opts['checkpoint_path'] = MODEL_PATH
opts= Namespace(**opts)
generator_net = pSp(opts)
# net.eval()
# net.cuda()
print('Generation model successfully loaded!')


def encode(img):
    """ img = PIL Image """
    latents = generator_net.encoder(img_transforms(img).unsqueeze(0).to("cuda").float())[0]
    latents += generator_net.latent_avg
    return latents


def decode(latents):
    """ Return PIL Image"""
    edit_image = generator_net.decoder([latents.unsqueeze(0)], randomize_noise=False, input_is_latent=True)[0][0]
    edit_image = edit_image.detach().cpu().transpose(0, 1).transpose(1, 2).numpy()
    edit_image = ((edit_image + 1) / 2)
    edit_image[edit_image < 0] = 0
    edit_image[edit_image > 1] = 1
    edit_image = edit_image * 255
    edit_image = edit_image.astype("uint8")
    edit_image = Image.fromarray(edit_image)
    edit_image = edit_image.resize((512, 512))

    return edit_image


def reencode(img):
    """ img = PIL Image """
    latents = encode(img)
    return decode(latents)


def _reencode(img_path):
    img = Image.open(img_path)
    plt.figure("Original Image")
    plt.imshow(img)
    plt.figure("Re-encoded Image")
    plt.imshow(reencode(img))
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--reencode", type=str, help="Path to image to reencode")
args = parser.parse_args()


if args.reencode:
    _reencode(args.reencode)
