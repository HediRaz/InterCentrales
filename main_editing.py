import os
import sys
sys.path.append(os.path.join(os.path.realpath(os.curdir), "encoder4editing"))
sys.path.append(os.path.join(os.path.realpath(os.curdir), "face_parsing"))
import argparse
from argparse import Namespace
from functools import partial

import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from face_parsing import infer
from encoder4editing.models.psp import pSp



# Load face parsing network
parsing_net = infer.load_model()
print("Parsing model sucessfully loaded!")


# Load e4e model
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
opts['checkpoint_path'] = MODEL_PATH
opts= Namespace(**opts)
generator_net = pSp(opts)
print('e4e model successfully loaded!')


# Hyperparameters for transformations
# The stored values are the path to the vectors and the required projection value
SEX = {"vector_path": "vectors_editing/custom/sex.npy", "female": 1.2, "male": 0.9}
BALD = {"vector_path": "vectors_editing/custom/from_bald.npy", "bald": 2, "hairy": -1.5}
AGE = {"vector_path": "vectors_editing/custom/interface_age.npy", "young": 1.1, "adult": 0.99, "old": 0.96}
CHUBBY = {"vector_path": "vectors_editing/custom/chubby.npy", "big": 0.9, "thin": 1.3}

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


def apply_projection(latents, vector_path, proj_value):
    vector = np.load(vector_path)
    vector = torch.tensor(vector, dtype=latents.dtype, device=latents.device)
    latents = latents + ((proj_value - torch.sum(latents*vector)) / torch.sum(latents*latents))
    return latents


def apply_translation(latents, vector_path, scroll_value):
    vector = np.load(vector_path)
    vector = torch.tensor(vector, dtype=latents.dtype, device=latents.device)
    latents = latents + scroll_value * vector
    return latents

def reencode(img):
    """ img = PIL Image """
    latents = encode(img)
    return decode(latents)


def _reencode(img_path, img_transformations=[], latents_transformations=[]):
    matplotlib.use("TkAgg")
    img = Image.open(img_path).resize((512, 512))
    plt.figure("Original Image")
    plt.imshow(img)

    parsing = infer.compute_mask(img, parsing_net)
    img = np.array(img)
    for t in img_transformations:
        img = t(img, parsing)
    plt.figure("Original Image edited")
    plt.imshow(img)

    latents = encode(Image.fromarray(img))
    for t in latents_transformations:
        latents = t(latents)

    plt.figure("Re-encoded Image")
    plt.imshow(decode(latents))
    plt.show()


parser = argparse.ArgumentParser()
# Transfomations by editing original image and encode/decode it
parser.add_argument("--img_path", type=str, help="Path to image to reencode")
parser.add_argument("--visualize_parsing", action="store_true", help="Visualize face parsing")
parser.add_argument("--hair_color", type=str, help="New hair color", choices=["blond", "brown", "black", "gray"])
parser.add_argument("--hair_color_brut", type=str, help="New hair color abrut", choices=["blond", "brown", "black", "gray"])
parser.add_argument("--bag_under_eyes", type=str, help="Bag under eyes", choices=["min", "max"])
parser.add_argument("--pointy_nose", type=str, help="Pointy nose", choices=["min", "max"])
# Transformations by manipulating the latent representation of the image
parser.add_argument("--chubby", type=str, choices=["big", "thin"],help="Chubby")
parser.add_argument("--age", type=str, choices=["young", "adult", "old"], help="Age")
parser.add_argument("--sex", type=str, choices=["female", "male"], help="Sex")
parser.add_argument("--bald", type=str, choices=["bald", "hairy"], help="Bald")
args = parser.parse_args()


if args.visualize_parsing:
    matplotlib.use("TkAgg")
    img = Image.open(args.img_path).resize((512, 512))
    plt.figure("Original Image")
    plt.imshow(img)
    plt.figure("Parsing")
    infer.vis_parsing_maps(img, infer.compute_mask(img, parsing_net))

img_transformations = []
if args.hair_color:
    img_transformations.append(partial(infer.change_hair_color_smooth, color=args.hair_color))
if args.hair_color_brut:
    img_transformations.append(partial(infer.change_hair_color_brut, color=args.hair_color_brut))
if args.bag_under_eyes:
    img_transformations .append(partial(infer.make_bags, max=args.bag_under_eyes == "max"))
if args.pointy_nose:
    if args.pointy_nose == "max":
        img_transformations.append(infer.make_pointy_nose)
    if args.pointy_nose == "min":
        img_transformations.append(infer.make_flat_nose)

latents_transformations = []
if args.chubby:
    latents_transformations.append(partial(apply_projection, vector_path=CHUBBY["vector_path"], proj_value=CHUBBY[args.chubby]))
if args.age:
    latents_transformations.append(partial(apply_projection, vector_path=AGE["vector_path"], proj_value=AGE[args.age]))
if args.sex:
    latents_transformations.append(partial(apply_projection, vector_path=SEX["vector_path"], proj_value=SEX[args.sex]))
if args.bald:
    latents_transformations.append(partial(apply_translation, vector_path=BALD["vector_path"], scroll_value=BALD[args.bald]))


_reencode(args.img_path, img_transformations, latents_transformations)
