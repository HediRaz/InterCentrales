"""Reencode the latent space after manipulation."""

import argparse
from argparse import Namespace
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from encoder4editing.models.psp import pSp
from face_parsing import infer

matplotlib.use("TkAgg")
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
opts = Namespace(**opts)
generator_net = pSp(opts)
# net.eval()
# net.cuda()
print('Generation model successfully loaded!')


def encode(img):
    """Encode an image.

    Parameter
    ---------
    img : PIL Image
        Image to encode.

    Returns
    -------
    latents : torch.Tensor
        Encoded image.
    """
    latents = (generator_net.encoder(img_transforms(img)
                                     .unsqueeze(0).to("cuda").float())[0])
    latents += generator_net.latent_avg
    return latents


def decode(latents):
    """Decode latent vector.

    Parameter
    ---------
    latents : torch.Tensor
        Latent vector to decode.

    Returns
    -------
    edit_image : PIL Image
        Decoded edited image.
    """
    edit_image = generator_net.decoder([latents.unsqueeze(0)],
                                       randomize_noise=False,
                                       input_is_latent=True)[0][0]
    edit_image = (edit_image.detach().cpu().transpose(0, 1)
                  .transpose(1, 2).numpy())
    edit_image = ((edit_image + 1) / 2)
    edit_image[edit_image < 0] = 0
    edit_image[edit_image > 1] = 1
    edit_image = edit_image * 255
    edit_image = edit_image.astype("uint8")
    edit_image = Image.fromarray(edit_image)
    edit_image = edit_image.resize((512, 512))

    return edit_image


def reencode(img):
    """Re-encode an image."""
    latents = encode(img)
    return decode(latents)


def _reencode(img_path, transformations=None):
    """Re-encode an image with transformations."""
    if transformations is None:
        transformations = []
    matplotlib.use("TkAgg")
    img = Image.open(img_path).resize((512, 512))
    plt.figure("Original Image")
    plt.imshow(img)

    parsing = infer.compute_mask(img, parsing_net)
    img = np.array(img)
    for transfo in transformations:
        img = transfo(img, parsing)
    plt.figure("Original Image edited")
    plt.imshow(img)

    plt.figure("Re-encoded Image")
    plt.imshow(reencode(Image.fromarray(img)))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str,
                        help="Path to image to reencode")
    parser.add_argument("--visualize_parsing", action="store_true",
                        help="Visualize face parsing")
    parser.add_argument("--hair_color", type=str, help="New hair color",
                        choices=["blond", "brown", "black", "gray"])
    parser.add_argument("--hair_color_brut", type=str,
                        help="New hair color abrut",
                        choices=["blond", "brown", "black", "gray"])
    parser.add_argument("--bag_under_eyes", type=str, help="Bag under eyes",
                        choices=["min", "max"])
    parser.add_argument("--pointy_nose", type=str, help="Pointy nose",
                        choices=["min", "max"])
    parser.add_argument("--chubby", action="store_true", help="Chubby")
    args = parser.parse_args()

    if args.visualize_parsing:
        matplotlib.use("TkAgg")
        img = Image.open(args.img_path).resize((512, 512))
        plt.figure("Original Image")
        plt.imshow(img)
        plt.figure("Parsing")
        infer.vis_parsing_maps(img, infer.compute_mask(img, parsing_net))

    transformations = []
    if args.hair_color:
        transformations.append(partial(infer.change_hair_color_smooth,
                                       color=args.hair_color))
    if args.hair_color_brut:
        transformations.append(partial(infer.change_hair_color_brut,
                                       color=args.hair_color_brut))
    if args.bag_under_eyes:
        transformations .append(partial(infer.make_bags,
                                        max=args.bag_under_eyes == "max"))
    if args.pointy_nose:
        if args.pointy_nose == "max":
            transformations.append(infer.make_pointy_nose)
        if args.pointy_nose == "min":
            transformations.append(infer.make_flat_nose)
    if args.chubby:
        transformations.append(infer.make_balls_around_mouth)

    _reencode(args.img_path, transformations)
