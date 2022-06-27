import os
import sys

sys.path.append(os.path.join(os.path.realpath(os.curdir), "encoder4editing"))
sys.path.append(os.path.join(os.path.realpath(os.curdir), "face_parsing"))
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
opts = Namespace(**opts)
generator_net = pSp(opts)
print('e4e model successfully loaded!')


def apply_projection(latents, vector_path, proj_value):
    vector = np.load(vector_path)
    vector = torch.tensor(vector, dtype=latents.dtype, device=latents.device)
    alpha = proj_value - (torch.sum(latents*vector)/torch.sum(vector*vector))
    edited_latents = latents + (alpha * vector / 18)
    if torch.abs(proj_value - torch.sum(edited_latents*vector)/torch.sum(vector*vector)) > 0.1:
        alpha = proj_value - (torch.sum(latents*vector)/torch.sum(vector*vector))
        edited_latents = latents + (alpha * vector / 1)
    return edited_latents


def apply_translation(latents, vector_path, scroll_value):
    vector = np.load(vector_path)
    vector = torch.tensor(vector, dtype=latents.dtype, device=latents.device)
    latents = latents + scroll_value * vector
    return latents


# Hyperparameters for transformations
# The stored values are the path to the vectors and the required projection value
LATENT_TRANSFORMATIONS = {
    "Se_0": partial(apply_projection, vector_path="vectors_editing/custom/sex.npy", proj_value=1.2),
    "Se_1": partial(apply_projection, vector_path="vectors_editing/custom/sex.npy", proj_value=-1),
    "Bald": partial(apply_projection, vector_path="vectors_editing/custom/from_bald.npy", proj_value=5.0),
    "make_hair": partial(apply_translation, vector_path="vectors_editing/custom/from_bald.npy", scroll_value=-1.5),
    "A_0": partial(apply_projection, vector_path="vectors_editing/custom/interface_age.npy", proj_value=-20),
    "A_1": partial(apply_projection, vector_path="vectors_editing/custom/interface_age.npy", proj_value=5),
    "A_2": partial(apply_projection, vector_path="vectors_editing/custom/interface_age.npy", proj_value=50),
    "Ch_min": partial(apply_projection, vector_path="vectors_editing/custom/chubby.npy", proj_value=-0.5),
    "Ch_max": partial(apply_projection, vector_path="vectors_editing/custom/chubby.npy", proj_value=1.2),
    "Ne_min": partial(apply_projection, vector_path="vectors_editing/custom/4_7_narrow_eyes.npy", proj_value=-5),
    "Ne_max": partial(apply_projection, vector_path="vectors_editing/custom/4_7_narrow_eyes.npy", proj_value=20),
    "B_0": partial(apply_projection, vector_path="vectors_editing/custom/bangs.npy", proj_value=-1.5),
    "B_1": partial(apply_projection, vector_path="vectors_editing/custom/bangs.npy", proj_value=3),
    "D_0": partial(apply_projection, vector_path="vectors_editing/custom/46_4_double_chin.npy", proj_value=-0.3),
    "D_1": partial(apply_projection, vector_path="vectors_editing/custom/46_4_double_chin.npy", proj_value=1),
}
IMG_TRANSFORMATIONS = {
    "Hc_0": partial(infer.change_hair_color_smooth, color="black"),
    "Hc_1": partial(infer.change_hair_color_smooth, color="blond"),
    "Hc_2": partial(infer.change_hair_color_smooth, color="blond"),
    "Hc_3": partial(infer.change_hair_color_smooth, color="gray"),
    "Bn_max": infer.make_big_nose,
    "Bp_max": infer.make_big_lips,
    "Be_max": infer.make_bags,
    "Pn_max": infer.make_pointy_nose,
}
IMG_TRANSFORMATIONS_INVERSE = {
    "Bn_min": {"function": infer.make_big_nose, "value": 1},
    "Bp_min": {"function": infer.make_big_lips, "value": 1},
    "Be_min": {"function": infer.make_bags, "value": 0.5},
    "Pn_min": {"function": infer.make_pointy_nose, "value": 1},
}


def encode(img):
    """ img = PIL Image """
    with torch.no_grad():
        latents = generator_net.encoder(img_transforms(img).unsqueeze(0).to("cuda").float())[0]
        latents += generator_net.latent_avg
    return latents


def decode(latents):
    """ Return PIL Image"""
    with torch.no_grad():
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


def parse_img_name(img_name):
    Sk, A, Se, C, P, B, Hc, D, Hs = img_name.split("_")
    Hs = Hs.split(".")[0]
    return dict(Sk=Sk, A=A, Se=Se, C=C, P=P, B=B, Hc=Hc, D=D, Hs=Hs)


POSSIBLE_VALUES = {
    "Sk": {"0", "1", "2"},
    "A": {"0", "1", "2"},
    "Se": {"0", "1"},
    "B": {"0", "1"},
    "Hc": {"0", "1", "2", "3"},
    "D": {"0", "1"},
    "Hs": {"0", "1"}
}
CURSOR_FEATURES = ("Be", "N", "Pn", "Bp", "Bn", "Ch")


def get_img_transformations(img_name, list_of_transformations):
    img_att = parse_img_name(img_name)
    transformations = []
    for att in POSSIBLE_VALUES:
        if att in list_of_transformations:
            for val in POSSIBLE_VALUES[att]:
                if img_att[att] != val:
                    transformations.append(att+"_"+val)
    for att in CURSOR_FEATURES:
        if att in list_of_transformations:
            transformations.append(att+"_max")
            transformations.append(att+"_min")
    if "Bald" in list_of_transformations and img_att["Hc"] != 4:
        transformations.append("Bald")
    return transformations


def process_img(img_path, destination_folder, list_of_transformations):
    img_name = os.path.basename(img_path)

    destination_folder = os.path.join(destination_folder, img_name[:-4])
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    img_att = parse_img_name(img_name)
    img_transforms = get_img_transformations(img_name, list_of_transformations)

    img = Image.open(img_path).resize((512, 512))
    parsing = infer.compute_mask(img, parsing_net)
    latents = encode(img)

    for t in img_transforms:
        if t in LATENT_TRANSFORMATIONS:
            edited_img = decode(LATENT_TRANSFORMATIONS[t](latents))
            edited_img.save(os.path.join(destination_folder, t+".png"))

    for t in img_transforms:
        if t in IMG_TRANSFORMATIONS:
            try:
                if "Hc" in t and img_att["Hc"] == "4":
                    edited_img = decode(LATENT_TRANSFORMATIONS["make_hair"](latents))
                    edited_img = IMG_TRANSFORMATIONS[t](np.array(edited_img), parsing)
                else:
                    edited_img = IMG_TRANSFORMATIONS[t](np.array(img), parsing)
                edited_img = reencode(Image.fromarray(edited_img))
                edited_img.save(os.path.join(destination_folder, t+".png"))
            except Exception:
                print("WARNING: Error while processing image:", img_path, "with transformation:", t)

    for t in img_transforms:
        if t in IMG_TRANSFORMATIONS_INVERSE:
            try:
                edited_img = IMG_TRANSFORMATIONS_INVERSE[t]["function"](np.array(img), parsing)
                edited_latents = encode(Image.fromarray(edited_img))
                edited_img = decode(latents + IMG_TRANSFORMATIONS_INVERSE[t]["value"]*(latents - edited_latents))
                edited_img.save(os.path.join(destination_folder, t+".png"))
            except Exception:
                print("WARNING: Error while processing image:", img_path, "with transformation:", t)


def iterate_over_dataset(dataset_folder, destination_folder, list_of_transformations):
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    for img_name in os.listdir(dataset_folder):
        img_path = os.path.join(dataset_folder, img_name)
        try:
            process_img(img_path, destination_folder, list_of_transformations)
        except Exception:
            print("WARNING: Error while processing image:", img_path)


if __name__ == "__main__":
    LIST_OF_TRANSFORMATIONS = ["A", "B", "Bald", "Be", "Bn", "Bp", "Ch", "D", "Hc", "Hs", "N", "Pn", "Se", "Sk"]

    iterate_over_dataset("CeterisParibusDataset/", "Test", LIST_OF_TRANSFORMATIONS)
