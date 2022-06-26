from dataclasses import replace
import os
import numpy as np
from PIL import Image
from infer import get_nose_mask, load_model, compute_mask, get_background_mask, delete_foreground, replace_obj



def global_change(origin_folder, edit_folder, dest_folder="results"):
    net = load_model()

    for img_name in os.listdir(origin_folder):
        img_path = os.path.join(origin_folder, img_name)
        original_img = Image.open(img_path)
        original_parsing = compute_mask(original_img, net)
        original_bck, o_bck_pos = get_background_mask(original_parsing, add_glasses=True, add_earings=True)

        img_edit_folder = os.path.join(edit_folder, img_name[:-4])
        dest_img_folder = os.path.join(dest_folder, img_name[:-4])
        for img_edit_name in os.listdir(img_edit_folder):
            img_edit_path = os.path.join(img_edit_folder, img_edit_name)
            edit_image = Image.open(img_edit_path)
            edit_parsing = compute_mask(edit_image, net)
            edit_bck, e_bck_pos = get_background_mask(edit_parsing, add_glasses=True, add_earings=True)

            edit_image = np.array(edit_image)
            edit_image = replace_obj(
                origin_image=np.array(original_img),
                edit_image=edit_image,
                origin_mask=np.logical_not(original_bck),
                edit_mask=np.logical_not(edit_bck),
                o_pos=o_bck_pos,
                e_pos=e_bck_pos,
                blur_foreground=True
                )
            
            dest_path = os.path.join(dest_img_folder, img_edit_name)
            edit_image = Image.fromarray(edit_image)
            edit_image.save(dest_path)


def nose_change(origin_folder, edit_folder, dest_folder="results"):
    net = load_model()

    for img_name in os.listdir(origin_folder):
        img_path = os.path.join(origin_folder, img_name)
        o_img = Image.open(img_path)
        o_parsing = compute_mask(o_img, net)
        o_nose_mask, o_nose_pos = get_background_mask(o_parsing, add_glasses=True, add_earings=True)

        img_edit_folder = os.path.join(edit_folder, img_name[:-4])
        dest_img_folder = os.path.join(dest_folder, img_name[:-4])
        for img_edit_name in ["Bn_min.png", "Bn_max.png", "Pn_min.png", "Pn_max.png"]:
            img_edit_path = os.path.join(img_edit_folder, img_edit_name)
            e_image = Image.open(img_edit_path)
            e_parsing = compute_mask(e_image, net)
            e_nose_mask, e_nose_pos = get_nose_mask(e_parsing)

            e_image = np.array(e_image)
            e_image = replace_obj(
                origin_image=np.array(o_img),
                edit_image=edit_image,
                origin_mask=np.logical_not(o_nose_mask),
                edit_mask=np.logical_not(e_nose_mask),
                o_pos=o_nose_pos,
                e_pos=e_nose_pos,
                blur_foreground=True,
                delta=2
                )
            
            dest_path = os.path.join(dest_img_folder, img_edit_name)
            edit_image = Image.fromarray(edit_image)
            edit_image.save(dest_path)


def mouse_change(origin_folder, edit_folder, dest_folder="results"):
    net = load_model()

    for img_name in os.listdir(origin_folder):
        img_path = os.path.join(origin_folder, img_name)
        o_img = Image.open(img_path)
        o_parsing = compute_mask(o_img, net)
        o_mouse_mask, o_mouse_pos = get_background_mask(o_parsing, add_glasses=True, add_earings=True)

        img_edit_folder = os.path.join(edit_folder, img_name[:-4])
        dest_img_folder = os.path.join(dest_folder, img_name[:-4])
        for img_edit_name in ["Bp_min.png", "Bp_max.png"]:
            img_edit_path = os.path.join(img_edit_folder, img_edit_name)
            e_image = Image.open(img_edit_path)
            e_parsing = compute_mask(e_image, net)
            e_mouse_mask, e_mouse_pos = get_nose_mask(e_parsing)

            e_image = np.array(e_image)
            e_image = replace_obj(
                origin_image=np.array(o_img),
                edit_image=edit_image,
                origin_mask=np.logical_not(o_mouse_mask),
                edit_mask=np.logical_not(e_mouse_mask),
                o_pos=o_mouse_pos,
                e_pos=e_mouse_pos,
                blur_foreground=True,
                delta=2
                )
            
            dest_path = os.path.join(dest_img_folder, img_edit_name)
            edit_image = Image.fromarray(edit_image)
            edit_image.save(dest_path)

