from model import BiSeNet

import torch
from torch.nn.functional import max_pool2d, avg_pool2d

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def load_model(cp='79999_iter.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('face_parsing/res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    return net


def vis_parsing_maps(im, parsing_anno):
    """ Visualize parsing annotation on the image """
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
    index = np.where(vis_parsing_anno == 0)
    vis_parsing_anno_color[index[0], index[1], :] = 0

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    # vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    vis_im = 0.4*vis_im + 0.6*vis_parsing_anno_color
    vis_im = np.uint8(vis_im)

    plt.imshow(vis_im)
    plt.show()
    # return vis_im


def compute_mask(img, net):
    """
    Compute the face mask of the image

    Args:
    img: PIL image
    net: parsing model

    Returns:
    mask: numpy coords = np.stack(np.where(mask_eye), axis=-1)
    dx = np.max(coords[:, 1]) - np.min(coords[:, 1])
    dy = np.max(coords[:, 0]) - np.min(coords[:, 0])
    coords[:, 0] += dy
    if not max:
        color = np.clip(np.mean(img[coords[:, 0], coords[:, 1], :], axis=0) + 60, 0, 255)
        print(color)
    for x in range(-1, 2):
        for y in range(0, 3):
            tmp_coords = np.copy(coords)
            tmp_coords[:, 1] += x * dx // 2
            tmp_coords[:, 0] += y * dy // 2
     array of the mask
    """

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)

        # vis_parsing_maps(image, parsing)
    return parsing


def get_background_mask(parsing, add_clothes=False, add_glasses=False, add_earings=False):
    """ Get the background mask of the image """
    mask = np.logical_or(parsing == 0, parsing == 18)
    if add_clothes:
        mask = np.logical_or(mask, parsing == 16)
    if add_glasses:
        mask = np.logical_or(mask, parsing == 6)
    if add_earings:
        mask = np.logical_or(mask, parsing == 9)
    indexes = np.where(mask)
    g_i = int(np.mean(indexes[0]))
    g_j = int(np.mean(indexes[1]))
    return mask, (g_i, g_j)


def get_nose_mask(parsing):
    """ Get the nose mask of the image """
    mask = parsing == 10
    return mask


def get_left_eye_mask(parsing):
    """ Get the left eye mask of the image """
    mask = parsing == 4
    return mask


def get_right_eye_mask(parsing):
    """ Get the right eye mask of the image """
    mask = parsing == 5
    return mask


def get_mouth_mask(parsing):
    """ Get the mouth mask of the image """
    mask = np.logical_or(parsing == 11, parsing == 12)
    mask = np.logical_or(mask, parsing == 13)
    # indexes = np.where(mask)
    # g_i = int(np.mean(indexes[0]))
    # g_j = int(np.mean(indexes[1]))
    return mask


def get_hair_mask(parsing):
    """ Get the hair mask of the image """
    mask = parsing == 17
    return mask

def delete_foreground(img, foreground_mask, delta=5):
    """ Delete the foreground of the image

    Args:
    img: PIL image
    foreground_mask: numpy array of the foreground mask to delete
    delta: the distance to the foreground to delete

    Returns:
    img: PIL image

    """
    f_indexes = np.where(foreground_mask)
    img[f_indexes[0], f_indexes[1], :] = 0

    for i in np.unique(f_indexes[0]):
        arg_i = f_indexes[0] == i
        indexes_j = f_indexes[1][arg_i]
        j_max = max(indexes_j)
        j_min = min(indexes_j)
        j_max = min(511, j_max+delta)
        j_min = max(0, j_min-delta)
        # img[i, indexes_j, :] = img[i, j_min] + np.tile(np.linspace(0, 1, indexes_j.shape[0]), (3, 1)).T * (img[i, j_max] - img[i, j_min])
        img[i, indexes_j[indexes_j < (j_min + j_max) / 2], :] = img[i, j_min]
        img[i, indexes_j[indexes_j >= (j_min + j_max) / 2], :] = img[i, j_max]

    img = np.where(np.stack((foreground_mask, foreground_mask, foreground_mask), -1), cv2.medianBlur(img, 5), img)
    # for j in np.unique(f_indexes[1]):
    #     arg_j = f_indexes[1] == j
    #     indexes_i = f_indexes[0][arg_j]
    #     i_max = max(indexes_i)
    #     i_min = min(indexes_i)
    #     i_max = min(511, i_max+5)
    #     i_min = max(0, i_min-5)
    #     img[indexes_i, j, :] = (img[indexes_i, j, :] + img[i_min, j] + np.tile(np.linspace(0, 1, indexes_i.shape[0]), (3, 1)).T * (img[i_max, j] - img[i_min, j])) / 2
    
    return img


def replace_obj(origin_image, edit_image, origin_mask, edit_mask, o_pos, e_pos, background_mask=None, delta=5, blur_foreground=False):
    """ Replace the object in the image """
    if blur_foreground:
        origin_image = delete_foreground(origin_image, origin_mask, delta=delta)

    e_indexes = np.where(edit_mask)
    print(e_indexes[1].shape)
    o_indexes  = [
        e_indexes[0] + (o_pos[0] - e_pos[0]),
        e_indexes[1] + (o_pos[1] - e_pos[1]),
    ]
    print(o_indexes[0].shape, o_indexes[1].shape)
    o_indexes[0] = np.clip(o_indexes[0], 0, 511)
    o_indexes[1] = np.clip(o_indexes[1], 0, 511)
    # print(o_indexes[0].shape, o_indexes[1].shape)
    # o_indexes[1] = np.minimum(o_indexes[1], 511)
    # print(o_indexes[0].shape, o_indexes[1].shape)
    # o_indexes[1] = np.maximum(o_indexes[1], 0)
    # print(o_indexes[0].shape, o_indexes[1].shape)
    print(origin_image[o_indexes[0], o_indexes[1], :].shape)
    print(edit_image[e_indexes[0], e_indexes[1], :].shape)
    print(origin_image.shape, edit_image.shape)
    if background_mask is not None:
        origin_bg = origin_image[background_mask].copy()
        origin_image[o_indexes[0], o_indexes[1], :] = edit_image[e_indexes[0], e_indexes[1], :]
        origin_image[background_mask] = origin_bg
    else:
        origin_image[o_indexes[0], o_indexes[1], :] = edit_image[e_indexes[0], e_indexes[1], :]

    return origin_image


def change_hair_color_brut(img, parsing, color="blond"):
    """ Change the hair color of the image replacing original hair with new hair """
    hair_mask = get_hair_mask(parsing)

    if color == "blond":
        color_a = np.array(Image.open("face_parsing/makeup/blond_hair.png"))
        color_a = color_a[:512, :512]
    if color == "brown":
        color_a = np.array(Image.open("face_parsing/makeup/brown_hair.png"))
        color_a = color_a[:512, :512]
    if color == "black":
        color_a = np.array(Image.open("face_parsing/makeup/black_hair.png"))
        color_a = color_a[:512, :512]
    if color == "gray":
        color_a = np.array(Image.open("face_parsing/makeup/gray_hair.png"))
        color_a = color_a[:512, :512]

    edit_img = img.copy()
    edit_img[hair_mask] = color_a[hair_mask]
    return edit_img


def change_hair_color_smooth(img, parsing, color="blond"):
    """ Change the hair color of the image replacing original hair with new hair """
    hair_mask = get_hair_mask(parsing)

    edit_img = img.copy()
    if color == "blond":
        color_a = np.array([245, 232, 39])
        edit_img[hair_mask] = 0.3*color_a + 0.7*edit_img[hair_mask]
    if color == "brown":
        color_a = np.array([88, 51, 3])
        edit_img[hair_mask] = 0.6*color_a + 0.4*edit_img[hair_mask]
    if color == "black":
        color_a = np.array([0, 0, 0])
        edit_img[hair_mask] = 0.8*color_a + 0.2*edit_img[hair_mask]
    if color == "gray":
        color_a = np.array([200, 200, 200])
        edit_img[hair_mask] = 0.6*color_a + 0.4*edit_img[hair_mask]

    return edit_img


def make_shape_under_eye(img, mask_eye, color:list, max=True):
    """
    Makes a large shape under mask_eye, of the color 'color'
    max=True means we keep the color, else we average the color under the mask and add 60 to lighten it further
    """
    coords = np.stack(np.where(mask_eye), axis=-1)
    dx = np.max(coords[:, 1]) - np.min(coords[:, 1])
    dy = np.max(coords[:, 0]) - np.min(coords[:, 0])
    coords[:, 0] += dy
    if not max:
        color = np.clip(np.mean(img[coords[:, 0], coords[:, 1], :], axis=0) + 60, 0, 255)
        print(color)
    for x in range(-1, 2):
        for y in range(0, 3):
            tmp_coords = np.copy(coords)
            tmp_coords[:, 1] += x * dx // 2
            tmp_coords[:, 0] += y * dy // 2
            img[tmp_coords[:, 0], tmp_coords[:, 1], :] = np.array(color)
    return img


def make_bags(img : np.array, parsing, max=True, occident=True):
    """
    Adds drawings for bags under eyes feature under the eye.
    max=True means we darken the eye, lighten if False
    occident=True means clear skin color, False means dark
    """
    if occident:
        color = [120, 81, 100]
    else:
        color = [60, 40, 50]
    img = make_shape_under_eye(img, get_left_eye_mask(parsing), color, max=max)
    img = make_shape_under_eye(img, get_right_eye_mask(parsing), color, max=max)
    return img


def replace(o_img, e_img, mask):
    """
    Puts the pixels in e_img described by mask into o_img
    """
    indexes = np.where(mask)
    img = np.copy(o_img)
    img[indexes[0], indexes[1], :] = e_img[indexes[0], indexes[1], :]
    return img


def replace_background(o_img, e_img, net):
    """
    Computes the background mask of o_img, replaces the foreground by averaging
    Then takes e_img's foreground and puts it into o_img
    """
    o_parsing = np.array(compute_mask(o_img, net))
    e_parsing = np.array(compute_mask(e_img, net))

    o_truth = [o_parsing == i for i in list(range(1, 16)) + [17]]
    o_mask = np.zeros_like(o_truth[0], dtype=bool)
    for i in o_truth:
        o_mask += i

    img = delete_foreground(np.array(o_img), o_mask)

    e_truth = [e_parsing == i for i in list(range(1, 16)) + [17]]
    e_mask = np.zeros_like(e_truth[0], dtype=bool)
    for i in e_truth:
        e_mask += i

    img = replace(img, np.array(e_img), e_mask)

    return img


def make_pointy_nose(img, parsing):
    """
    Adds a nose colored bar on the nose, that's 20% longer than the nose and half of its width
    """
    mask = get_nose_mask(parsing)

    indexes = [i for i in np.where(mask)]
    color = np.clip(np.mean(img[indexes[0], indexes[1], :], axis=0) - 0, 0, 255)
    x_min = np.min(indexes[1])
    x_max = np.max(indexes[1])
    y_min = np.min(indexes[0])
    y_max = np.max(indexes[0])
    dx = x_max - x_min
    dy = y_max - y_min

    x_bounds = [x_min + dx // 4, x_max - dx // 4]

    cond = np.logical_and(indexes[1]>x_bounds[0], indexes[1]<x_bounds[1])

    indexes[0] = indexes[0][cond]
    indexes[1] = indexes[1][cond]

    img[indexes[0], indexes[1], :] = color

    indexes[0] +=  dy // 5
    img[indexes[0], indexes[1], :] = color

    return img


def make_flat_nose(img, parsing):
    """
    Adds a nose colored bar on the nose, that's 20% longer than the nose and half of its width
    """
    mask = get_nose_mask(parsing)
    img = make_pointy_nose(img, parsing)
    indexes = [i for i in np.where(mask)]
    color = np.array([0, 0, 0])
    x_min = np.min(indexes[1])
    x_max = np.max(indexes[1])
    y_min = np.min(indexes[0])
    y_max = np.max(indexes[0])
    dx = x_max - x_min
    dy = y_max - y_min

    y_bounds = [y_min + 3*dy // 4, y_max]

    cond = np.logical_and(indexes[0]>y_bounds[0], indexes[0]<y_bounds[1])

    indexes[0] = indexes[0][cond]
    indexes[1] = indexes[1][cond]

    indexes[0] +=  dy // 10

    img[indexes[0], indexes[1], :] = color

    indexes[1] +=  dx // 5
    img[indexes[0], indexes[1], :] = color

    indexes[1] -=  2 * dx // 5
    img[indexes[0], indexes[1], :] = color


    return img


def make_balls_around_mouth(img, parsing):
    mask_mouth = get_mouth_mask(parsing)

    indexes = np.where(mask_mouth)
    x_min = np.min(indexes[1])
    x_max = np.max(indexes[1])
    y_min = np.min(indexes[0])
    y_max = np.max(indexes[0])
    dx = int((x_max - x_min) * 1.3)
    dy = y_max - y_min
    
    x_left = x_min - dx // 2
    x_right = x_max + dx // 2

    square_indexes = [np.concatenate([np.arange(0, dx) for _ in range(dx)], axis=0), np.concatenate([np.ones(dx) * i for i in range(dx)], axis=0)]

    # coords1 = [square_indexes[0] , (square_indexes[1]+y_min + dx // 2).astype(int)]
    # coords2 = [square_indexes[0], (square_indexes[1]+y_min + dx // 2).astype(int)]
    coords1 = [square_indexes[0]+y_min - dx // 2, (square_indexes[1] + x_min - dx).astype(int)]
    coords2 = [square_indexes[0]+y_min - dx // 2, (square_indexes[1] + x_min + dx).astype(int)]

    print(coords1, coords2)

    color = (np.clip(np.mean(img[coords1[0], coords1[1], :], axis=0), 0, 255) + np.clip(np.mean(img[coords2[0], coords2[1], :], axis=0), 0, 255))/2 + 25

    img[coords1[0], coords1[1], :] = color
    img[coords2[0], coords2[1], :] = color

    return img


if __name__ == "__main__":
    from PIL import image
    net = load_model()
    o_path = "../Dataset/2_1_0_1_0_0_4_1_2.png"
    o_img = Image.open(o_path)
    e_path = "../Dataset/1_0_0_0_1_0_3_0_0.png"
    e_img = Image.open(e_path)

    parsing = compute_mask(o_img, net)
    mask, _ = get_mouth_mask(parsing)
    img = make_balls_around_mouth(np.array(o_img), mask)

    img = Image.fromarray(img)
    img.save("makeup/test_fonts.png")

    print("Finished")
