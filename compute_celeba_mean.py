# Perform operations over celaba encodings
import os
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm


df = pd.read_csv("list_attr_celeba.csv", header=0)
df["idx"] = pd.to_numeric(df["idx"])
KEY = "idx"
df.set_index(KEY, inplace=True)
ATTRIBUTES = df.columns[1:]
print(ATTRIBUTES)
for att in ATTRIBUTES:
    df[att] = df[att] == 1

TOTAL_IMG = len(df)
print(f"There is {TOTAL_IMG} in the dataset")


def get_att_df(df, att_name, yes=True):
    """ Select rows from df where att_name == yes """
    if yes:
        res = df[df[att_name]]
    else:
        res = df[df[att_name] == False]
    return res


def load_latents(filename, latents_folder="latents_celeba"):
    latents_path = os.path.join(latents_folder, filename[:-3]+"npy")
    return np.load(latents_path)


def get_latents_mean(df, latents_folder="latents_celeba", return_count=False, **kwargs):
    """ Compute mean of encodings of images with specific attributes
    Args:
    df: dataframe with attributes
    latents_folder: folder where latents are stored
    return_count: if True, return number of images with specific attributes
    kwargs: dictionary with attributes and their values

    Returns:
    mean: mean of encodings of images with specific attributes
    count (if return_count is True): number of images with specific attributes
    """
    for att in ATTRIBUTES:
        if att in kwargs:
            df = get_att_df(df, att, kwargs[att])

    m = np.zeros((18, 512), dtype=np.float32)
    for f in df["filename"]:
        m += load_latents(f, latents_folder=latents_folder) / len(df["filename"])
    if return_count:
        return m, len(df["filename"])
    else:
        return m


def get_latents_sum(df, latents_folder="latents_celeba", **kwargs):
    """ Compute sum of encodings of images with specific attributes 
    Args:
    df: dataframe with attributes
    latents_folder: folder where latents are stored
    kwargs: dictionary with attributes and their values

    Returns:
    sum: sum of encodings of images with specific attributes
    count: number of images with specific attributes
    """
    for att in ATTRIBUTES:
        if att in kwargs:
            df = get_att_df(df, att, kwargs[att])

    for i, f in enumerate(df["filename"]):
        if i == 0:
            m = load_latents(f, latents_folder=latents_folder)
        else:
            m += load_latents(f, latents_folder=latents_folder)
    return m, len(df["filename"])


def all_change():
    """ Compute dirrefence between mean of all attributes and mean of all attributes with one attribute changed """
    dest_folder = "vectors_editing/all_change"
    for att in tqdm(ATTRIBUTES):
        l1 = get_latents_mean(df, **{att: False})
        l2 = get_latents_mean(df, **{att: True})
        delta = l2 - l1
        np.save(os.path.join(dest_folder, att+".npy"), delta)


if __name__ == "__main__":
    dest_folder = "latents_group/all_change"
    compet_attributes = ["Pale_Skin", "Young", "Male", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Double_Chin", "Straight_Hair", "Wavy_Hair", "Bald", "Bags_Under_Eyes", "Chubby"]
    for att in tqdm(ATTRIBUTES[6:]):
        if att in compet_attributes:
            continue
        l1 = get_latents_mean(df, **{att: False})
        l2 = get_latents_mean(df, **{att: True})
        delta = l2 - l1
        np.save(os.path.join(dest_folder, att+".npy"), delta)

    """
    related_attributes = []
    att = "Smiling"

    delta = np.zeros((18, 512), dtype=np.float32)
    cpt = 0
    yes_list = [[True, False]] * (len(related_attributes)-1)
    for choices in tqdm(itertools.product(*yes_list)):
        args = dict()
        for i, c in enumerate(choices):
            args[related_attributes[i]] = c
        args1 = args.copy()
        args2 = args.copy()
        args1[att] = False
        args2[att] = True

        l1, n1 = get_latents_mean(df, return_count=True, **args1)
        l2, n2 = get_latents_mean(df, return_count=True, **args2)
        cpt += n1 + n2
        delta += (l2 - l1) * (n1 + n2)
    delta /= cpt
    np.save(os.path.join(dest_folder, att+".npy"), delta)
    """
