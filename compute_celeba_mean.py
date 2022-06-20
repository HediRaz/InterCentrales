"""Script to get mean latent vectors from Celeba dataset."""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

dataframe = pd.read_csv("list_attr_celeba.csv", header=0)
dataframe["idx"] = pd.to_numeric(dataframe["idx"])
KEY = "idx"
dataframe.set_index(KEY, inplace=True)
ATTRIBUTES = dataframe.columns[1:]


def get_att_df(dataframe, att_name, yes=True):
    """Select rows from dataframe where att_name == yes."""
    if yes:
        res = dataframe[dataframe[att_name]]
    else:
        res = dataframe[dataframe[att_name] is False]
    return res


def load_latents(filename, latents_folder="latents_celeba"):
    """Load latent vectors from file."""
    latents_path = os.path.join(latents_folder, filename[:-3] + "npy")
    return np.load(latents_path)


def get_latents_mean(
        dataframe, latents_folder="latents_celeba", return_count=False,
        **kwargs
        ):
    """Compute mean of encodings of images with specific attributes.

    Parameters
    ----------
    dataframe : pd.dataframe
        Dataframe with attributes.
    latents_folder : str, optional
        Folder where latents are stored. By default "latents_celeba".
    return_count : bool, optional
        If True, return number of images with specific attributes.
        By default False.
    kwargs : dict, optional
        Dictionary with attributes and their values.

    Returns
    -------
    mean : np.ndarray
        Mean of encodings of images with specific attributes.
    If return_count is True:
    count : int
        Number of images with specific attributes
    """
    for att in ATTRIBUTES:
        if att in kwargs:
            dataframe = get_att_df(dataframe, att, kwargs[att])

    mean = np.zeros((18, 512), dtype=np.float32)
    for file in dataframe["filename"]:
        mean += (
                load_latents(file, latents_folder=latents_folder)
                / len(dataframe["filename"])
                )
        print("""
coucou
              """)
    if return_count:
        return mean, len(dataframe["filename"])
    return mean


def get_latents_sum(dataframe, latents_folder="latents_celeba", **kwargs):
    """Compute sum of encodings of images with specific attributes.

    Parameters
    ----------
    dataframe : pd.dataframe
        Dataframe with attributes.
    latents_folder : str, optional
        Folder where latents are stored. By default "latents_celeba".
    kwargs : dict, optional
        Dictionary with attributes and their values.

    Returns
    -------
    sum_enc : np.ndarray
        Sum of encodings of images with specific attributes.
    count : int
        Number of images with specific attributes.
    """
    for att in ATTRIBUTES:
        if att in kwargs:
            dataframe = get_att_df(dataframe, att, kwargs[att])

    for i, file in enumerate(dataframe["filename"]):
        if i == 0:
            sum_enc = load_latents(file, latents_folder=latents_folder)
        else:
            sum_enc += load_latents(file, latents_folder=latents_folder)
    return sum_enc, len(dataframe["filename"])


def all_change():
    """Get all changes.

    Compute difference between mean of all attributes and mean of all
    attributes with one attribute changed.
    """
    dest_folder = "vectors_editing/all_change"
    for att in tqdm(ATTRIBUTES):
        l_1 = get_latents_mean(dataframe, **{att: False})
        l_2 = get_latents_mean(dataframe, **{att: True})
        delta = l_2 - l_1
        np.save(os.path.join(dest_folder, att + ".npy"), delta)


if __name__ == "__main__":
    for att in ATTRIBUTES:
        dataframe[att] = dataframe[att] == 1

    TOTAL_IMG = len(dataframe)
    print(f"There is {TOTAL_IMG} in the dataset")

    DEST_FOLDER = "latents_group/all_change"
    compet_attributes = [
            "Pale_Skin", "Young", "Male", "Bangs", "Black_Hair", "Blond_Hair",
            "Brown_Hair", "Gray_Hair", "Double_Chin", "Straight_Hair",
            "Wavy_Hair", "Bald", "Bags_Under_Eyes", "Chubby"
            ]

    for att in tqdm(ATTRIBUTES[6:]):
        if att in compet_attributes:
            continue
        l1 = get_latents_mean(dataframe, **{att: False})
        l2 = get_latents_mean(dataframe, **{att: True})
        delta = l2 - l1
        np.save(os.path.join(DEST_FOLDER, att + ".npy"), delta)
