# Gan Face Editing -- 2nd place

**DISCLAIMER : This is a fork from [InterCentrale](https://github.com/HediRaz/InterCentrales) which is the work of the second team of Ceteris Paribus Face Data Challenge. The goal of this fork is to make it be merged on [GAN-Face-Editing](https://github.com/valentingol/gan-face-editing) for the branch `face_challenge`, consisting on the best submission of the challenge.**

This repository and the repositories it contains are licensed under the [MIT license](LICENSE.md).

---

This is the work of the team ArgoCS composed by Thibault Le Sellier De Chezelles and Hédi Razgallah for the Ceteris Paribus Challenge.

## Setup

Run

```script
sh setup.sh
pip install -r requirements.txt
```

## Run the app

### User interface

Two user interface are available:

- The first explore the latent space of e4e using the first PCA vectors given when encoding FFHQ dataset
- The second allows to load custom vectors firstly saved in numpy format

To run the first :

```bash
python main_ui.py --pca
```

To run the second :

```bash
python main_ui.py --custom
```

## Reencoding images

Our second method to edit images was to edit the original images and then reencode them using e4e.

To visualize face parsing:

```bash
python main_reencode.py --img_path /path/to/img --visualize_parsing
```

A list of editing is implemented:

- hair color
- bags under eyes
- pointy nose
- chubby

To see how are the results, each transformation is optional:

```bash
python main_reencode.py --img_path /path/to/img
    --hair_color (optional) [blond, brown, black, gray]
    --hair_color_brut (optional) [blond, brown, black, gray]
    --bag_under_eyes (optional) [min, max]
    --pointy_nose (optional) [min, max]
    --chubby (optional)
```

Play yourself with that method editing an image and reencode it running the script without any transfomation:

```bash
python main_reencode.py --img_path /path/to/img
```

## References

- e4e:  [https://github.com/omertov/encoder4editing]
- restyle encoder: [https://github.com/yuval-alaluf/restyle-encoder]
- face parsing: [https://github.com/zllrunning/face-parsing.PyTorch]
