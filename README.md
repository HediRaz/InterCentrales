This is the work of the team ArgoCS composed by Thibault Le Sellier De Chezelles and Hédi Razgallah for the Ceteris Paribus Challenge.

# Setup
Make sure to install pretrained models needed to inference.

## With docker
A tested dockerfile is provided.
Just run:
```bash
docker build -t image_name
xhost +
docker run --rm -it --init --gpus=all --ipc=host -e DISPLAY=$DISPLAY -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -w /workspace
```

## Otherwise
Just install pytorch and tkinter



# Run the app
## User interface
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

A list of editing is implemented:
- chubby
- narrow eyes
- ...

Play yourself with that method editing an image and reencode it running:
```bash
python main_reencode.py --reencode /path/to/img
```
