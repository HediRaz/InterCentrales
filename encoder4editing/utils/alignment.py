"""Alignment functions."""

import dlib
import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage


def get_landmark(filepath, predictor):
    """Get landmark with dlib.

    Parameters
    ----------
    filepath : str
        Path to image file.
    predictor : dlib.shape_predictor
        Shape predictor.

    Returns
    -------
    landmarks : np.array, shape=(68, 2)
        Landmarks.
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    for det in dets:
        shape = predictor(img, det)

    parts = list(shape.parts())
    landmarks = []
    for part in parts:
        landmarks.append([part.x, part.y])
    landmarks = np.array(landmarks)
    return landmarks


def align_face(filepath, predictor):
    """Align face with dlib.

    Parameters
    ----------
    filepath : str
        Path to image file.
    predictor : dlib.shape_predictor
        Shape predictor.

    Returns
    -------
    img: PIL.Image
        Aligned image.
    """
    landmarks = get_landmark(filepath, predictor)

    # Calculate auxiliary vectors.
    eye_left = np.mean(landmarks[36: 42], axis=0)  # left-clockwise
    eye_right = np.mean(landmarks[42: 48], axis=0)  # left-clockwise
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = landmarks[48: 60][0]
    mouth_right = landmarks[48: 60][6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    center = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([center - x - y, center - x + y,
                     center + x + y, center + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(filepath)

    output_size = 256
    transform_size = 256
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0),
           max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img),
                     ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                     'reflect')
        height, weight, _ = img.shape
        y, x, _ = np.ogrid[:height, :weight, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                           np.float32(weight - 1 - x)
                                           / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1],
                                           np.float32(height - 1 - y)
                                           / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img
                ) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)),
                                  'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Return aligned image.
    return img
