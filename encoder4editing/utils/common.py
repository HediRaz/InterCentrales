"""Common utilities for image manipulation."""

import matplotlib.pyplot as plt
from PIL import Image


def log_input_image(x, opts):
    """Log images."""
    return tensor2im(x)


def tensor2im(var):
    """Convert a tensor to an image."""
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var+1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
    """Visualize faces."""
    display_count = len(log_hooks)
    fig = plt.figure(figsize=(8, 4 * display_count))
    gridspec = fig.add_gridspec(display_count, 3)
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        fig.add_subplot(gridspec[i, 0])
        if 'diff_input' in hooks_dict:
            vis_faces_with_id(hooks_dict, fig, gridspec, i)
        else:
            vis_faces_no_id(hooks_dict, fig, gridspec, i)
    plt.tight_layout()
    return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
    """Visualize faces with id."""
    plt.imshow(hooks_dict['input_face'])
    diff_input = float(hooks_dict['diff_input'])
    diff_views = float(hooks_dict['diff_views'])
    diff_target = float(hooks_dict['diff_target'])
    plt.title(f'Input\nOut Sim={diff_input:.2f}')
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title(f'Target\nIn={diff_views:.2f}, Out={diff_target:.2f}')
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title(f'Output\n Target Sim={diff_target:.2f}')


def vis_faces_no_id(hooks_dict, fig, gs, i):
    """Visualize faces without id."""
    plt.imshow(hooks_dict['input_face'], cmap="gray")
    plt.title('Input')
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target')
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output')
