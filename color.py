import json
import numpy as np

def generate_color_palette(n_classes):
    """
    Generate a consistent color palette for class visualization.
    """
    np.random.seed(42)  # For reproducibility
    palette = {i: np.random.randint(0, 255, 3).tolist() for i in range(n_classes)}
    palette[0] = [0, 0, 0]  # Background
    return palette

def load_color_palette(filename):
    """ Load the color palette from a JSON file. """
    with open(filename, 'r') as file:
        color_palette = json.load(file)
        color_palette = {int(k): tuple(v) for k, v in color_palette.items()}  # Ensure keys are integers and values are tuples
    return color_palette

def save_color_palette(color_palette, filename):
    """ Save the color palette to a JSON file. """
    with open(filename, 'w') as file:
        json.dump({k: list(v) for k, v in color_palette.items()}, file)