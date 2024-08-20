import os
import torch
import json
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Lambda, Resize, InterpolationMode, Grayscale
import torchvision.transforms.functional as TF
import cv2
import torch.nn.functional as F
from torch.utils.data import random_split, Subset
from collections import OrderedDict
from color import generate_color_palette, save_color_palette, load_color_palette

num_classes = 33

class UNET(nn.Module):

    def __init__(self, in_channels=1, classes=33):
        super(UNET, self).__init__()
        self.layers = [in_channels, 64, 128, 256, 512, 1024]

        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])

        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])

        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer, layer // 2) for layer in self.layers[::-1][:-2]])

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv

    def forward(self, x):
        concat_layers = []

        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)

        concat_layers = concat_layers[::-1]

        for up_trans, double_conv_up, concat_layer in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])

            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)

        x = self.final_conv(x)
        return F.softmax(x, dim=1)  # Use softmax to get probabilities

def classlabel(class_title):
    try:
        label = int(class_title)
        if 1 <= label <= 32:
            return label
        else:
            return 0
    except ValueError:
        return 0  

class TeethDataset(Dataset):
    def __init__(self, num_classes, root_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.num_classes = num_classes 
        self.transform = transform if transform is not None else Compose([ToTensor()])
        self.mask_transform = mask_transform if mask_transform is not None else Compose([ Resize((1024, 2048), interpolation=InterpolationMode.NEAREST)])
        self.image_paths = []
        self.json_paths = []

        images_dir = os.path.join(root_dir, 'img/')
        json_dir = os.path.join(root_dir, 'ann/')
        for filename in os.listdir(images_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(images_dir, filename)
                json_filename = os.path.splitext(filename)[0] + '.jpg.json'
                json_path = os.path.join(json_dir, json_filename)
                if os.path.exists(json_path):
                    self.image_paths.append(image_path)
                    self.json_paths.append(json_path)

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        json_path = self.json_paths[index]
        
        image = Image.open(image_path).convert('L')
        
        single_channel_mask = np.zeros((1024, 2048), dtype=np.float32)
        
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        for obj in json_data['objects']:
            class_title = obj['classTitle']
            class_label = classlabel(class_title)
            polygon = obj['points']['exterior']
            object_mask = Image.new('L', (2048, 1024), 0)
            draw = ImageDraw.Draw(object_mask)
            polygon_tuples = [tuple(point) for point in polygon]
            draw.polygon(polygon_tuples, outline=1, fill=1)
            object_mask = np.array(object_mask)

            single_channel_mask[object_mask == 1] = class_label

        mask = torch.from_numpy(single_channel_mask).long()
        mask = mask.unsqueeze(0)
       
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = mask.squeeze(0)

        return image, mask, os.path.basename(image_path)

def save_indices(indices, filename):
    with open(filename, 'w') as file:
        json.dump(indices, file)

def load_indices(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def get_teeth_data(root_dir, transform, batch_size, split_ratio=0.8, indices_path=None):
    dataset = TeethDataset(num_classes=num_classes, root_dir=root_dir, transform=transform)
    if indices_path and os.path.exists(indices_path):
        with open(indices_path, 'r') as file:
            indices_data = json.load(file)
        train_indices = indices_data['train']
        val_indices = indices_data['val']
    else:
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        if indices_path:
            save_indices({'train': train_indices, 'val': val_indices}, indices_path)

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader

def visualize_mask(mask, color_palette):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=0)
    elif mask.ndim != 3:
        raise ValueError("Mask must be either 2D (HxW) or 3D (CxHxW)")

    annotated_image = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)

    for class_index in range(mask.shape[0]):
        class_mask = mask[class_index]
        if class_index in color_palette:
            color = color_palette[class_index]
            annotated_image[class_mask > 0] = color
            
            coords = np.argwhere(class_mask)
            if coords.any():
                y, x = coords[len(coords) // 2]
                cv2.putText(annotated_image, str(class_index), (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()

def visualize_first_few_masks(data_loader, color_palette, num_images=2):
    count = 0
    for images, masks, _ in data_loader:
        if count >= num_images:
            break
        for i in range(images.size(0)):
            if count >= num_images:
                break
            visualize_mask(masks[i].cpu(), color_palette)
            count += 1    

def map_class_indices_to_rgb(mask, palette):
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in palette.items():
        rgb_image[mask == class_index] = color
    return rgb_image

def save_image(image_tensor, true_mask, pred_mask, color_palette, save_dir, image_name):
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.squeeze().cpu().numpy()
    else:
        image = image_tensor
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)

    true_mask_rgb = map_class_indices_to_rgb(true_mask.cpu().numpy(), color_palette)
    pred_mask_rgb = map_class_indices_to_rgb(pred_mask.cpu().numpy(), color_palette)

    # Add a singleton dimension to the grayscale image to match the RGB arrays
    image_rgb = np.stack((image,) * 3, axis=-1)

    combined_image = np.hstack((image_rgb, true_mask_rgb, pred_mask_rgb))

    save_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_result.png")
    cv2.imwrite(save_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))


def visualize_each_image(image_tensor, true_mask, pred_mask, color_palette):
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.squeeze().cpu().numpy()
    else:
        image = image_tensor
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    
    true_mask_rgb = map_class_indices_to_rgb(true_mask.cpu().numpy(), color_palette)
    pred_mask_rgb = map_class_indices_to_rgb(pred_mask.cpu().numpy(), color_palette)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(true_mask_rgb)
    axes[1].set_title('True Mask')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask_rgb)
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.show()

def evaluate_and_visualize_predictions(data_loader, model, device, num_classes, color_palette, save_dir, num_images=None, confidence_threshold=0.5):
    if num_images is None:
        num_images = len(data_loader.dataset)
    model.eval()
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'specificity': [],
        'iou': [],
        'dice': []
    }
    
    with torch.no_grad():
        count = 0
        for images, masks, image_names in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            
            for i in range(images.size(0)):
                if count >= num_images:
                    break

                output_probs = outputs[i].cpu()
                pred_mask = output_probs.argmax(0)  # Get the most probable class
                max_probs = output_probs.max(0)[0]  # Get the max probability for each pixel

                # Apply the confidence threshold
                pred_mask[max_probs < confidence_threshold] = 0

                true_mask = masks[i].cpu()

                # Print true labels and predicted labels for each image
                print(f"Image name: {image_names[i]}")
                print(f"True labels: {np.unique(true_mask.numpy())}")
                print(f"Predicted labels: {np.unique(pred_mask.numpy())}")

                metrics['accuracy'].append(accuracy(pred_mask, true_mask, num_classes))
                metrics['precision'].append(precision_multiclass(pred_mask, true_mask, num_classes))
                metrics['recall'].append(recall_multiclass(pred_mask, true_mask, num_classes))
                metrics['specificity'].append(specificity_multiclass(pred_mask, true_mask, num_classes))
                metrics['iou'].append(iou_multiclass(pred_mask, true_mask, num_classes))
                metrics['dice'].append(dice_score_multiclass(pred_mask, true_mask, num_classes))
                
                #visualize_each_image(images[i].cpu(), true_mask, pred_mask, color_palette)
                save_image(images[i].cpu(), true_mask, pred_mask, color_palette, save_dir, image_names[i])
                count += 1

    print(f"Evaluated on {count} images.")
    for metric, values in metrics.items():
        avg_metric = sum(values) / len(values) if values else 0
        print(f'Average {metric}: {avg_metric:.4f}')

def accuracy(pred_mask, true_mask, num_classes):
    correct = (pred_mask == true_mask).sum().item()
    total = true_mask.numel()
    return correct / total

def precision_multiclass(pred_mask, true_mask, num_classes):
    precisions = []
    for c in range(1, num_classes):
        pred_c = (pred_mask == c).float()
        true_c = (true_mask == c).float()
        tp = (pred_c * true_c).sum()
        fp = (pred_c * (1 - true_c)).sum()
        precision = tp / (tp + fp + 1e-10)
        precisions.append(precision.item())
    return sum(precisions) / len(precisions)

def recall_multiclass(pred_mask, true_mask, num_classes):
    recalls = []
    for c in range(1, num_classes):
        pred_c = (pred_mask == c).float()
        true_c = (true_mask == c).float()
        tp = (pred_c * true_c).sum()
        fn = ((1 - pred_c) * true_c).sum()
        recall = tp / (tp + fn + 1e-10)
        recalls.append(recall.item())
    return sum(recalls) / len(recalls)

def specificity_multiclass(pred_mask, true_mask, num_classes):
    specificities = []
    for c in range(1, num_classes):
        pred_c = (pred_mask == c).float()
        true_c = (true_mask == c).float()
        tn = ((1 - pred_c) * (1 - true_c)).sum()
        fp = (pred_c * (1 - true_c)).sum()
        specificity = tn / (tn + fp + 1e-10)
        specificities.append(specificity.item())
    return sum(specificities) / len(specificities)

def iou_multiclass(pred_mask, true_mask, num_classes):
    ious = []
    for c in range(1, num_classes):
        pred_c = (pred_mask == c).float()
        true_c = (true_mask == c).float()
        intersection = (pred_c * true_c).sum()
        union = (pred_c + true_c - pred_c * true_c).sum()
        iou = intersection / (union + 1e-10)
        ious.append(iou.item())
    return sum(ious) / len(ious)

def dice_score_multiclass(pred_mask, true_mask, num_classes):
    dices = []
    for c in range(1, num_classes):
        pred_c = (pred_mask == c).float()
        true_c = (true_mask == c).float()
        intersection = (pred_c * true_c).sum()
        dice = (2 * intersection) / (pred_c.sum() + true_c.sum() + 1e-10)
        dices.append(dice.item())
    return sum(dices) / len(dices)

def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = '/data2/Shaily/UNET_working/UNet-Multiclass-main/scripts/checkpoint3.pt'
    print(f"Loading model from: {MODEL_PATH}")
    ROOT_DIR = '/data2/Shaily/UNET_working/UNet-Multiclass-main/datasets/ODON'
    indices_path = '/data2/Shaily/UNET_working/UNet-Multiclass-main/scripts/model_indices/indices.json'
    save_images = True
    save_dir = '/data2/Shaily/UNET_working/UNet-Multiclass-main/scripts/result_images_checkpt3new'

    image_transforms = Compose([
        Grayscale(num_output_channels=1),
        Resize((1024, 2048)),
        ToTensor(),
    ])

    mask_transform = Compose([
        Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), size=(1024, 2048), mode='nearest').squeeze(0))
    ])

    color_palette_path = '/data2/Shaily/UNET_working/UNet-Multiclass-main/scripts/color_palette.json'
    if not os.path.exists(color_palette_path):
        color_palette = generate_color_palette(num_classes)
        save_color_palette(color_palette, color_palette_path)
    else:
        color_palette = load_color_palette(color_palette_path)

    train_loader, val_loader = get_teeth_data(
        root_dir=ROOT_DIR,
        transform=image_transforms,
        batch_size=2,
        indices_path=indices_path
    )

    model = UNET(in_channels=1, classes=num_classes).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    evaluate_and_visualize_predictions(val_loader, model, device, num_classes, color_palette, save_dir, confidence_threshold=0.5)

    print(f"Processed {len(val_loader.dataset)} images.")

if __name__ == '__main__':
    main()
