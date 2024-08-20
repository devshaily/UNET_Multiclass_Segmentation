import os
import torch
import json
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose , Lambda, Resize, InterpolationMode ,  Grayscale
import torchvision.transforms.functional as TF
from torch.utils.data import random_split
from torch.utils.data import Subset



# Training configurations
#MODEL_DIR = '/data2/Shaily/UNET_working/UNet-Multiclass-main/models/'  # Directory for saving models
#MODEL_FILENAME = 'unet_checkpoint.pth'  # Filename for the model checkpoint
#MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)  # Full path including filename
OUTPUT_DIR = '/data2/Shaily/UNET_working/UNet-Multiclass-main/output_images_new'
#ROOT_DIR = '/data2/Shaily/Models/UNet-Multiclass-main/datasets/teeth/'
ROOT_DIR = '/data2/Shaily/UNET_working/UNet-Multiclass-main/datasets/ODON'
indices_file_path = '/data2/Shaily/UNET_working/UNet-Multiclass-main/scripts/model_indices/indices.json'
IMG_HEIGHT = 1024
IMG_WIDTH = 2048
BATCH_SIZE = 2
LEARNING_RATE = 0.0001
EPOCHS = 100  # Update to desired number of training epochs
num_classes = 33

image_transforms = Compose([
    Grayscale(num_output_channels=1),  # Converts to grayscale
    Resize((1024, 2048)),              # Resizes the image
    ToTensor(),                        # Converts the image to a tensor
])

# Define a mask transform function separately
mask_transform = Compose([
    Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), size=(1024, 2048), mode='nearest').squeeze(0))
])


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
        [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])
        
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
        # down layers
        concat_layers = []
        
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)
        
        concat_layers = concat_layers[::-1]
        
        # up layers
        for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])
            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)
            
        x = self.final_conv(x)
        #print(x.shape)
        
        return x 

def classlabel(class_title):
    # Assuming class_title comes as "32", "31", etc., and you want to map them directly to integers
    try:
        label = int(class_title)
        if 1 <= label <= 32:
            return label
        else:
            return 0  # Assuming 0 is your background class
    except ValueError:
        return 0 
        
class TeethDataset(Dataset):
    def __init__(self, num_classes, root_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.num_classes = num_classes 
        self.transform = transform if transform is not None else Compose([ToTensor()])
        self.mask_transform = mask_transform if mask_transform is not None else Compose([ToTensor(), Resize((1024, 2048), interpolation=InterpolationMode.NEAREST)])
        self.image_paths = []
        self.json_paths = []

        # Load image and JSON paths
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
        print(f"Total images loaded: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        json_path = self.json_paths[index]
        #print(f"Loading image: {self.image_paths[index]}")
        
        image = Image.open(image_path).convert('L')  # Ensuring grayscale
        #print(f"Original Image Size: {image.size}")
        
        # Initialize a multi-channel mask with zeros
        mask = np.zeros((self.num_classes, 1024, 2048), dtype=np.float32)
        
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        for obj in json_data['objects']:
            class_title = obj['classTitle']
            class_label = classlabel(class_title)
            
            polygon = obj['points']['exterior']
            object_mask = Image.new('L', (2048, 1024), 0)  # Adjusted to match desired output size directly
            draw = ImageDraw.Draw(object_mask)
            polygon_tuples = [tuple(point) for point in polygon]
            draw.polygon(polygon_tuples, outline=1, fill=1)
            object_mask = np.array(object_mask)
            
            mask[class_label] = np.logical_or(mask[class_label], object_mask)

            # Temporary visualization for debugging, consider saving instead of showing

            # if index < 5:  # Visualize for the first 5 images only
            #         plt.imshow(object_mask, cmap='gray')
            #         plt.title(f"Image Index: {index}, Class Label: {class_label}")
         
        mask = torch.from_numpy(mask)  # Convert numpy array to torch tensor
        
        if self.transform:
            image = self.transform(image)
            #print(f"Transformed Image Shape: {image.shape}")
        if self.mask_transform:
            mask = self.mask_transform(mask)
            #print(f"Transformed Mask Shape: {mask.shape}")
        return image, mask


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint3.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose  # Add this line
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_loss_min = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:  # Use verbose flag here
                    self.trace_func("Early stopping")
        else:
             if val_loss < self.val_loss_min:
                self.best_loss = val_loss
                self.best_loss_min = val_loss
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:  # Use verbose flag here
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)

def save_indices(indices, filename):
    with open(filename, 'w') as file:
        json.dump(indices, file)

def load_indices(filename):
    with open(filename, 'r') as file:
        return json.load(file)
        
def get_teeth_data(split_ratio=0.8, save_indices_path=None ):
    dataset = TeethDataset(num_classes=33, root_dir=ROOT_DIR, transform=image_transforms, mask_transform=mask_transform)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    if save_indices_path:
        # Save both sets of indices
        indices_data = {'train': train_dataset.indices, 'val': val_dataset.indices}
        save_indices(indices_data, save_indices_path)
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    return train_loader, val_loader


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    early_stopping = EarlyStopping(patience=5, verbose=True, path='checkpoint3.pt')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Convert masks to class indices if they are currently one-hot encoded
            if masks.dtype == torch.float32:
                masks = masks.max(dim=1)[1].long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # Debugging prints
            #print(f"Outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
            #print(f"Masks shape: {masks.shape}, dtype: {masks.dtype}")

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                # Convert masks to class indices if they are currently one-hot encoded
                if masks.dtype == torch.float32:
                    masks = masks.max(dim=1)[1].long()

                outputs = model(images)
                loss = criterion(outputs, masks)
                total_val_loss += loss.item()

                # Debugging prints
                #print(f"Validation Outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
                #print(f"Validation Masks shape: {masks.shape}, dtype: {masks.dtype}")

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        scheduler.step()

    return train_losses, val_losses


# Main execution with plotting
if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    train_loader, val_loader = get_teeth_data(split_ratio=0.8, save_indices_path=indices_file_path)
    print(f"Total training batches: {len(train_loader)}")
    print(f"Total validation batches: {len(val_loader)}")
    # for images, masks in train_loader:
    #     print(f"Batch size: {images.size(0)}") # Adjust this based on your dataset specifics
        
    model = UNET(in_channels=1, classes=33).to(DEVICE)
    model = torch.nn.DataParallel(model, device_ids=[0, 3])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses, val_losses = train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, DEVICE)

    # Plotting
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


