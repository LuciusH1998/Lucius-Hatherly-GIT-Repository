"""
This is the main script for training and evaluating the Unet model on the segmentation of retina blood vessels.

Input:
    - dataset: path to the dataset (the Data folder containing train and test folders)
    - logdir: str, directory to save logs
    - batch_size: int, batch size
    - num_epochs: int, number of epochs
    - lr: float, learning rate
    - weight_decay: float, weight decay
    - print_every: int, print every

Author:
    Credits to the work done previously by Johan Obando and Jerry Huang
    Pierre-Louis Benveniste
"""
import argparse
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from unet import UNet, UNetNoSkip
from utils import DiceCELoss, DiceLoss
from glob import glob
import cv2
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp


# seed experiment
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True


class GetDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        # Defining self.transform as none 
        self.transform = transform

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        ## If self.transform is not none, call transform on image and mask 
        if self.transform is not None:
            image, mask = self.transform(image, mask)
            return image, mask
        

        return image, mask

    def __len__(self):
        return self.n_samples


def parse_args():
    """
    This function parses the command line arguments.
    """
    parser = argparse.ArgumentParser(description='Retina blood vessel segmentation training script')
    parser.add_argument('--dataset', type=str, default=None, help='Path to the dataset (the Data folder containing train and test folders)', required=True)
    parser.add_argument('--logdir', type=str, default=None, help='Directory to save logs', required=True)
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: %(default)s).')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for training (default: %(default)s).')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: %(default)s).')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: %(default)s).')
    parser.add_argument('--print_every', type=int, default=80, help='Number of minibatches after which we print the loss (default: %(default)s).')
    return parser.parse_args()


def dice_score(preds, targets):
    # Compute the dice score using the DiceLoss class
    # Defining predictions
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    # Defining smooth, intersection, union, dice and dice.mean()
    smooth = 1e-6
    intersec = (preds*targets).sum(dim=(1,2, 3))
    union = preds.sum(dim=(1,2, 3)) + targets.sum(dim=(1,2, 3))
    dice = (2*intersec + smooth)/(union + smooth)
    return dice.mean()


def train(epoch, model, dataloader, optimizer, loss_fn, accuracy_fn, device, args):
    model.train()
    total_iters = 0
    epoch_accuracy=0
    epoch_loss=0
    start_time = time.time()
    for idx, (X,y) in enumerate(dataloader):
        # Format the data
        y = y.float()
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
 
        # Compute loss and accuracy
        batch_loss = loss_fn(y_pred, y)
        loss = batch_loss
        acc = accuracy_fn(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update epoch accuracy and loss
        epoch_accuracy += acc.item() / len(dataloader)
        epoch_loss += loss.item() / len(dataloader)
        total_iters += 1

        # Print every args.print_every iterations
        if idx % args.print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}")
    tqdm.write(f"== [TRAIN] Epoch: {epoch}, Accuracy: {epoch_accuracy:.3f} ==>")
    return epoch_loss, epoch_accuracy, time.time() - start_time


def evaluate(epoch, model, dataloader, loss_fn, accuracy_fn, device, args, mode="val"):
    model.eval()
    epoch_accuracy=0
    epoch_loss=0
    total_iters = 0
    start_time = time.time()
    with torch.no_grad():
        for idx, (X,y) in enumerate(dataloader):
            # Format the data
            y = y.float()
            X, y = X.to(device), y.to(device)

            # Forward pass
            logits = model(X)
            
            # Compute loss and accuracy
            batch_loss = loss_fn(logits, y)
            loss = batch_loss
            acc = accuracy_fn(logits, y)

            # Update epoch accuracy and loss
            epoch_accuracy += acc.item() / len(dataloader)
            epoch_loss += loss.item() / len(dataloader)
            total_iters += 1

            # Print every args.print_every iterations
            if idx % args.print_every == 0:
                tqdm.write(
                    f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}"
                )
        tqdm.write(
            f"=== [{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Accuracy: {epoch_accuracy:.3f} ===>"
        )
    return epoch_loss, epoch_accuracy, time.time() - start_time

# Defining Data Augmentation 1: Random Horizontal Flips:
class RHorizontalFlip:
    # Defining constructor 
    def __init__(self, p=0.5):
        self.p = p
    
    # Defining Call Function 
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        # Returning Image and Mask 
        return image, mask

# Defining Data Augmentation 2: Random Vertical Flip 
class RVerticalFlip:
    # Defining constructor 
    def __init__(self, p=0.5):
        self.p = p
    
    # Defining Call Function 
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        # Returning Image and Mask 
        return image, mask

# Defining Data Augmentation 3: Random Rotation 90 degrees
class RRotate90:
    # Defining constructor 
    def __init__(self, p=0.5):
        self.p = p
    
    # Defining Call Function 
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            k = np.random.randint(1, 4)
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])
        # Returning Image and Mask 
        return image, mask

# Defining Data Augmentation 4: Random Brightness
class RBrightness:
    # Defining constructor 
    def __init__(self, p=0.5, factor=0.2):
        self.p = p
        self.factor = factor 
    
    # Defining Call Function 
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            delta = (torch.rand(1) - 0.5) * 2 * self.factor
            image = torch.clamp(image + delta, 0, 1)
        # Returning Image and Mask 
        return image, mask

# Function to evaluate q3 augmentations 
def evaluate_q3_augmentations(train_x, train_y, valid_x, valid_y, args, device):
    # Write print statement for Q3
    print("\n========== Q3: Data Augmentation Comparison ==========\n")
    
    # Defining different augmentations
    augmentations = {
        "HorizontalFlip": RHorizontalFlip(),
        "VerticalFlip": RVerticalFlip(),
        "Rotate90": RRotate90(),
        "Brightness": RBrightness()
    }
    # Defining results
    augmentation_results = {}
    
    # Iterating through augmentation and transform in augmentation items 
    for aug_name, transform in augmentations.items():
        print(f"\nTraining with augmentation: {aug_name}")
        
        # Defining training and validation set 
        train_set = GetDataset(train_x, train_y, transform=transform)
        val_set   = GetDataset(valid_x, valid_y)
        
        # Defining training and validation loader 
        loader_train = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True)
        loader_val = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False)
        
        # Definining model, optimizer, and loss
        model = UNet(input_shape=3, num_classes=1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        loss_fn = DiceCELoss()
        
        # Defining dice validation scores 
        dice_val_scores = []
        val_losses = []
        
        # Iterating through epochs and defining train, dice_val_scores, and augmentation results
        for epoch in range(args.epochs):
            train(epoch, model, loader_train, optimizer,
                  loss_fn, dice_score, device, args)

            val_loss, val_acc, _ = evaluate(epoch, model, loader_val,
                                     loss_fn, dice_score, device, args)

            dice_val_scores.append(val_acc)
            val_losses.append(val_loss)

        augmentation_results[aug_name] = dice_val_scores

        # Plot Accuracy and Losses curve
        plt.figure()
        plt.plot(dice_val_scores)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy Score")
        plt.title(f"{aug_name} Augmentation")
        plt.savefig(os.path.join(args.logdir,
                    f"augmentation_{aug_name}.png"))
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(val_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss Score")
        plt.title(f"{aug_name} Augmentation")
        plt.savefig(os.path.join(args.logdir,
                    f"augmentation_{aug_name}.png"))
        plt.show()
        plt.close()


    # Returning augmentation results 
    return augmentation_results

def evaluate_q4_pretrained(train_x, train_y, valid_x, valid_y, args, device):
    print("\n========== Q4: Pretrained UNet Fine-Tuning ==========\n")
    # Defining train and val data set without augmentations 
    train_set = GetDataset(train_x, train_y)
    val_set   = GetDataset(valid_x, valid_y)
     
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False)

    # Defining the pretrained UNet
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
    # Defining optimizer, loss, and validation scores 
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)

    loss_fn = DiceCELoss()

    dice_val_scores = []
    val_losses = []

    for epoch in range(args.epochs):
        train(epoch, model, train_loader,
              optimizer, loss_fn, dice_score, device, args)

        val_loss, val_acc, _ = evaluate(epoch, model, val_loader,
                                 loss_fn, dice_score, device, args)

        dice_val_scores.append(val_acc)
        val_losses.append(val_loss)

    # Plot curve for accuracy and loss 
    plt.figure()
    plt.plot(dice_val_scores)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy Score")
    plt.title("Pretrained UNet (ResNet18 Encoder)")
    plt.savefig(os.path.join(args.logdir, "q4_pretrained_unet.png"))
    plt.show()
    plt.close()

    # Plot curve for accuracy and loss 
    plt.figure()
    plt.plot(val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Losses Score")
    plt.title("Pretrained UNet (ResNet18 Encoder)")
    plt.savefig(os.path.join(args.logdir, "q4_pretrained_unet.png"))
    plt.show()
    plt.close()



    print("\nFinished Q4 pretrained training.\n")

    return dice_val_scores



def main():
    # Parse the arguments
    args = parse_args()

    # Build output folder
    os.makedirs(args.logdir, exist_ok=True)

    # Check for the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # We load the dataset
    dataset_path = args.dataset
    train_x = sorted(glob(f"{dataset_path}/train/image/*"))
    train_y = sorted(glob(f"{dataset_path}/train/mask/*"))
    valid_x = sorted(glob(f"{dataset_path}/test/image/*"))
    valid_y = sorted(glob(f"{dataset_path}/test/mask/*"))
    print(f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n")
    # Load the datasets with the custom Dataset class
    train_set = GetDataset(train_x, train_y)
    val_set = GetDataset(valid_x, valid_y)
    
    # Load model
    print(f'Build UNET model...')
    model = UNet(input_shape=3, num_classes=1)
    model.to(device)
    print(f"Initialized UNET model with {sum(p.numel() for p in model.parameters())} "
          f"total parameters, of which {sum(p.numel() for p in model.parameters() if p.requires_grad)} are learnable.")
    print("Model architecture:\n", model)
    print("\n")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Loss function
    loss_fn = DiceCELoss()
    # Accuracy function
    accuracy_fn = dice_score
    
    # Initialize lists to store metrics
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    train_times, valid_times = [], []
    
    # We define a set of data loaders that we can use for various purposes later.
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    # Train and evaluate the model
    print(f'Training UNET model...')
    for epoch in tqdm(range(args.epochs)):
        tqdm.write(f"====== Epoch {epoch} ======>")

        # Train the model
        loss, acc, wall_time = train(epoch, model, train_dataloader, optimizer, loss_fn, accuracy_fn, device, args)
        train_losses.append(loss)
        train_accs.append(acc)
        train_times.append(wall_time)

        # Evaluate the model
        loss, acc, wall_time = evaluate(epoch, model, valid_dataloader, loss_fn, accuracy_fn, device, args)
        valid_losses.append(loss)
        valid_accs.append(acc)
        valid_times.append(wall_time)
    
    ## Evaluating U1 NoSkip Model
    # Load model
    print(f'Build UNET model...')
    model_noskip = UNetNoSkip(input_shape=3, num_classes=1)
    model_noskip.to(device)
    print(f"Initialized UNET NoSkip model with {sum(p.numel() for p in model_noskip.parameters())} "
          f"total parameters, of which {sum(p.numel() for p in model_noskip.parameters() if p.requires_grad)} are learnable.")
    print("Model architecture:\n", model_noskip)
    print("\n")
    
    # Optimizer
    optimizer_noskip = optim.AdamW(model_noskip.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Loss function
    loss_fn_noskip = DiceCELoss()
    # Accuracy function
    accuracy_fn_noskip = dice_score
    
    # Initialize lists to store metrics
    train_losses_noskip, valid_losses_noskip = [], []
    train_accs_noskip, valid_accs_noskip = [], []
    train_times_noskip, valid_times_noskip = [], []
    
    # We define a set of data loaders that we can use for various purposes later.
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    # Train and evaluate the model
    print(f'Training UNET NoSkip model...')
    for epoch in tqdm(range(args.epochs)):
        tqdm.write(f"====== Epoch {epoch} ======>")

        # Train the model
        loss_noskip, acc_noskip, wall_time_noskip = train(epoch, model_noskip, train_dataloader, optimizer_noskip, loss_fn_noskip, accuracy_fn_noskip, device, args)
        train_losses_noskip.append(loss_noskip)
        train_accs_noskip.append(acc_noskip)
        train_times_noskip.append(wall_time_noskip)

        # Evaluate the model
        loss_noskip, acc_noskip, wall_time_noskip = evaluate(epoch, model_noskip, valid_dataloader, loss_fn_noskip, accuracy_fn_noskip, device, args)
        valid_losses_noskip.append(loss_noskip)
        valid_accs_noskip.append(acc_noskip)
        valid_times_noskip.append(wall_time_noskip)
    
    # Plotting UNet Skip vs UNet NoSkip validation accuracy results 
    plt.figure()
    plt.plot(valid_accs, label="UNet (Skip)")
    plt.plot(valid_accs_noskip, label="UNet (No Skip)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.title("Skip vs NoSkip Comparison")
    plt.savefig(os.path.join(args.logdir, "q1_skip_vs_noskip.png"))
    plt.show()
    plt.close()

    # Plotting UNet Skip vs UNet NoSkip Loss results 
    plt.figure()
    plt.plot(valid_losses, label="UNet (Skip)")
    plt.plot(valid_losses_noskip, label="UNet (No Skip)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Losses")
    plt.legend()
    plt.title("Skip vs NoSkip Comparison")
    plt.savefig(os.path.join(args.logdir, "q1_skip_vs_noskip.png"))
    plt.show()
    plt.close()


    ## Q2 Defining Learning Rates:
    l_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    results = {}
    
    # Defining different learning rates 
    for lr in l_rates:
        print(f"\n\n==============================")
        print(f"Training with Learning Rate: {lr}")
        print(f"==============================")
        
        # Defining UNet model with appropriate parameters 
        model = UNet(input_shape=3, num_classes=1)
        model.to(device)
        
        # Defining optimizer and loss 
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        loss_fn = DiceCELoss()
        
        # Defining training and validation loss, as well as validation accuracy 
        train_losses = []
        train_accs = []
        valid_losses = []
        valid_accs = []
        
        # Iterating through the different epochs 
        for epoch in range(args.epochs):
            # Defining traning loss and accuracy as well as validation loss and accuracy 
            train_loss, train_acc, _ = train(epoch, model, train_dataloader, optimizer, loss_fn, dice_score, device, args)
            val_loss, val_acc, _ = evaluate(epoch, model, valid_dataloader, loss_fn, dice_score, device, args)
            
            # Appending training loss, training accuracy, validation loss and validation accuracy 
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(val_loss)
            valid_accs.append(val_acc)
        
        # Defining results for validation accuracy and loss
        results[lr] = {"val_dice": valid_accs,"val_loss": valid_losses}

    # Plotting the curves of the different learning rates     
    plt.figure()
    for lr in l_rates:
        plt.plot(results[lr]["val_dice"], label=f"LR={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy Score")
    plt.title("Learning Rate Comparison")
    plt.legend()
    plt.savefig(os.path.join(args.logdir, "lr_comparison.png"))
    plt.show()

    
    # Plotting the curves of the different learning rates     
    plt.figure()
    for lr in l_rates:
        plt.plot(results[lr]["val_loss"], label=f"LR={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss Score")
    plt.title("Learning Rate Comparison")
    plt.legend()
    plt.savefig(os.path.join(args.logdir, "lr_comparison.png"))
    plt.show()
    
    # Calling function to conduct q3 image augmentations 
    evaluate_q3_augmentations(train_x, train_y, valid_x, valid_y, args, device)

    # Calling function to execute q4
    evaluate_q4_pretrained(train_x, train_y, valid_x, valid_y, args, device)

if __name__ == "__main__":
    main()