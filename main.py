import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from numpy import linalg as LA
from PIL import Image
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Main references:
# https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
# Paper: https://arxiv.org/pdf/1404.4661.pdf

# Define constants and settings
VERSION = "v3"
PATH = ""
IMAGE_SIZE = (224, 224)  # Default image size for Resnet18
BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.00001
NESTEROV = True
TRAIN_FILE = "train_triplets_split.txt"
TEST_FILE = "test_triplets.txt"
VAL_FILE = "validation_triplets_split.txt"
# The validation set was created by selecting 1000 triplets from the
# training set.

np.random.seed(42)
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# Neural network architecture
# -----------------------------------------------------------------------------


class SiameseNeuralNetwork(torch.nn.Module):
    """
    Siamese Neural Network that combines extracted features. This is done
    for a triplet of images. This Network has to be initialized with a
    pretrained feature extractor.
    """

    def __init__(self, feature_extractor):
        """Initializes the Siamese Network."""
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, image1, image2, image3):
        """Forward pass of the network."""

        # Calculate the features of the images
        image1_feat = self.feature_extractor(image1)
        image2_feat = self.feature_extractor(image2)
        image3_feat = self.feature_extractor(image3)

        return image1_feat, image2_feat, image3_feat


class FeatureExtractorNetwork(torch.nn.Module):
    """
    Pretrained Neural Network used for extracting image features.
    Currently, this is the ResNet18 network with a customized fc layer.
    """

    def __init__(self):
        """Initializes the feature extractor."""
        super().__init__()

        # Initialize the ResNet18 network (with customized fc layer)
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(512, 1024)  # Double the size

    def forward(self, x):
        """Forward pass of the network."""
        x = self.resnet(x)
        return x


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------


class NormalizationDataset(Dataset):
    """Initializes the dataset for calculating the mean and std."""

    def __init__(self, root_dir, transform):
        self.root_dit = root_dir
        self.transform = transform

    def __len__(self):
        return 10000

    def __getitem__(self, id):
        """Returns the image for a given index."""
        img_path = os.path.join(
            self.root_dit, "food", f"{str(id).zfill(5)}.jpg"
        )
        image = self.transform(Image.open(img_path))
        return image


class SiameseDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform):
        """
        Initializes the dataset containing the triplets.

        Args:
            txt_file: Name of the file containing the triplets.
            root_dir: Path to the root dir.
            transform: Transformation to apply to the images.
        """
        triplets_file = os.path.join(root_dir, txt_file)

        self.triplet_ids = pd.read_csv(
            triplets_file, sep=" ", header=None, dtype=str
        )
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.triplet_ids)

    def __getitem__(self, id):
        """Returns the triplet at the given index."""
        image1_id = self.triplet_ids.iloc[id, 0]
        image2_id = self.triplet_ids.iloc[id, 1]
        image3_id = self.triplet_ids.iloc[id, 2]

        img1_path = os.path.join(self.root_dir, "food", f"{image1_id}.jpg")
        img2_path = os.path.join(self.root_dir, "food", f"{image2_id}.jpg")
        img3_path = os.path.join(self.root_dir, "food", f"{image3_id}.jpg")

        image1 = self.transform(Image.open(img1_path))
        image2 = self.transform(Image.open(img2_path))
        image3 = self.transform(Image.open(img3_path))

        return image1, image2, image3


def get_mean_and_std(root_input, norm_transform):
    """
    Calculates the mean and std of the training dataset. These values are then
    used for the data normalization.
    """
    print("-------- Calculating mean and std of the dataset --------")
    dataset = NormalizationDataset(root_input, norm_transform)
    data_loader = DataLoader(dataset, batch_size=100, shuffle=False)

    channels_sum, channels_sq_sum, num_batches = 0, 0, 0
    for data in data_loader:
        channels_sum += torch.mean(data, dim=(0, 2, 3))
        channels_sq_sum += torch.mean(data**2, dim=(0, 2, 3))
        num_batches += 1

        if num_batches % 10 == 0:
            print(f"Processed {num_batches*100} images.")

    mean = channels_sum / num_batches
    std = ((channels_sq_sum / num_batches) - mean**2) ** 0.5

    print("-------- Calculation of mean and std done --------")
    return mean, std


def get_data_loaders(root_input, IMAGE_SIZE, BATCH_SIZE):
    """Returns the dataloaders for the training, test and validation sets."""

    # norm_transforms = transforms.Compose(
    #     [
    #         transforms.Resize(IMAGE_SIZE),
    #         transforms.ToTensor(),
    #     ]
    # )

    # WARNING: Run this function if the dataset changes and replace the
    #          mean and std with the new ones.
    # data_mean, data_std = get_mean_and_std(root_input, norm_transforms)
    data_mean = [0.6082, 0.5161, 0.4123]
    data_std = [0.2617, 0.2719, 0.2934]

    # Define transformations ot be used on images
    image_transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std),
        ]
    )

    # Load loaders
    train_set = SiameseDataset(TRAIN_FILE, root_input, image_transforms)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_set = SiameseDataset(TEST_FILE, root_input, image_transforms)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    val_set = SiameseDataset(VAL_FILE, root_input, image_transforms)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, val_loader


def main():
    print(f"---------- MODEL Siamese Network {VERSION}----------")
    print("----------------- LOADING DATA -----------------")
    # Load the data
    train_loader, test_loader, val_loader = get_data_loaders(
        PATH, IMAGE_SIZE, BATCH_SIZE
    )
    print("----------------- DATA LOADED! -----------------")

    # Define Siamese Network with the pretrained network
    model = SiameseNeuralNetwork(FeatureExtractorNetwork()).cuda()

    # Define loss function and optimizer
    loss_fn = torch.nn.TripletMarginLoss(margin=0.5)
    optim = SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV,
    )

    print("----------------- TRAINING THE MODEL -----------------")
    # Train the Siamese Network
    loss_fn_val = torch.nn.TripletMarginLoss(margin=0)
    losses_val = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss_predict = 0.0
        val_loss_predict = 0.0
        loss_epoch = 0.0
        t1 = time.time()

        for batch, (image1, image2, image3) in enumerate(train_loader):
            # zero grads
            optim.zero_grad()

            batch_size = image1.shape[0]  # Might differ for the last batch

            # Move to GPU
            image1, image2, image3 = (
                Variable(image1.cuda()),
                Variable(image2.cuda()),
                Variable(image3.cuda()),
            )

            # Forward pass
            img1_feat, img2_feat, img3_feat = model(image1, image2, image3)
            loss = loss_fn(img1_feat, img2_feat, img3_feat)

            # Backward pass
            loss.backward()
            optim.step()

            # Calculate training loss and cumulative loss
            train_loss_predict += torch.sum(
                (loss_fn_val(img1_feat, img2_feat, img3_feat) > 0).int()
                / batch_size
            ).data
            loss_epoch += loss.data

            if batch % 10 == 0:
                print(f"Batch loss: {loss.data:.6f}")

        train_loss_predict /= len(train_loader)
        loss_epoch = loss_epoch / len(train_loader)

        # Validate the model
        model.eval()
        for batch, (image1, image2, image3) in enumerate(val_loader):

            batch_size = image1.shape[0]  # Might differ for the last batch

            # Move to GPU
            image1, image2, image3 = (
                Variable(image1.cuda()),
                Variable(image2.cuda()),
                Variable(image3.cuda()),
            )

            with torch.no_grad():
                img1_feat, img2_feat, img3_feat = model(image1, image2, image3)
                val_loss_predict += torch.sum(
                    (loss_fn_val(img1_feat, img2_feat, img3_feat) > 0).int()
                    / batch_size
                ).data

        val_loss_predict /= len(val_loader)
        losses_val.append(val_loss_predict)
        t2 = time.time() - t1

        print(
            f"Epoch: {epoch} | Loss: {loss_epoch:.4f} "
            f"| Train loss: {train_loss_predict:.4f} | Val loss: "
            f"{val_loss_predict:.4f} | Time: {t2:.2f} sec"
        )

        # Save the model state
        save_path = f"model-siam-net-v3_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
            },
            save_path,
        )
        print(f"Saved model checkpoint to {save_path}")

    print("----------------- CALCULATING PREDICTIONS -----------------")
    # Calculate predictions
    model.eval()

    test_predictions = []

    # Iterate over the test set
    for batch, (image1, image2, image3) in enumerate(test_loader):
        if batch % 10 == 0:
            print("Batch:", batch)

        # Move to GPU
        image1, image2, image3 = (
            Variable(image1.cuda()),
            Variable(image2.cuda()),
            Variable(image3.cuda()),
        )

        with torch.no_grad():
            img1_feat, img2_feat, img3_feat = model(image1, image2, image3)

        # Calculate distances between images
        diff_12 = (img1_feat - img2_feat).cpu().detach().numpy()
        d_12 = LA.norm(np.squeeze(diff_12), ord=2, axis=-1)
        diff_13 = (img1_feat - img3_feat).cpu().detach().numpy()
        d_13 = LA.norm(np.squeeze(diff_13), ord=2, axis=-1)

        test_predictions.append((d_13 >= d_12).astype(int))

    # Complete the predictions and save to a txt file
    print("----------------- SAVING PREDICTIONS -----------------")
    predictions = np.hstack(test_predictions)
    np.savetxt(f"predictions-{VERSION}.txt", predictions, fmt="%d")
    print("----------------- SCRIPT FINISHED -----------------")


if __name__ == "__main__":
    main()
