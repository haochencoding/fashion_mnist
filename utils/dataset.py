import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def import_data(file_path, images, train, flatten=True):
    """
    Imports FashionMNIST data from a given file path.

    Args:
        file_path (str): The path to the data file (either images or labels).
        images (bool): If True, expects an image file;
            otherwise, expects a label file.
        train (bool): If True, expects the training dataset;
            otherwise, expects the test dataset.
        flatten (bool): If True, flatten the image.
    Returns:
        numpy.ndarray: A numpy array of the imported data.
            For images, it is an ND array, where each row/matrix is a image.
            For labels, it will be a 1D array of labels.
    """
    if images:
        with open(file_path, 'rb') as file:
            # Read the labels data starting from the 8th byte.
            data = np.frombuffer(file.read(), dtype=np.uint8, offset=16)
            # Flatten the data
            # Each row represents an image, each column represents a pixel
            if train & flatten:
                data = data.reshape(60000, 28*28)
            elif train:
                data = data.reshape(60000, 28, 28)
            elif flatten:
                data = data.reshape(10000, 28*28)
            else:
                data = data.reshape(10000, 28, 28)
    else:
        with open(file_path, 'rb') as file:
            # Read the labels data starting from the 8th byte
            data = np.frombuffer(file.read(), dtype=np.uint8, offset=8)

    # Print data summary
    data_type = 'images' if images else 'labels'
    print(f'Imported file: {file_path}')
    print(f'Data type: {data_type}; Data shape: {data.shape}')

    return data


class FashionData(Dataset):
    def __init__(self, images, labels, transform=None, unsqueeze=False):
        # transform data into tensor
        images_tensor = torch.from_numpy(images.copy()).float()
        labels_tensor = torch.from_numpy(labels.copy())

        self.images = images_tensor
        self.labels = labels_tensor
        self.transform = transform
        self.unsqueeze = unsqueeze

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image and label at a given index in the data
        image = self.images[idx]
        label = self.labels[idx]

        if self.unsqueeze:
            # Add a gray scale channel dimension
            # Transform from [28, 28] to [1, 28, 28]
            image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, label
