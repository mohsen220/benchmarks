import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    convolutional neural network for EEG signal classification,
    with particular adjustments for robust handling of kernel sizes and padding to better suit EEG data.

    Attributes:
        features (torch.nn.Sequential): A sequence of convolutional, batch normalization, activation,
                                        dropout, and pooling layers that process the input data.
        classifier (torch.nn.Sequential): A fully connected layer that outputs the class probabilities.

    Args:
        input_shape (tuple of int): The shape of the input data, specified as (batch_size, 1, channels, time_points).
                                    It must be provided to set up the layers correctly.
        num_classes (int): The number of classes for the output layer.

    Raises:
        ValueError: If `input_shape` is None, which is essential for setting up the layers.
    """

    def __init__(self, input_shape, num_classes=4):
        super(CNN, self).__init__()

        if input_shape is None:
            raise ValueError("input_shape must be specified")
        T, C = input_shape[1], input_shape[2]  # Time points and channels

        # Define a simple CNN structure
        self.features = nn.Sequential(
            # Temporal Convolution
            nn.Conv2d(1, 16, kernel_size=(1, max(1, T // 2)), padding=(0, max(1, T // 4))),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(1, 4)),

            # Spatial Convolution
            nn.Conv2d(16, 32, kernel_size=(C, 1), groups=16, padding=(C // 2, 0)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(1, 4)),

            # Pointwise Convolution to mix channels
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, C, T)  # Create a dummy input to calculate the size
            dummy_features = self.features(dummy_input)
            self._to_linear = dummy_features.view(-1).shape[0]

        # Fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, num_classes)
        )

    def forward(self, x):
        if x.shape[1] != 1:
            x = x.permute(0, 3, 2, 1)  # Reordering dimensions if necessary

        x = self.features(x)
        log_probs = F.log_softmax(self.classifier(x), dim=1)
        return log_probs


