import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelConvNet(nn.Module):
   """
    A parallel convolutional neural network designed for EEG signal classification,
    utilizing separate pathways to process temporal and spatial features simultaneously,
    inspired by the EEGNet architecture. This design aims to maximize feature extraction
    efficiency and improve classification accuracy by combining different aspects of EEG signals.

    Attributes:
        temporal_branch (torch.nn.Sequential): A sequential model processing temporal features.
        spatial_branch (torch.nn.Sequential): A sequential model processing spatial features.
        classifier (torch.nn.Sequential): A fully connected layer that outputs class probabilities.

    Args:
        input_shape (tuple of int): The shape of the input data (batch_size, 1, channels, time_points).
                                    It must be provided to setup the layers appropriately.
        num_classes (int): The number of target classes for classification.

    Raises:
        ValueError: If `input_shape` is not provided, which is crucial for configuring the layers.
    """

    def __init__(self, input_shape, num_classes=4):
        super(ParallelConvNet, self).__init__()
        
        if input_shape is None:
            raise ValueError("input_shape must be specified")
        T, C = input_shape[1], input_shape[2]  # Time points and channels
        
        # Temporal Convolution Branch
        self.temporal_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, T // 2), padding=(0, T // 4)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(1, 4))
        )
        
        # Spatial Convolution Branch
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(C, 1), groups=1, padding=(C // 2, 0)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(1, 4))
        )
        
        # Combine features from both branches
        with torch.no_grad():
            # Dummy input to calculate output size after branches
            dummy_input = torch.zeros(1, 1, C, T)
            dummy_temporal = self.temporal_branch(dummy_input)
            dummy_spatial = self.spatial_branch(dummy_input)
            combined_size = dummy_temporal.numel() + dummy_spatial.numel()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(combined_size, num_classes)
        )

    def forward(self, x):
        if x.shape[1] != 1:
            x = x.permute(0, 3, 2, 1) 
        
        
        temporal_features = self.temporal_branch(x)
        spatial_features = self.spatial_branch(x)
        
        
        combined_features = torch.cat((temporal_features.flatten(1), spatial_features.flatten(1)), dim=1)
        
        # Classifier
        log_probs = F.log_softmax(self.classifier(combined_features), dim=1)
        return log_probs
