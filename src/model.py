import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super(MyModel, self).__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.model = nn.Sequential(
     
                        #Backbone of the CNN
            
                        # First layer of CNN
                        nn.Conv2d(3, 16, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(16),
                        nn.MaxPool2d(2, 2),
            
                        # Second layer of CNN
                        nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(32),
                        nn.MaxPool2d(2, 2),
            
                        # Third layer of CNN
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2, 2),
                        
                        # Fourth layer of CNN
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2, 2),
            
                        # Fifth layer of CNN
                        nn.Conv2d(128, 256, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(256),
                        nn.MaxPool2d(2, 2),
            
                        # Embedding or Feature vector
                        nn.Flatten(),
            
                        # Head of the network
            
                        # First fully connected layer
                        nn.Linear(7 * 7 * 256, 1024),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(1024),
                        nn.Dropout(p=dropout),
                        
                        # Second fully connected layer
                        nn.Linear(1024, 512),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(512),
                        nn.Dropout(p=dropout),
            
                        # Third fully connected layer
                        #nn.Linear(2048, 4096),
                        #nn.ReLU(),
                        #nn.Dropout(p=dropout),
                        
            
                        # Output layer
                        nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.model(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
"""

import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"

 """   