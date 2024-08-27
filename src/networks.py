import abc
import torch
import torch.nn as nn
from torch.distributions import TransformedDistribution
from typing import Union

def initialize_weights(m: nn.Module, initializer=nn.init.xavier_uniform_):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        initializer(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def mlp(input_dim, output_dim, hidden_units, output_activation=None, use_batchnorm=False, dropout_rate=0.0):
    layers = []
    in_units = input_dim
    for out_units in hidden_units:
        layers.append(nn.Linear(in_features=in_units, out_features=out_units))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_units))  # Add BatchNorm layer if use_batchnorm is True
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))  # Add Dropout layer if dropout_rate > 0
        in_units = out_units
    layers.append(nn.Linear(in_features=in_units, out_features=output_dim))
    if output_activation:
        layers.append(output_activation)
    return nn.Sequential(*layers).apply(initialize_weights)

def cnn(input_channels, output_dim, conv_layers, fc_hidden_units, output_activation=None, use_batchnorm=False, dropout_rate=0.0):
    """
    Creates a convolutional neural network (CNN) followed by fully connected layers.
    
    Args:
        input_channels (int): Number of input channels (e.g., 4 for a stacked grayscale frame input).
        output_dim (int): Number of output dimensions (e.g., action size).
        conv_layers (list): List of tuples defining convolutional layers (out_channels, kernel_size, stride).
        fc_hidden_units (list): List of hidden units for fully connected layers after convolutions.
        output_activation (callable, optional): Activation function to apply to the output layer. Defaults to None.
        use_batchnorm (bool, optional): Whether to include BatchNorm layers. Defaults to False.
        dropout_rate (float, optional): Dropout rate. If 0.0, no Dropout layers are added. Defaults to 0.0.
        
    Returns:
        nn.Sequential: A sequential container of the CNN followed by FC layers.
    """
    layers = []
    in_channels = input_channels

    # Convolutional layers
    for out_channels, kernel_size, stride in conv_layers:
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))  # Add BatchNorm layer if use_batchnorm is True
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))  # Add Dropout layer if dropout_rate > 0
        in_channels = out_channels
    
    # Add an adaptive pooling layer to handle varying input sizes automatically
    layers.append(nn.AdaptiveAvgPool2d(1))
    
    # Flatten the output of the conv layers
    layers.append(nn.Flatten())
    
    # Fully connected layers
    fc_in_dim = in_channels  # Flattened to a single dimension due to adaptive pooling
    for out_units in fc_hidden_units:
        layers.append(nn.Linear(in_features=fc_in_dim, out_features=out_units))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_units))  # Add BatchNorm layer if use_batchnorm is True
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))  # Add Dropout layer if dropout_rate > 0
        fc_in_dim = out_units
    
    # Output layer
    layers.append(nn.Linear(fc_in_dim, output_dim))
    if output_activation:
        layers.append(output_activation)
    
    return nn.Sequential(*layers).apply(initialize_weights)


class BaseActorNetwork(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, state_size, action_size, fc_hidden_units=None, conv_layers=None, 
                 use_batchnorm=False, dropout_rate=0.0):
        super().__init__()
        self.image_input = len(state_size) == 3 # Check if input is an image (3D: channels, height, width)
        input_dim = state_size[0]

        if self.image_input:  # CNN for image input (3D: channels, height, width)
            self.actor = self.create_cnn(input_dim, action_size, conv_layers, fc_hidden_units, 
                                         use_batchnorm, dropout_rate)
        else:  # MLP for flat vector input 
            self.actor = self.create_mlp(input_dim, action_size, fc_hidden_units, dropout_rate)
    @abc.abstractmethod 
    def create_cnn(self, input_dim, action_size, conv_layers, fc_hidden_units, 
                use_batchnorm, dropout_rate) -> nn.Module:
        pass
    
    @abc.abstractmethod
    def create_mlp(self, input_dim, action_size, fc_hidden_units, dropout_rate) -> nn.Module:
        pass

    @abc.abstractmethod
    def forward(self, state) -> Union[torch.Tensor, 'TransformedDistribution']:
        pass


class DoubleQCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc_hidden_units=None, conv_layers=None, 
                 use_batchnorm=False, dropout_rate=0.0):
        super().__init__()
        self.image_input = len(state_size) == 3  # Check if input is an image

        input_dim = state_size[0]
        if self.image_input:  # Image input (channels, height, width)
            self.q1 = cnn(input_channels=input_dim + action_size,
                          output_dim=1,
                          conv_layers=conv_layers,
                          fc_hidden_units=fc_hidden_units,
                          use_batchnorm=use_batchnorm,
                          dropout_rate=dropout_rate)
            self.q2 = cnn(input_channels=input_dim + action_size,
                          output_dim=1,
                          conv_layers=conv_layers,
                          fc_hidden_units=fc_hidden_units,
                          use_batchnorm=use_batchnorm,
                          dropout_rate=dropout_rate)
        else:  # Flat vector input
            self.q1 = mlp(input_dim=input_dim + action_size,
                          output_dim=1,
                          hidden_units=fc_hidden_units,
                          use_batchnorm=use_batchnorm,
                          dropout_rate=dropout_rate)
            self.q2 = mlp(input_dim=input_dim + action_size,
                          output_dim=1,
                          hidden_units=fc_hidden_units,
                          use_batchnorm=use_batchnorm,
                          dropout_rate=dropout_rate)
    
    def forward(self, state, action):
        if self.image_input:
            action = action.unsqueeze(-1).unsqueeze(-1) # Add height and width dimensions
            action = action.expand(-1, -1, state.size(-2), state.size(-1)) # Broadcast action to match state dimensions
        state = state.unsqueeze(0) if state.dim() == 3 and self.image_input else state
        x = torch.cat([state, action], dim=1) # Concatenate state and action along the channel/feature dimension
        return self.q1(x), self.q2(x)