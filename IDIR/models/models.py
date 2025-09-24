"""
Wolterink, Jelmer M., Jesse C. Zwienenberg, and Christoph Brune. 
‘Implicit Neural Representations for Deformable Image Registration’. 
Proceedings of The 5th International Conference on Medical 
Imaging with Deep Learning, PMLR, 4 December 2022, 1349–59. 
https://proceedings.mlr.press/v172/wolterink22a.html.

"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from ..networks import networks
from ..objectives import ncc, regularizers
from ..utils import general

DEFAULT_ARGS = {
            "mask": None,
            "mask_2": None,
            # "method": 1,
            "lr": 1e-4, # Paper default: 1e-4
            "batch_size": 10000,
            "layers": [3, 256, 256, 256, 3],
            "velocity_steps": 1,

            "output_regularization": False,
            "alpha_output": 0.2,
            "reg_norm_output": 1,

            "jacobian_regularization": False,
            "alpha_jacobian": 0.05,

            "hyper_regularization": False,
            "alpha_hyper": 0.25,

            "bending_regularization": False,
            "alpha_bending": 10.0,

            "image_shape": (200, 200),
            "network": None,

            "epochs": 2500, # Paper default: 2500
            "log_interval": 2500 // 4,   # auto-calculated
            "verbose": True,
            "save_folder": "output",

            "network_type": "MLP",
            "optimizer": "Adam", # Paper default: Adam
            "loss_function": "ncc",
            "momentum": 0.5,

            "positional_encoding": False,
            "weight_init": True,
            "omega": 32,
            "seed": 1,
        }

class ImplicitRegistrator:
    """This is a class for registering implicitly represented images."""

    def __call__(
        self, coordinate_tensor=None, output_shape=(28, 28), dimension=0, slice_pos=0
    ):
        """ Queries a slice of the trained Implicit Neural Network.
            Returns the image-values for the given input-coordinates.
            Supports 2D

        Args:
            coordinate_tensor (torch.tensor, optional): coordinates to evaluate
                the neural network at. Defaults to None.
            output_shape (tuple, optional): output shape of image. Defaults to (28, 28).
            dimension (int, optional): _description_. Defaults to 0.
            slice_pos (int, optional): _description_. Defaults to 0.

        Returns:
            transformed_image (np.ndarray): _description_
        """

        # Use standard coordinate tensor if none is given
        if coordinate_tensor is None:
            coordinate_tensor = general.make_coordinate_slice(
                output_shape, dimension, slice_pos, device=self.device
            )

        output = self.network(coordinate_tensor)

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(output, coordinate_tensor)

        transformed_image = self.transform_no_add(coord_temp)
        return (
            transformed_image.cpu()
            .detach()
            .numpy()
            .reshape(output_shape[0], output_shape[1])
        )

    def __init__(self, moving_image, fixed_image, **kwargs):
        """Initialize the learning model."""

        # Set all default arguments in a dict: self.args
        self.args = self.make_default_arguments()
        # Check if all kwargs keys are valid (this checks for typos)
        assert all(kwarg in self.args.keys() for kwarg in kwargs)

        # Parse important argument from kwargs
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        for key in ["epochs", "log_interval", "lr", "momentum", "optimizer",
            "loss_function", "layers", "weight_init",
            "omega", "save_folder", "verbose", 
            "network_type",
            # Mask,
            "mask",
            # Regularisation
            "jacobian_regularization", 
            "alpha_jacobian",
            "hyper_regularization", "alpha_hyper",
            "bending_regularization", "alpha_bending",
            "image_shape", "batch_size"
        ]:
            setattr(self, key if key != "optimizer" and key != "loss_function" 
                        else key + "_arg", 
                    kwargs.get(key, self.args[key]))

        # Make folder for output
        if not self.save_folder == "" and not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        # Add slash to divide folder and filename
        self.save_folder += "/"

        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Load network
        self.network_from_file = (
            kwargs["network"] if "network" in kwargs else self.args["network"]
        )
        self.network =  self.make_network(self.network_from_file)

        # Choose the optimizer
        self.optimizer = self.make_optimizer(self.optimizer_arg.lower(), 
                            self.network.parameters(), 
                            lr=self.lr, 
                            momentum=self.momentum)

        # Choose the loss function
        self.criterion = self.make_loss(self.loss_function_arg.lower())

        # Initialization
        self.moving_image = moving_image
        self.fixed_image = fixed_image

        # Set seed
        torch.manual_seed(self.args["seed"])

        self.possible_coordinate_tensor = general.make_coordinate_tensor(self.fixed_image.shape, 
                                                                         mask=self.mask, 
                                                                         device=self.device)
        # Move variables to GPU
        self.network.to(self.device)
        self.moving_image = self.moving_image.to(self.device)
        self.fixed_image = self.fixed_image.to(self.device)
        self.possible_coordinate_tensor = self.possible_coordinate_tensor.to(self.device)

        if self.verbose:
            print('Image shape: ', self.fixed_image.shape)
            print('Coordinate tensor shape: ', self.possible_coordinate_tensor.shape)
            print('Batch size: ', self.batch_size)


    # Factory functions
    def make_network(self, network_from_file):
        if network_from_file is None:
            if self.network_type == "MLP":
                network = networks.MLP(self.layers)
            else:
                network = networks.Siren(self.layers, self.weight_init, self.omega)
  
        else:
            self.network = torch.load(network_from_file)

        if self.verbose:
            print(
                "Network contains {} trainable parameters.".format(
                    general.count_parameters(network)
                )
            )
            print(network)
        return network
    
    def make_optimizer(self, optimizer_arg, network_params, lr, momentum):    
        if optimizer_arg == "sgd":
            optimizer = optim.SGD(
                network_params, lr=lr, momentum=momentum
            )
        elif optimizer_arg == "adam":
            optimizer = optim.Adam(network_params, lr=lr)
        elif optimizer_arg == "adadelta":
            optimizer = optim.Adadelta(network_params, lr=lr)
        else:
            optimizer = optim.SGD(
                network_params, lr=lr, momentum=momentum
            )
            print(
                "WARNING: "
                + str(optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )
        return optimizer
    
    def make_loss(self, loss_key):
        loss_functions = {
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "ncc": ncc.NCC(),
            "smoothl1": nn.SmoothL1Loss(beta=0.2),
            "huber": nn.HuberLoss(),
        }

        criterion = loss_functions.get(loss_key, nn.MSELoss())

        if loss_key not in loss_functions:
            print(f"WARNING: {self.loss_function_arg} not recognized as loss function, picked MSE instead")

        return criterion
    
    def make_default_arguments(self):
        """Set default arguments."""
        # Inherit default arguments from standard learning model
        args = DEFAULT_ARGS
        args["log_interval"] = args["epochs"] // 4
        return args
    
    # Training code
    def training_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.train()

        loss = 0

        # Sample batch_size coordinates from the large coordinate tensor
        indices = torch.randperm(
            self.possible_coordinate_tensor.shape[0], 
            device = self.device
        )[: self.batch_size]

        coordinate_tensor = self.possible_coordinate_tensor[indices, :] # [B,3]
        coordinate_tensor = coordinate_tensor.requires_grad_(True)

        # Get actual coordinates by addition as Network gives displacements u(x) 
        # from F(x) -> M(x) 
        output = self.network(coordinate_tensor)
        coord_temp = torch.add(output, coordinate_tensor)
        output = coord_temp

        # Perform registration via sampling moving image M in mapped 
        # coordinates Phi i.e. M(Phi(x))
        transformed_image = self.transform_no_add(coord_temp)
        # Sample fixed image at the same coordinates
        fixed_image = general.fast_trilinear_interpolation(
            self.fixed_image,
            coordinate_tensor[:, 0],
            coordinate_tensor[:, 1],
            coordinate_tensor[:, 2],
        )

        # Compute the loss
        loss += self.criterion(transformed_image, fixed_image)

        # Store the value of the data loss
        if self.verbose:
            self.data_loss_list[epoch] = loss.detach().cpu().numpy()

        # Relativation of output
        output_rel = torch.subtract(output, coordinate_tensor)

        # Regularization
        if self.jacobian_regularization:
            loss += self.alpha_jacobian * regularizers.compute_jacobian_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.hyper_regularization:
            loss += self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.bending_regularization:
            loss += self.alpha_bending * regularizers.compute_bending_energy(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )

        # Perform the backpropagation and update the parameters accordingly

        for param in self.network.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()

        # Store the value of the total loss
        if self.verbose:
            self.loss_list[epoch] = loss.detach().cpu().numpy()

    def transform(
        self, transformation, coordinate_tensor=None, moving_image=None, reshape=False
    ):
        """Transform moving image given a transformation."""

        # If no specific coordinate tensor is given use the standard one of 28x28
        if coordinate_tensor is None:
            coordinate_tensor = self.coordinate_tensor

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image

        # From relative to absolute
        transformation = torch.add(transformation, coordinate_tensor)
        return general.fast_trilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def transform_no_add(self, transformation, moving_image=None):
        """Transform moving image given a transformation."""

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image
        # print('GET MOVING')
        return general.fast_trilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def fit(self, epochs=None):
        """Train the network."""

        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Extend lost_list if necessary
        if not len(self.loss_list) == epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]

        # Perform training iterations
        for i in tqdm.tqdm(range(epochs)):
            self.training_iteration(i)
