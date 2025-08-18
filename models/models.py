import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from IDIR.networks import networks
from IDIR.objectives import ncc, regularizers
from IDIR.utils import general


class ImplicitRegistrator:
    """This is a class for registrating implicitly represented images."""

    def __call__(
        self, coordinate_tensor=None, output_shape=(28, 28), dimension=0, slice_pos=0
    ):
        """ Query the trained Implicit Neural Network.
            Returns the image-values for the given input-coordinates.
            Supports 2D

        Args:
            coordinate_tensor (torch.tensor, optional): coordinates to evaluate
                the neural network at. Defaults to None.
            output_shape (tuple, optional): _description_. Defaults to (28, 28).
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
        self.set_default_arguments()

        # Check if all kwargs keys are valid (this checks for typos)
        assert all(kwarg in self.args.keys() for kwarg in kwargs)

        # Parse important argument from kwargs
        self.epochs = kwargs["epochs"] if "epochs" in kwargs else self.args["epochs"]
        self.log_interval = (
            kwargs["log_interval"]
            if "log_interval" in kwargs
            else self.args["log_interval"]
        )
        # self.gpu = kwargs["gpu"] if "gpu" in kwargs else self.args["gpu"]

        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.lr = kwargs["lr"] if "lr" in kwargs else self.args["lr"]
        self.momentum = (
            kwargs["momentum"] if "momentum" in kwargs else self.args["momentum"]
        )
        self.optimizer_arg = (
            kwargs["optimizer"] if "optimizer" in kwargs else self.args["optimizer"]
        )
        self.loss_function_arg = (
            kwargs["loss_function"]
            if "loss_function" in kwargs
            else self.args["loss_function"]
        )
        self.layers = kwargs["layers"] if "layers" in kwargs else self.args["layers"]
        self.weight_init = (
            kwargs["weight_init"]
            if "weight_init" in kwargs
            else self.args["weight_init"]
        )
        self.omega = kwargs["omega"] if "omega" in kwargs else self.args["omega"]
        self.save_folder = (
            kwargs["save_folder"]
            if "save_folder" in kwargs
            else self.args["save_folder"]
        )

        # Parse other arguments from kwargs
        self.verbose = (
            kwargs["verbose"] if "verbose" in kwargs else self.args["verbose"]
        )

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
        self.network_type = (
            kwargs["network_type"]
            if "network_type" in kwargs
            else self.args["network_type"]
        )
        if self.network_from_file is None:
            if self.network_type == "MLP":
                self.network = networks.MLP(self.layers)
            else:
                self.network = networks.Siren(self.layers, self.weight_init, self.omega)
            if self.verbose:
                print(
                    "Network contains {} trainable parameters.".format(
                        general.count_parameters(self.network)
                    )
                )
                print(self.network)
        else:
            self.network = torch.load(self.network_from_file)
            self.network.to(self.device)


        # Choose the optimizer
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )

        elif self.optimizer_arg.lower() == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        elif self.optimizer_arg.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.lr)

        else:
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )

        # Choose the loss function
        loss_functions = {
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "ncc": ncc.NCC(),
            "smoothl1": nn.SmoothL1Loss(beta=0.2),
            "huber": nn.HuberLoss(),
        }

        loss_key = self.loss_function_arg.lower()
        self.criterion = loss_functions.get(loss_key, nn.MSELoss())

        if loss_key not in loss_functions:
            print(f"WARNING: {self.loss_function_arg} not recognized as loss function, picked MSE instead")

        # Move variables to GPU
        self.network.to(self.device)

        # Parse arguments from kwargs
        self.mask = kwargs["mask"] if "mask" in kwargs else self.args["mask"]

        # Parse regularization kwargs
        # and other arguments
        for key in [
            "jacobian_regularization", "alpha_jacobian",
            "hyper_regularization", "alpha_hyper",
            "bending_regularization", "alpha_bending",
            "image_shape", "batch_size"
        ]:
            setattr(self, key, kwargs.get(key, self.args[key]))
        # Set seed
        torch.manual_seed(self.args["seed"])

        # Initialization
        self.moving_image = moving_image
        self.fixed_image = fixed_image

        # self.possible_coordinate_tensor = general.make_masked_coordinate_tensor(
        #     self.mask, self.fixed_image.shape
        # )
        self.possible_coordinate_tensor = general.make_coordinate_tensor()

        self.moving_image = self.moving_image.to(self.device)
        self.fixed_image = self.fixed_image.to(self.device)
        self.possible_coordinate_tensor = self.possible_coordinate_tensor.to(self.device)

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model

        self.args = {
            "mask": None,
            "mask_2": None,
            "method": 1,
            "lr": 1e-5,
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

            "epochs": 2500,
            "log_interval": 2500 // 4,   # auto-calculated
            "verbose": True,
            "save_folder": "output",

            "network_type": "MLP",
            # "gpu": torch.cuda.is_available(),
            "optimizer": "Adam",
            "loss_function": "ncc",
            "momentum": 0.5,

            "positional_encoding": False,
            "weight_init": True,
            "omega": 32,
            "seed": 1,
        }

        self.args["log_interval"] = self.args["epochs"] // 4


    def training_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.train()

        loss = 0
        indices = torch.randperm(
            self.possible_coordinate_tensor.shape[0], 
            device = self.device
        )[: self.batch_size]
        # indices = torch.randperm(
        #     self.possible_coordinate_tensor.shape[0], device="cuda"
        # )[: self.batch_size]
        # indices = indices.to(self.device)
        # self.possible_coordinate_tensor = self.possible_coordinate_tensor.to(torch.mps)

        coordinate_tensor = self.possible_coordinate_tensor[indices, :]
        coordinate_tensor = coordinate_tensor.requires_grad_(True)

        output = self.network(coordinate_tensor)
        coord_temp = torch.add(output, coordinate_tensor)
        output = coord_temp

        transformed_image = self.transform_no_add(coord_temp)
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

    def transform_no_add(self, transformation, moving_image=None, reshape=False):
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

    def fit(self, epochs=None, red_blue=False):
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
    
# 

