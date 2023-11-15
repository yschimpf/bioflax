import argparse
import os
from bioflax.train import train


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for model training.")

    parser.add_argument(
        "--jax_seed", type=int, default=0, help="Seed for JAX RNG. Type: int, Default: 0"
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "sinreg", "teacher"],
        help="Dataset for training. Choices: ['mnist', 'sinprop', 'teacher'], Type: str, Default: 'mnist'",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size during training. Type: int, Default: 32",
    )
    parser.add_argument(
        "--train_set_size",
        type=int,
        default=50,
        help="Size of the training set. Type: int, Default: 50",
    )
    parser.add_argument(
        "--test_set_size", type=int, default=10, help="Size of the test set. Type: int, Default: 10"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Percentage of training samples for validation. Type: float, Default: 0.1",
    )
    parser.add_argument(
        "--in_dim", type=int, default=1, help="Input vector dimension. Type: int, Default: 1"
    )
    parser.add_argument(
        "--seq_len", type=int, default=1, help="Length of input sequence. Type: int, Default: 1"
    )
    parser.add_argument(
        "--output_features",
        type=int,
        default=10,
        help="Number of output features. Type: int, Default: 10",
    )
    parser.add_argument(
        "--teacher_act",
        type=str,
        default="sigmoid",
        help="Activation function of teacher network. Type: str, Default: 'sigmoid'",
    )

    # Network
    parser.add_argument(
        "--mode",
        type=str,
        default="fa",
        choices=["bp", "fa", "kp", "dfa"],
        help="Training mode. Choices: ['bp', 'fa', 'kp', 'dfa']. Type: str, Default: 'fa'",
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        type=str,
        default=["relu", "relu"],
        help="Activation functions for each layer. Type: str (list), Default: ['relu', 'relu']",
    )
    parser.add_argument(
        "--hidden_layers",
        nargs="+",
        type=int,
        default=[500, 500],
        help="Neurons in each hidden layer. Type: int (list), Default: [500, 500]",
    )

    # Optimizer
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs. Type: int, Default: 5"
    )
    parser.add_argument(
        "--lr", type=float, default=0.02, help="Learning rate. Type: float, Default: 0.02"
    )
    parser.add_argument(
        "--momentum", type=float, default=0, help="Momentum value. Type: float, Default: 0"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Weight decay for learning. Type: float, Default: 0.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="Optimizer. Choices: ['sgd', 'adam']. Type: str, Default: 'sgd'",
    )

    # Initializer
    parser.add_argument(
        "--initializer",
        type=str,
        choices=["lecun", "uniform", "variance_scaling"],
        default="lecun",
    )
    parser.add_argument(
        "--w", type=float, default=0.5, help="kernel initialization range for uniform init. Type: float, Default: 1.0"
    )
    parser.add_argument(
        "--b", type=float, default=0.5, help="Bias initialization range for uniform init. Type: float, Default: 0.0"
    )
    parser.add_argument(
        "--scale_w", type=float, default=1.0, help="Scaling factor for var_scaling init. Type: float, Default: 1.0"
    )
    parser.add_argument(
        "--scale_b", type=float, default=1.0, help="Scaling factor for var_scaling init. Type: float, Default: 1.0"
    )

    # Logging
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="test_project",
        help="Weights & Biases project name. Type: str, Default: 'bioflax'",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="bioflax",
        help="Weights & Biases entity name. Type: str, Default: 'bioflax'",
    )
    parser.add_argument(
        "--use_wandb",
        type=str2bool,
        default=True,
        help="Enable/disable Weights & Biases for logging. Type: boolean, Default: True",
    )
    parser.add_argument(
        "--plot",
        type=str2bool,
        default=False,
        help="Enable/disable plotting. Type: boolean, Default: False",
    )
    parser.add_argument(
        "--compute_alignments",
        type=str2bool,
        default=True,
        help="Enable/disable alignment computations. Type: boolean, Default: True",
    )
    parser.add_argument(
        "--n", type=int, default=5, help="Batches for alignment averaging. Type: int, Default: 5"
    )

    args = parser.parse_args()

    train(args)
