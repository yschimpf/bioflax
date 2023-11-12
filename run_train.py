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

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--val_split", type=float,
                        default=0.1, help="Validation split ratio.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs for training.")
    parser.add_argument("--mode", type=str, default="fa",
                        choices=['bp', 'fa', 'kp', 'dfa'], help="Mode for training. Choices: ['bp', 'fa', 'kp', 'dfa'].")
    parser.add_argument("--activations", nargs='+', type=str, default=[
                        "relu", "relu"], help="List of activation functions for each hidden layer.")
    parser.add_argument("--hidden_layers", nargs='+', type=int, default=[
                        500, 500], help="List of the number of neurons for each hidden layer.")
    parser.add_argument("--jax_seed", type=int, default=0,
                        help="Seed for JAX random number generator.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=['mnist', 'sinprop', 'teacher'],
                        help="Dataset to use for training. Choices: ['mnist', 'sinprop', 'teacher'].")
    parser.add_argument("--in_dim", type=int, default=1,
                        help="Dimension of input vector.")
    parser.add_argument("--seq_len", type=int, default=1,
                        help="Length of input sequence.")
    parser.add_argument("--output_features", type=int,
                        default=10, help="Number of output features.")
    parser.add_argument("--train_set_size", type=int, default=50,
                        help="Size of the training set to generate.")
    parser.add_argument("--test_set_size", type=int, default=10,
                        help="Size of the test set to generate.")
    parser.add_argument("--lr", type=float, default=0.02,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--momentum", type=float, default=0,
                        help="Momentum for the optimizer.")
    parser.add_argument("--weight_decay", type=float,
                        default=0, help="Weight decay for the optimizer.")
    parser.add_argument("--plot", type=str2bool, default=True,
                        help="Flag to plot the results.")
    parser.add_argument("--compute_alignments", type=str2bool, default=True,
                        help="Flag to compute gradient alignments.")
    parser.add_argument("--wandb_project", type=str,
                        default="test_project", help="Weights & Biases project name.")
    parser.add_argument("--use_wandb", type=str2bool, default=True,
                        help="Flag to use Weights & Biases for logging.")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of batches over which to average alignments.")
    parser.add_argument("--wandb_entity", type=str, default="bioflax",
                        help="Weights & Biases entity name.")

    args = parser.parse_args()

    print("Start")
    train(args)
    print("End")
