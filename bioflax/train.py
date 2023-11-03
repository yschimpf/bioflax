import jax
from jax import random
import jax.numpy as jnp
from tqdm import tqdm
from flax.training import train_state
import optax
from typing import Any
from .model import (
    BioNeuralNetwork, 
    BatchBioNeuralNetwork
)
import wandb
from functools import partial
from .train_helpers import (
    create_train_state,
    train_epoch,
    validate,
    plot_mnist_sample,
    plot_regression_sample
)
from .dataloading import (
    create_dataset
)


"""
desired logging: log loss (accuracy classification), gradient alignment ⇒ 
full one and per layer (requires side modem with backprop), the relative 
norm of the gradients, alignment of matrices ⇒ do so in the training file 
(!not per layer) → if it takes to much place in a separate logging.py


args needs: 
    - args.lr (learning rate)
    - args.use_wandb (boolean)
    - args.wandb_project (stirng of project name)
    - args.wandb_entity (string of entity name)
    - args.jax_seed (int)
    - args.dir_name (string of directory name)
    - args.batch_size (int)
    - args.dataset (string of dataset name)
    - args.hidden_layers (list of ints)
    - args.activations (list of strings)
    - args.mode (string of mode name)
    - args.momentum (float)
    - args.epochs (int)
    - args.loss_fun (string of loss function name)
    - args.val_split (float) (0. corresponds to val_set = None)

"""

def train(): #(args):
    """
    Main function to train over a certain amount of epochs for model and data given by args
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    # parameter initialization
    batch_size = 32 #args.batch_size
    loss_fn = "MSE" #args.loss_fun
    val_split = 0.1 #args.val_split
    epochs = 500 #args.epochs
    mode = "dfa" #args.mode
    activations = ["sigmoid", "sigmoid"] #args.activations
    hidden_layers = [64,64] #args.hidden_layers
    key = random.PRNGKey(0)#args.jax_seed)
    dataset="sinprop" #args.dataset
    task = "regression" #args.task
    in_dim = 1 #args.in_dim (dim des vectors)
    seq_len = 1 #args.seq_len (länge des vectors)
    output_features = 1 #args.output_features
    train_set_size = 50 #args.input_size => size of the teacher/sin train set to generate
    test_set_size = 10 #args.input_size => size of the teacher/sin test set to generate


    if False: #args.use_wandb:
        # Make wandb config dictionary
        wandb.init(
            project=args.wandb_project,
            job_type="model_training",
            config=vars(args),
            entity=args.wandb_entity,
        )
    else:
        wandb.init(mode="offline")

    lr = 0.1#args.lr # learning rate
    momentum = 0#args.momentum # momentum

    # Set seed for randomness
    print("[*] Setting Randomness...")
    
    init_rng, train_rng = random.split(key, num=2)

    # Create dataset
    init_rng, key = random.split(init_rng, num=2)
    (
        trainloader,
        valloader,
        testloader,
        train_size,
        output_features, #in case I let the output features be defined like this which makes sense I don't need to specify that as an argument in args anymore
        seq_len,
        in_dim,
    ) = create_dataset(seed=42, batch_size=32, dataset=dataset, val_split=val_split, input_dim=in_dim, output_dim=output_features, L=seq_len, train_set_size=train_set_size, test_set_size=test_set_size)#(seed=args.jax_seed, batch_size=args.batch_size, dataset = args.dataset)
    print(f"[*] Starting training on mnist =>> Initializing...") # `{args.dataset}` =>> Initializing...")

    # Initialize model
    model = BatchBioNeuralNetwork(
        hidden_layers=hidden_layers, #args.hidden_layers,
        activations=activations, #args.activations,
        features = output_features,
        mode=mode, #args.mode,
    )
    #print(model)

    state = create_train_state(
        model = model,
        rng = init_rng,
        lr = lr,
        momentum = momentum,
        in_dim = in_dim,
        batch_size = batch_size, #args.batch_size,
        seq_len = seq_len,
    )

    #print(state)

    #Training Loop over epochs (bis hierhin hat mal alles funktioniert)
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    steps_per_epoch = int(train_size / batch_size)
    for epoch in range(epochs): #(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        
        state, train_loss = train_epoch(
            state, model, trainloader, seq_len, in_dim, loss_fn
        )

        # HERE 25.10.2023 BZW. ich glaube es könnte hier wirklich besser sein hier vorerst mit 
        # den training loops von dem quickstart tutorial zu arbeiten. wäre wichtiger das bis 
        # abfahrt auf mnist zum laufen zu bringen. dementsprechend dann auch die datalaoder anpassen

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(state, valloader, seq_len, in_dim, loss_fn)

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(state, testloader, seq_len, in_dim, loss_fn)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} "
                f"-- Val Loss: {val_loss:.5f} "
                f"-- Test Loss: {test_loss:.5f}\n"
                f"\tVal Accuracy: {val_acc:.4f} "
                f"-- Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(state, testloader, seq_len, in_dim, loss_fn)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  -- Test Loss: {val_loss:.5f}\n"
                f"\tTest Accuracy: {val_acc:.4f}"
            )

        # For early stopping purposes
        # if val_loss < best_val_loss:
        #     count = 0
        #     best_val_loss = val_loss
        # else:
        #     count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc
        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        metrics = {
            "Training Loss": train_loss,
            "Val Loss": val_loss,
            "Val Accuracy": val_acc,
            "Count": count,
        }
        if valloader is not None:
            metrics["Test Loss"] = test_loss
            metrics["Test Accuracy"] = test_acc
        wandb.log(metrics)

        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Val Accuracy"] = best_acc
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        wandb.run.summary["Best Test Accuracy"] = best_test_acc
    
    if(task == "classification"):
        plot_mnist_sample(testloader, state, seq_len, in_dim, task)
    elif (task == "regression"):
        plot_regression_sample(testloader, state, seq_len, in_dim, task)
    