import jax
from jax import random
import jax.numpy as jnp
from tqdm import tqdm
from flax.training import train_state
import optax
from typing import Any
from .model import (
    #BioNeuralNetwork, 
    BatchBioNeuralNetwork
)
import wandb
from functools import partial
from .train_helpers import (
    create_train_state,
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

"""

def train(): #(args):
    """
    Main function to train over a certain amount of epochs for model and data given by args
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

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
    key = random.PRNGKey(0)#args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Create dataset
    init_rng, key = random.split(init_rng, num=2)
    (
        trainloader,
        testloader,
        output_features, #in case I let the output features be defined like this which makes sense I don't need to specify that as an argument in args anymore
        seq_len,
        in_dim,
        train_size,
    ) = create_dataset(seed=0, batch_size=32, dataset="mnist")#(seed=args.jax_seed, batch_size=args.batch_size, dataset = args.dataset)
    print(f"[*] Starting training on mnist =>> Initializing...") # `{args.dataset}` =>> Initializing...")

    # Initialize model
    model = partial(
        BatchBioNeuralNetwork,
        hidden_layers=["40", "40"], #args.hidden_layers,
        activations=["relu", "relu"], #args.activations,
        features = output_features,
        mode="bp", #args.mode,
    )

    state = create_train_state(
        model = model,
        rng = init_rng,
        lr = lr,
        momentum = momentum,
        in_dim = in_dim,
        batch_size = 32, #args.batch_size,
        seq_len = seq_len,
    )

    # Training Loop over epochs
    # best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    # steps_per_epoch = int(train_size / args.batch_size)
    # for epoch in range(args.epochs):
    #     print(f"[*] Starting Training Epoch {epoch + 1}...")
        
    #     state, train_loss = train_epoch(
    #         state, model, trainloader, seq_len, in_dim, args.norm, loss_fun
    #     )

    #     # HERE 25.10.2023 BZW. ich glaube es könnte hier wirklich besser sein hier vorerst mit 
    #     # den training loops von dem quickstart tutorial zu arbeiten. wäre wichtiger das bis 
    #     # abfahrt auf mnist zum laufen zu bringen. dementsprechend dann auch die datalaoder anpassen

    #     if valloader is not None:
    #         print(f"[*] Running Epoch {epoch + 1} Validation...")
    #         val_loss, val_acc = validate(state, model_cls, valloader, seq_len, in_dim, args.norm)

    #         print(f"[*] Running Epoch {epoch + 1} Test...")
    #         test_loss, test_acc = validate(state, model_cls, testloader, seq_len, in_dim, args.norm)

    #         print(f"\n=>> Epoch {epoch + 1} Metrics ===")
    #         print(
    #             f"\tTrain Loss: {train_loss:.5f} "
    #             f"-- Val Loss: {val_loss:.5f} "
    #             f"-- Test Loss: {test_loss:.5f}\n"
    #             f"\tVal Accuracy: {val_acc:.4f} "
    #             f"-- Test Accuracy: {test_acc:.4f}"
    #         )

    #     else:
    #         # else use test set as validation set (e.g. IMDB)
    #         print(f"[*] Running Epoch {epoch + 1} Test...")
    #         val_loss, val_acc = validate(state, model_cls, testloader, seq_len, in_dim, args.norm)

    #         print(f"\n=>> Epoch {epoch + 1} Metrics ===")
    #         print(
    #             f"\tTrain Loss: {train_loss:.5f}  -- Test Loss: {val_loss:.5f}\n"
    #             f"\tTest Accuracy: {val_acc:.4f}"
    #         )

    #     # For early stopping purposes
    #     if val_loss < best_val_loss:
    #         count = 0
    #         best_val_loss = val_loss
    #     else:
    #         count += 1

    #     if val_acc > best_acc:
    #         # Increment counters etc.
    #         count = 0
    #         best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
    #         if valloader is not None:
    #             best_test_loss, best_test_acc = test_loss, test_acc
    #         else:
    #             best_test_loss, best_test_acc = best_loss, best_acc

    #     # For learning rate decay purposes:
    #     input = lr, ssm_lr, lr_count, val_acc, opt_acc
    #     lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(
    #         input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min
    #     )

    #     # Print best accuracy & loss so far...
    #     print(
    #         f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
    #         f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
    #         f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
    #         f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
    #     )

    #     metrics = {
    #         "Training Loss": train_loss,
    #         "Val Loss": val_loss,
    #         "Val Accuracy": val_acc,
    #         "Count": count,
    #         "Learning rate count": lr_count,
    #         "Opt acc": opt_acc,
    #         "lr": state.opt_state.inner_states["regular"].inner_state.hyperparams["learning_rate"],
    #         "ssm_lr": state.opt_state.inner_states["ssm"].inner_state.hyperparams["learning_rate"],
    #     }
    #     if valloader is not None:
    #         metrics["Test Loss"] = test_loss
    #         metrics["Test Accuracy"] = test_acc
    #     wandb.log(metrics)

    #     wandb.run.summary["Best Val Loss"] = best_loss
    #     wandb.run.summary["Best Val Accuracy"] = best_acc
    #     wandb.run.summary["Best Epoch"] = best_epoch
    #     wandb.run.summary["Best Test Loss"] = best_test_loss
    #     wandb.run.summary["Best Test Accuracy"] = best_test_acc

    #     if count > args.early_stop_patience:
    #         break









"""

@jax.jit
def train_step(state, rng, inputs, labels, model, classification=False):

    def loss_fn(params):
        p = {"params": params}
        vars = None
        logits = model.apply(p, inputs)
        return logits

    (loss, vars), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)

    if norm in ["batch"]:
        state = state.apply_gradients(grads=grads, batch_stats=vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return state, loss


def train_epoch(state, rng, model, trainloader, seq_len, in_dim, norm, lr_params):
    model = model(training=True)  # model in training mode
    batch_losses = []
    decay_function, ssm_lr, lr, step, end_step, lr_min = lr_params
    print(“length train loader”, len(trainloader))
    for batch in trainloader:
        inputs, labels, masks = prep_batch(batch, seq_len, in_dim)
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(
            state, drop_rng, inputs, labels, masks, model, norm, model.classification
        )
        batch_losses.append(loss)  # log loss value
        lr_params = (decay_function, ssm_lr, lr, step, end_step, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)
    # Return average loss over batches
    return state, jnp.mean(jnp.array(batch_losses)), step
"""