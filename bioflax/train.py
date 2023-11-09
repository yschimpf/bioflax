import wandb
from jax import random
from typing import Any
from .model import (
    BatchBioNeuralNetwork
)
from .train_helpers import (
    create_train_state,
    train_epoch,
    validate,
    plot_sample
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
    - compute weight alignment

"""


def train(args):  # (args):
    """
    Main function to train over a certain amount of epochs for model and data given by args
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    batch_size = args.batch_size
    val_split = args.val_split
    epochs = args.epochs
    mode = args.mode
    activations = args.activations
    hidden_layers = args.hidden_layers
    seed = args.jax_seed
    dataset = args.dataset
    in_dim = args.in_dim
    seq_len = args.seq_len
    output_features = args.output_features
    train_set_size = args.train_set_size
    test_set_size = args.test_set_size
    lr = args.lr
    momentum = args.momentum
    plot = args.plot
    compute_alignments = args.compute_grad_alignments
    project = args.wandb_project
    use_wandb = args.use_wandb
    n = args.n
    entity = args.wandb_entity

    key = random.PRNGKey(seed)
    task = "classification"  # args.task => as of now depends on dataset
    loss_fn = "CE"  # args.loss_fun => as of now depends on dataset

    if use_wandb:  # args.use_wandb:
        # Make wandb config dictionary
        wandb.init(
            project=project,
            job_type="model_training",
            config=vars(args),
            entity=entity,
        )
    else:
        wandb.init(mode="offline")

    # Set seed for randomness
    print("[*] Setting Randomness...")

    init_rng, key = random.split(key, num=2)

    # Create dataset
    (
        trainloader,
        valloader,
        testloader,
        output_features,
        seq_len,
        in_dim,
    ) = create_dataset(seed=42, batch_size=batch_size, dataset=dataset, val_split=val_split, input_dim=in_dim, output_dim=output_features, L=seq_len, train_set_size=train_set_size, test_set_size=test_set_size)  # (seed=args.jax_seed, batch_size=args.batch_size, dataset = args.dataset)
    print(f"[*] Starting training on mnist =>> '{dataset}' Initializing...")

    # model to run experiments with
    model = BatchBioNeuralNetwork(
        hidden_layers=hidden_layers,
        activations=activations,
        features=output_features,
        mode=mode,
    )

    state = create_train_state(
        model=model,
        rng=init_rng,
        lr=lr,
        momentum=momentum,
        in_dim=in_dim,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    init_rng_bp, key = random.split(key, num=2)

    bp_model = BatchBioNeuralNetwork(
        hidden_layers=hidden_layers,
        activations=activations,
        features=output_features,
        mode="bp",
    )

    bp_state = create_train_state(
        model=bp_model,
        rng=init_rng_bp,
        lr=lr,
        momentum=momentum,
        in_dim=in_dim,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, - \
        100000000.0, 0  # This best loss is val_loss
    for epoch in range(epochs):  # (args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        state, train_loss, avg_bias_al_per_layer, avg_wandb_grad_al_per_layer, avg_wandb_grad_al_total, avg_weight_al_per_layer, avg_rel_norm_grads = train_epoch(
            state, bp_model, trainloader, seq_len, in_dim, loss_fn, n, mode, compute_alignments
        )

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(
                state, valloader, seq_len, in_dim, loss_fn)

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(
                state, testloader, seq_len, in_dim, loss_fn)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} "
                f"-- Val Loss: {val_loss:.5f} "
                f"-- Test Loss: {test_loss:.5f}")
            if task == 'classification':
                print(
                    f"\tVal Accuracy: {val_acc:.4f} "
                    f"-- Test Accuracy: {test_acc:.4f} "
                )
            if compute_alignments:
                print(
                    f"\tRelative Norm Gradients: {avg_rel_norm_grads:.4f} "
                    f"-- Gradient Alignment: {avg_wandb_grad_al_total:.4f}"
                )

        else:
            # else use test set as validation set
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(
                state, testloader, seq_len, in_dim, loss_fn
            )

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}"
                f"-- Test Loss: {val_loss:.5f}"
            )
            if task == 'classification':
                print(f"-- Test Accuracy: {val_acc:.4f}\n")
            if compute_alignments:
                print(
                    f"\tRelative Norm Gradients: {avg_rel_norm_grads:.4f}"
                    f"-- Gradient Alignment: {avg_wandb_grad_al_total:.4f}"
                )

        if val_acc > best_acc:
            # Increment counters etc.
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc
        # Print best accuracy & loss so far...
        if task == 'regression':
            print(
                f"\tBest Val Loss: {best_loss:.5f} at Epoch {best_epoch + 1}\n"
                f"\tBest Test Loss: {best_test_loss:.5f} at Epoch {best_epoch + 1}\n"
            )
        elif task == 'classification':
            print(
                f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
                f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
                f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
                f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
            )

        metrics = {
            "Training Loss": train_loss,
            "Val Loss": val_loss,
        }

        if task == 'classification':
            metrics["Val Accuracy"]: val_acc
        if valloader is not None:
            metrics["Test Loss"] = test_loss
            if task == 'classification':
                metrics["Test Accuracy"] = test_acc

        if compute_alignments:
            metrics["Relative Norms Gradients"] = avg_rel_norm_grads
            metrics["Gradient Alignment"] = avg_wandb_grad_al_total
            for i, al in enumerate(avg_bias_al_per_layer):
                metrics[f"Alignment bias gradient layer {i}"] = al
            for i, al in enumerate(avg_wandb_grad_al_per_layer):
                metrics[f"Alignment gradient layer {i}"] = al
            if mode == "fa" or mode == "kp":
                for i, al in enumerate(avg_weight_al_per_layer):
                    metrics[f"Alignment layer {i}"] = al

        wandb.log(metrics)

        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        if task == 'classification':
            wandb.run.summary["Best Test Accuracy"] = best_test_acc
            wandb.run.summary["Best Val Accuracy"] = best_acc

    if (plot):
        plot_sample(testloader, state, seq_len, in_dim, task, output_features)
