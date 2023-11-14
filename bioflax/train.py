import wandb
from jax import random
from typing import Any
from .model import BatchBioNeuralNetwork
from .train_helpers import create_train_state, train_epoch, validate, plot_sample
from .dataloading import create_dataset


def train(args):
    """
    Main function for training and eveluating a biomodel. Training and evaluation set up by
    arguments passed in args.
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
    teacher_act = args.teacher_act
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    plot = args.plot
    compute_alignments = args.compute_alignments
    project = args.wandb_project
    use_wandb = args.use_wandb
    n = args.n

    if dataset == "mnist":
        task = "classification"
        loss_fn = "CE"
        output_features = 10
    else:
        task = "regression"
        loss_fn = "MSE"

    if dataset == "sinreg":
        output_features = 1

    if use_wandb:
        # Make wandb config dictionary
        wandb.init(
            project=project,
            job_type="model_training",
            config=vars(args),
        )

    # Set seed for randomness
    print("[*] Setting Randomness...")
    key = random.PRNGKey(seed)
    key_data, key_model, key_model_bp = random.split(key, num=3)

    # Create dataset
    (
        trainloader,
        valloader,
        testloader,
        output_features,
        seq_len,
        in_dim,
    ) = create_dataset(
        seed=key_data[0].item(),
        batch_size=batch_size,
        dataset=dataset,
        val_split=val_split,
        input_dim=in_dim,
        output_dim=output_features,
        L=seq_len,
        train_set_size=train_set_size,
        test_set_size=test_set_size,
        teacher_act=teacher_act,
    )
    print(f"[*] Starting training on '{dataset}'...")

    # Model to run experiments with
    model = BatchBioNeuralNetwork(
        hidden_layers=hidden_layers,
        activations=activations,
        features=output_features,
        mode=mode,
    )
    state = create_train_state(
        model=model,
        rng=key_model,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        in_dim=in_dim,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    # Backpropagation model to compute alignments
    bp_model = BatchBioNeuralNetwork(
        hidden_layers=hidden_layers,
        activations=activations,
        features=output_features,
        mode="bp",
    )
    _ = create_train_state(
        model=bp_model,
        rng=key_model_bp,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        in_dim=in_dim,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    # Training loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    for epoch in range(epochs):  # (args.epochs):
        print(f"[*] Starting training epoch {epoch + 1}...")
        (
            state,
            train_loss,
            avg_bias_al_per_layer,
            avg_wandb_grad_al_per_layer,
            avg_wandb_grad_al_total,
            avg_weight_al_per_layer,
            avg_rel_norm_grads,
        ) = train_epoch(state, bp_model, trainloader, loss_fn, n, mode, compute_alignments)

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(state, valloader, seq_len, in_dim, loss_fn)

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(state, testloader, seq_len, in_dim, loss_fn)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} "
                f"-- Val Loss: {val_loss:.5f} "
                f"-- Test Loss: {test_loss:.5f}"
            )
            if task == "classification":
                print(f"\tVal Accuracy: {val_acc:.4f} " f"-- Test Accuracy: {test_acc:.4f} ")
        else:
            # Use test set as validation set
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(state, testloader, seq_len, in_dim, loss_fn)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(f"\tTrain loss: {train_loss:.5f}" f"-- Test loss: {val_loss:.5f}")
            if task == "classification":
                print(f"-- Test accuracy: {val_acc:.4f}\n")

        if compute_alignments:
            print(
                f"\tRelative norm gradients: {avg_rel_norm_grads:.4f} "
                f"-- Gradient alignment: {avg_wandb_grad_al_total:.4f}"
            )

        if val_acc > best_acc:
            # Increment counters etc.
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

        # Print best accuracy & loss so far...
        if task == "regression":
            print(
                f"\tBest val loss: {best_loss:.5f} at epoch {best_epoch + 1}\n"
                f"\tBest test loss: {best_test_loss:.5f} at epoch {best_epoch + 1}\n"
            )
        elif task == "classification":
            print(
                f"\tBest val loss: {best_loss:.5f} -- Best val accuracy:"
                f" {best_acc:.4f} at epoch {best_epoch + 1}\n"
                f"\tBest test loss: {best_test_loss:.5f} -- Best test accuracy:"
                f" {best_test_acc:.4f} at epoch {best_epoch + 1}\n"
            )

        metrics = {
            "Training loss": train_loss,
            "Val loss": val_loss,
        }

        if task == "classification":
            metrics["Val accuracy"]: val_acc
        if valloader is not None:
            metrics["Test loss"] = test_loss
            if task == "classification":
                metrics["Test accuracy"] = test_acc

        if compute_alignments:
            metrics["Relative norms gradients"] = avg_rel_norm_grads
            metrics["Gradient alignment"] = avg_wandb_grad_al_total
            for i, al in enumerate(avg_bias_al_per_layer):
                metrics[f"Alignment bias gradient layer {i}"] = al
            for i, al in enumerate(avg_wandb_grad_al_per_layer):
                metrics[f"Alignment gradient layer {i}"] = al
            if mode == "fa" or mode == "kp":
                for i, al in enumerate(avg_weight_al_per_layer):
                    metrics[f"Alignment layer {i}"] = al

        if use_wandb:
            wandb.log(metrics)
            wandb.run.summary["Best val loss"] = best_loss
            wandb.run.summary["Best epoch"] = best_epoch
            wandb.run.summary["Best test loss"] = best_test_loss
            if task == "classification":
                wandb.run.summary["Best test accuracy"] = best_test_acc
                wandb.run.summary["Best val accuracy"] = best_acc

    if plot:
        plot_sample(testloader, state, seq_len, in_dim, task, output_features)
