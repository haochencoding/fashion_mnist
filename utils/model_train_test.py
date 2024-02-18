import torch
import pandas as pd
from collections import defaultdict
import time
from torchmetrics import Precision, Recall, Accuracy, F1Score


def train(dataloader, model, optimizer, loss_fn, metric, device):
    train_loss = 0.0

    # Set the model to train mode
    model.train()
    # Clear the internal state of the metric
    metric.reset()

    for images, labels in dataloader:

        # Send data to device
        X, y = images.to(device), labels.to(device)

        # Zero gradients for each batch
        optimizer.zero_grad()

        # Compute loss
        outputs = model(X)
        loss = loss_fn(outputs, y)

        # Backpropagation 
        loss.backward()
        optimizer.step()

        # Store loss
        train_loss += loss.item()

        # Update the the metric with prediction of the batch
        _, preds = torch.max(outputs, 1)
        metric.update(preds, y)

    # Calculate average training loss during one epoch
    avg_train_loss = train_loss / len(dataloader)
    print(f'average training loss: {avg_train_loss}')

    # Calculate the metric over the whole epoch
    train_metric = metric.compute().item()
    print(f'train set score: {train_metric}')

    # return accuracy over the epoch
    return avg_train_loss, train_metric


def test(dataloader, model, loss_fn, metric, device, validate=True):
    # Set the model to evaluation mode
    model.eval()
    # Clear internal state of metric
    metric.reset()

    # Seepd up the forward pass by disabling gradient calculation
    with torch.no_grad():
        test_loss = 0.0
        for images, labels in dataloader:
            # Send data to CPU
            X, y = images.to(device), labels.to(device)

            # Compute prediction error
            outputs = model(X)
            loss = loss_fn(outputs, y)

            # Store loss
            test_loss += loss.item()

            # Compute accuracy
            # Update the the metric with prediction of the batch
            _, preds = torch.max(outputs, 1)
            metric.update(preds, y)

        if validate:
            situation = 'validation'
        else:
            situation = 'test'

        # Calculate average validation loss during one epoch
        avg_test_loss = test_loss / len(dataloader)
        print(f'average {situation} loss: {avg_test_loss}')

        # Calculate accuracy over the epoch
        test_metric = metric.compute().item()
        print(f'{situation} set score: {test_metric}')

    return avg_test_loss, test_metric


def train_validate_iteration(
        train_dataloader, val_dataloader,
        num_epochs,
        model, optimizer, loss_fn, metric,
        device
):
    # initialise empty dictionary to store loss and metrics
    epoch_summary = defaultdict(list)

    print(f'Total number of epochs: {num_epochs}')
    print("-------------------------------")
    epoch_summary['num_epochs'].append(num_epochs)

    # Record the start time
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'Training epoch: {epoch+1}')
        # train loop
        avg_train_loss, train_metric = train(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric=metric,
            device=device
            )
        epoch_summary['train_loss'].append(avg_train_loss)
        epoch_summary['train_score'].append(train_metric)

        # validation loop
        avg_val_loss, val_metric = test(
            dataloader=val_dataloader,
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            device=device,
            validate=True
            )
        epoch_summary['val_loss'].append(avg_val_loss)
        epoch_summary['val_score'].append(val_metric)

        print("-------------------------------")

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    epoch_summary['time'].append(elapsed_time)

    print(f"Time taken: {elapsed_time:.2f} seconds")

    return epoch_summary


classnames = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle Boot'
    ]


def metrics_across_classes(dataloader, model, device, classnames=classnames):
    # send model to device % set evaluation mode
    model.to(device)
    model.eval()

    # Initialize metrics
    precision_metric = Precision(
        task="multiclass", num_classes=len(classnames), average=None
        ).to(device)
    recall_metric = Recall(
        task="multiclass", num_classes=len(classnames), average=None
        ).to(device)
    f1_metric = F1Score(
        task="multiclass", num_classes=len(classnames), average=None
        ).to(device)
    accuracy_metric = Accuracy(
        task="multiclass", num_classes=len(classnames)
        ).to(device)

    # Initialize lists to store per-class metrics
    precisions = []
    recalls = []
    f1_scores = []
    supports = torch.zeros(len(classnames), dtype=torch.int64).to(device)

    # Loop over test set
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            # Update metrics and supports
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            accuracy_metric.update(preds, labels)
            for i in range(len(classnames)):
                supports[i] += torch.sum(labels == i)

    # Compute the final results
    precision_score = precision_metric.compute()
    recall_score = recall_metric.compute()
    f1_score = f1_metric.compute()
    accuracy_score = accuracy_metric.compute()

    # Store scores
    for i in range(len(classnames)):
        precisions.append(precision_score[i].item())
        recalls.append(recall_score[i].item())
        f1_scores.append(f1_score[i].item())

    # Create a DataFrame for report
    df_metrics = pd.DataFrame({
        'class': classnames,
        'precision': precisions,
        'recall': recalls,
        'f1-score': f1_scores,
        'support': supports.cpu().numpy()
    })

    # Calculate and append overall accuracy, macro avg, and weighted avg
    total_support = df_metrics['support'].sum()
    macro_avg_precision = df_metrics['precision'].mean()
    macro_avg_recall = df_metrics['recall'].mean()
    macro_avg_f1 = df_metrics['f1-score'].mean()
    weighted_avg_precision = (df_metrics['precision'] * df_metrics['support']).sum() / total_support
    weighted_avg_recall = (df_metrics['recall'] * df_metrics['support']).sum() / total_support
    weighted_avg_f1 = (df_metrics['f1-score'] * df_metrics['support']).sum() / total_support

    print(df_metrics)
    print('-'*50)
    print(f'Overall accuracy: {accuracy_score.item():.3f}')
    print(f"Macro average precision: {macro_avg_precision:.3f}, Recall: {macro_avg_recall:.3f}, F1: {macro_avg_f1:.3f}")
    print(f"Weighted average precision: {weighted_avg_precision:.3f}, Recall: {weighted_avg_recall:.3f}, F1: {weighted_avg_f1:.3f}")
