from imports import *
from helper import *

import numpy as np
from sklearn import svm

def evaluate_svm(model, loader):
    total_err = 0.0
    total_epoch = 0

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs = inputs.reshape(inputs.shape[0], -1)
        
        outputs = model.predict(inputs)
        corr = outputs != labels.numpy()

        total_err += int(corr.sum())
        total_epoch += len(labels)

    err = float(total_err) / total_epoch
    accuracy = 1 - err
    return err, accuracy

def svm_baseline_classifier(train_loader, val_loader, test_loader, target_classes):
    # Load and flatten the training data
    train_features, train_labels = next(iter(train_loader))
    train_features = train_features.reshape(train_features.shape[0], -1)

    # Create and train the model
    model = svm.SVC(kernel='linear', C=1)
    model.fit(train_features, train_labels)

    # Evaluate on train, validation, and test sets
    train_err, train_acc = evaluate_svm(model, train_loader)
    val_err, val_acc = evaluate_svm(model, val_loader)
    test_err, test_acc = evaluate_svm(model, test_loader)

    return train_acc, val_acc, test_acc

def train_svm_baseline():
    # Load the data
    train_loader, val_loader, test_loader, target_classes = get_data_loader(batch_size=256)
    
    # Train the model and get accuracies
    train_acc, val_acc, test_acc = svm_baseline_classifier(train_loader, val_loader, test_loader, target_classes)
    
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    return train_acc, val_acc, test_acc

# Run the training
train_svm_baseline()

# Train accuracy: 0.6498
# Validation accuracy: 0.6561
# Test accuracy: 0.6485