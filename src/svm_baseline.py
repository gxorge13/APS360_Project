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

    kernel = 'poly'
    C = 0.1
    gamma = 10

    # Create and train the model
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma)    
    model.fit(train_features, train_labels)
    print(f"Model trained with kernel={kernel}, C={C}, gamma={gamma}")

    return model

def train_svm_baseline():

    start_time = time.time()
    # Load the data
    train_loader, val_loader, test_loader, target_classes = get_data_loader(batch_size=256)

    # Get the model
    model = svm_baseline_classifier(train_loader, val_loader, test_loader, target_classes)

    

    # Evaluate on train, validation, and test sets
    train_err, train_acc = evaluate_svm(model, train_loader)
    val_err, val_acc = evaluate_svm(model, val_loader)
    test_err, test_acc = evaluate_svm(model, test_loader)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    # Accuracy values
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Error values
    print(f"Train error: {train_err:.4f}")
    print(f"Validation error: {val_err:.4f}")
    print(f"Test error: {test_err:.4f}")

    
    return train_acc, val_acc, test_acc

# Run the training
train_svm_baseline()


