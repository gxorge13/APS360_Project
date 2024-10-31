from classifier import *

myClassifier = Classifier("model1")

train_net_with_features(myClassifier, batch_size = 256, learning_rate=0.01, num_epochs = 50)

plot_training_curve(f"./models/vgg_transfer_classifier/{myClassifier.name}_BS={256}_LR={0.01}_EP={50}")