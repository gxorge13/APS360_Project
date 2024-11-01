from classifier import *

myClassifier = Classifier("model1")

train_net_with_features(myClassifier, batch_size = 256, learning_rate=0.001, num_epochs = 20)

myClassifier2 = Classifier("model2")

train_net_with_features(myClassifier2, batch_size = 256, learning_rate=0.001, num_epochs = 20)

plot_training_curve(f"./models/vgg_transfer_classifier/{myClassifier.name}_BS={256}_LR={0.001}_EP={19}")
plot_training_curve(f"./models/vgg_transfer_classifier/{myClassifier2.name}_BS={256}_LR={0.001}_EP={19}")