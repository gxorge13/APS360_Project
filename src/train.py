from classifier import *

myClassifier = Classifier("model4")

# train_net_with_features(myClassifier, batch_size = 256, learning_rate=0.001, num_epochs = 25)

plot_training_curve(f"./models/vgg_transfer_classifier/{myClassifier.name}_BS={256}_LR={0.001}_EP={24}")

# C:\Users\georg\OneDrive\Desktop\SideProjects\APS360_Project\models\vgg_transfer_classifier\model3_BS=256_LR=0.001_EP=19