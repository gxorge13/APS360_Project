from imports import *
from helper import * 
from classifier import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def compute_confusion_matrix(net, dataloader, classes):
   net.eval() 
   all_preds = []
   all_labels = []

   with torch.no_grad(): 
      for inputs, labels in dataloader:
         inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
         labels = labels.to("cuda" if torch.cuda.is_available() else "cpu")
         
         outputs = net(inputs)
         
         _, preds = torch.max(outputs, 1)
         
         all_preds.append(preds.cpu().numpy())
         all_labels.append(labels.cpu().numpy())
   
   all_preds = np.concatenate(all_preds)
   all_labels = np.concatenate(all_labels)
   
   cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
   return cm


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
   disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
   disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
   plt.title(title)
   plt.show()


if __name__ == "__main__":
   dir = "./models/vgg/features"
   testDS = torchvision.datasets.DatasetFolder(f"{dir}/test",
                                                loader=torch.load,
                                                extensions=".tensor")
   test_loader = torch.utils.data.DataLoader(testDS, 256, shuffle=True)
   *_, classes = get_data_loader(1) 

   net = Classifier("model3")
   model_path = f"./models/vgg_transfer_classifier/{net.name}_BS={256}_LR={0.001}_EP={19}"
   net.load_state_dict(torch.load(model_path))
   net.to("cuda" if torch.cuda.is_available() else "cpu")

   cm = compute_confusion_matrix(net, test_loader, classes)
   plot_confusion_matrix(cm, classes, title="Confusion Matrix")
