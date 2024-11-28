from imports import *
from helper import * 
from classifier import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score

def eval_net(net, dataloader, classes):
   #eval_net
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

   accuracy = accuracy_score(all_labels, all_preds)
   f1 = f1_score(all_labels, all_preds, average=None)  
   cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
   precision = precision_score(all_labels, all_preds, average=None) 
   recall = recall_score(all_labels, all_preds, average=None)  
   return cm, f1, accuracy, precision, recall


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
   plt.figure(figsize=(10, 10))  # Adjust the width and height as needed
   disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
   disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=plt.gca())
   plt.title(title)
   plt.show()


if __name__ == "__main__":
   dir = "./models/vgg/features"
   # testDS = torchvision.datasets.DatasetFolder(f"{dir}/test",
   #                                              loader=torch.load,
   #                                              extensions=".tensor")
   
   valDS = torchvision.datasets.DatasetFolder(f"{dir}/val",
                                                loader=torch.load,
                                                extensions=".tensor")
   val_loader = torch.utils.data.DataLoader(valDS, 256, shuffle=True)
   *_, classes = get_data_loader(1) 

   net = Classifier("model3")
   model_path = f"./models/vgg_transfer_classifier/{net.name}_BS={256}_LR={0.001}_EP={19}"
   net.load_state_dict(torch.load(model_path))
   net.to("cuda" if torch.cuda.is_available() else "cpu")

   cm, f1, accuracy, precision, recall = eval_net(net, val_loader, classes)
   plot_confusion_matrix(cm, classes, title="Confusion Matrix")
   print(f"f1 score: {f1}, accuracy: {accuracy}, precision: {precision}, recall: {recall}")

#f1 score: [0.85522788 0.87128713 0.96818811], accuracy: 0.8979779411764706, precision: [0.85066667 0.87749288 0.96685083], recall: [0.85983827 0.86516854 0.96952909]