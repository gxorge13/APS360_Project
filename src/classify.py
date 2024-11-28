from classifier import *
from transfer_learning import *
import numpy as np
import torch.nn as nn

# Initialize classifier and load weights
myClassifier = Classifier("model4")
model_path = f"./models/vgg_transfer_classifier/{myClassifier.name}_BS={256}_LR={0.001}_EP={13}"

myClassifier.load_state_dict(torch.load(model_path))
myClassifier.eval()  # Ensure the model is in evaluation mode

# Load class names
_, _, _, classes = get_data_loader(1)

# Define Softmax
sm = nn.Softmax(dim=-1)

# Function to classify an image
def classify_image(image_path):
   feats = get_features(image_path).unsqueeze(0)  # Prepare features
   probs = sm(myClassifier(feats))  # Get probabilities
   predicted_class = classes[probs[0].detach().numpy().argmax(0)]  # Detach and use NumPy
   return probs[0], predicted_class

# Process images
# probs, predicted_class = classify_image("./Data/ProcessedCancer/Cancer4238.png")
# print(probs, predicted_class)

# probs, predicted_class = classify_image("./Data/Normal/images/Normal-8958.png")
# print(probs, predicted_class)

# probs, predicted_class = classify_image("./Data/COVID/images/COVID-3608.png")
# print(probs, predicted_class)

state = None
while True:
   print("Current state: ", state)
   i = input("Command: ").lower()
   
   if (state == None):
      if (i == "q"):
         break
      state = i
      continue
   elif (i == "q"):
      state = None
      continue
      
   
   if (state == "cancer"):
      probs, predicted_class = classify_image(f"./Data/ProcessedCancer/Cancer{i}.png")
      print(probs, predicted_class)
   elif state == "normal":
      probs, predicted_class = classify_image(f"./Data/Normal/images/Normal-{i}.png")
      print(probs, predicted_class)
   elif state == "covid":
      probs, predicted_class = classify_image(f"./Data/COVID/images/COVID-{i}.png")
      print(probs, predicted_class)
   else:
      print("Invalid state")
      state = None