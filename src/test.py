from helper import *
from classifier import *

dir = "./models/vgg/features"
testDS = torchvision.datasets.DatasetFolder(f"{dir}/test",
                                               loader=torch.load,
                                               extensions=".tensor")
test_loader = torch.utils.data.DataLoader(testDS, 256, shuffle=True)

myClassifier = Classifier("model4")
model_path = f"./models/vgg_transfer_classifier/{myClassifier.name}_BS={256}_LR={0.001}_EP={13}"
myClassifier.load_state_dict(torch.load(model_path))

print(round(100*evaluate(myClassifier, test_loader, nn.CrossEntropyLoss())[0], 5))
