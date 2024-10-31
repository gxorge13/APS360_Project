from imports import *
from helper import *
import torchvision.models

save_path = "./models/vgg/features"
VGG = torchvision.models.vgg16(pretrained=True)

def save_features():
   train_loader, val_loader, test_loader, classes = get_data_loader(batch_size=1)

   feature_count = [0 for i in range(len(classes))]
   for img, l in train_loader:
     dir = f"{save_path}/train/{classes[l]}"
     feat = VGG.features(img)
     if not os.path.isdir(dir):
       os.makedirs(dir, exist_ok=True)

     torch.save(feat, f"{dir}/feature_{feature_count[l]}.tensor")
     feature_count[l]+=1
     
   print(f"Number of features: {len(train_loader), len(val_loader), len(test_loader)}")
   feature_count = [0 for i in range(len(classes))]
   for img, l in val_loader:
     dir = f"{save_path}/val/{classes[l]}"
     feat = VGG.features(img)
     if not os.path.isdir(dir):
       os.makedirs(dir, exist_ok=True)

     torch.save(feat, f"{dir}/feature_{feature_count[l]}.tensor")
     feature_count[l]+=1

   feature_count = [0 for i in range(len(classes))]
   for img, l in test_loader:
     dir = f"{save_path}/test/{classes[l]}"
     feat = VGG.features(img)
     if not os.path.isdir(dir):
         os.makedirs(dir, exist_ok=True)
     torch.save(feat, f"{dir}/feature_{feature_count[l]}.tensor")
     feature_count[l]+=1
     
def get_last_layer():
   conv = nn.Sequential(*list(VGG.children())[0])
   print(conv)

   img = torch.randn(1, 3, 299, 299)

   return conv(img).shape