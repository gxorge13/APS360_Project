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
 
 
def get_features(path, num_cols=8, plot_size=3):
  from PIL import Image
  
  image = Image.open(path)
  
  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
  # Convert to RGB if the image is grayscale (single channel)
  if image.mode != 'RGB':
    image = image.convert('RGB')
  tensor = transform(image)
  
  conv = nn.Sequential(*list(VGG.children())[0])
  feature_maps = conv(tensor)
  if feature_maps.dim() == 4:
      feature_maps = feature_maps.squeeze(0)  # Shape (num_channels, height, width)

  num_channels = feature_maps.shape[0]
  num_rows = (num_channels + num_cols - 1) // num_cols  # Calculate rows needed for grid

  # Set up the figure with a larger figsize
  plt.figure(figsize=(num_cols * plot_size, num_rows * plot_size))
  for i in range(num_channels):
      plt.subplot(num_rows, num_cols, i + 1)
      plt.imshow(feature_maps[i].detach().cpu().numpy(), cmap="gray")
      plt.axis("off")
  plt.show()
  
  
get_features("./Data/Normal/images/Normal-1.png", 25, 4)