from imports import *
from helper import *
import torchvision.models
from concurrent.futures import ThreadPoolExecutor

save_path = "./models/vgg/features"
VGG = torchvision.models.vgg16(pretrained=True)

def save_features_for_loader(loader, classes, save_path, split):
    feature_count = [0 for _ in range(len(classes))]

    total = len(loader)
    count = 0
    print(f"Saving began for split: {split} 0/{total}")
    for imgs, labels in loader:  # Process batches
        feats = VGG.features(imgs)  # Extract features for the entire batch

        for i in range(imgs.size(0)):  # Iterate through the batch
            class_label = labels[i].item()
            class_dir = f"{save_path}/{split}/{classes[class_label]}"
            os.makedirs(class_dir, exist_ok=True)

            # Save feature for the i-th image in the batch
            torch.save(feats[i], f"{class_dir}/feature_{feature_count[class_label]}.tensor")
            feature_count[class_label] += 1
        count += 1
        print(f"Batch done for split: {split} {count}/{total}")

    print(f"Finished saving features for {split} set.")


def save_features_parallel(batch_size=16):
    # Load data loaders
    train_loader, val_loader, test_loader, classes = get_data_loader(batch_size=batch_size)
    print(f"Number of features: {len(train_loader)}, {len(val_loader)}, {len(test_loader)}")
    
    # Define save path
    # splits = {"train": train_loader, "val": val_loader, "test": test_loader}
    splits = { "val": val_loader, "test": test_loader}

    # Use ThreadPoolExecutor to parallelize split-level processing
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(save_features_for_loader, loader, classes, save_path, split)
            for split, loader in splits.items()
        ]
        # Wait for all tasks to complete
        for future in futures:
            future.result()

    print("Feature saving complete for all splits.")

def get_last_layer():
   conv = nn.Sequential(*list(VGG.children())[0])
   print(conv)

   img = torch.randn(1, 3, 299, 299)

   return conv(img).shape
 
 
def get_features(path):  
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
  return feature_maps
  

def plot_features(features, num_cols=8, plot_size=3):
  from PIL import Image

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
  
# get_features("./Data/Normal/images/Normal-1.png", 25, 4)
# save_features()