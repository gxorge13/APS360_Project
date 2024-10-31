from imports import *

global_path = "./Data/"

def is_file_valid(filepath):
  # List folders you want to ignore
  ignore_folders = ["masks"]
  supported_extensions = ("jpg", "jpeg", "png", "ppm", "bmp",
                          "pgm", "tif", "tiff", "webp")

  makeup = filepath.split(os.sep)
  for folder in ignore_folders:
      if folder in makeup:
          return False
  return makeup[-1].split(".")[-1] in supported_extensions

def get_relevant_indices(dataset, classes, target_classes):
  """ Returns indices of data that exist in target_classes """
  new_idx = {cls: idx for idx, cls in enumerate(target_classes)}
  indices = []
  for i, (_, label_idx) in enumerate(dataset.samples):
      class_label = classes[label_idx]
      if class_label in target_classes:
          indices.append((i, new_idx[class_label]))

  return indices

def get_data_loader(batch_size):
  np.random.seed(1000)

  # List of target classes
  classes = ("Lung_Opacity", "Normal", "COVID", "Viral Pneumonia")
  target_classes = ("Lung_Opacity", "Normal", "COVID")

  # Transforms applied to samples
  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  # Load the images from folder
  dataset = datasets.ImageFolder(global_path,
                                 transform,
                                 is_valid_file=is_file_valid)

  # Grab the indices
  relevant_indices, remapped_labels = zip(*get_relevant_indices(dataset, classes,
                                                     target_classes))
  
  relevant_indices = np.array(relevant_indices)

  # Shuffle the indices
  np.random.shuffle(relevant_indices)

  # Set up split 95% for train and val, 5% for test
  trainval_test_split = int(len(relevant_indices)*0.85)

  # 80% of 95% for train, 20% for validation
  train_val_split = int(trainval_test_split*0.82)

  train_indices = relevant_indices[:train_val_split]
  val_indices = relevant_indices[train_val_split:trainval_test_split]
  test_indices = relevant_indices[trainval_test_split:]
    
  # Get loaders
  train_sampler = SubsetRandomSampler(train_indices)
  train_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             sampler=train_sampler)

  val_sampler = SubsetRandomSampler(val_indices)
  val_sampler = SubsetRandomSampler(val_indices)
  val_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           num_workers=0,
                                           sampler=val_sampler)

  test_sampler = SubsetRandomSampler(test_indices)
  test_sampler = SubsetRandomSampler(test_indices)
  test_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            num_workers=0,
                                            sampler=test_sampler)

  return train_loader, val_loader, test_loader, target_classes

def evaluate(net, loader, criterion):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   net.to(device) 
   total_loss = 0.0
   total_err = 0.0
   total_epoch = 0   
   
   for i, data in enumerate(loader, 0):
      inputs, labels = data
      labels = labels.long() # labels to values
      inputs, labels = inputs.to(device), labels.to(device)  
   
      outputs = net(inputs) 
   
      loss = criterion(outputs, labels.long())
      corr = outputs.argmax(dim=1) != labels  
   
      total_err += int(corr.sum())
      total_loss += loss.item()
      total_epoch += len(labels)  
   
   err = float(total_err) / total_epoch
   loss = float(total_loss) / (i + 1)
   return err, loss

def plot_training_curve(path):
  """ Plots the training curve for a model run, given the csv files
  containing the train/validation error/loss.
  Args:
  path: The base path of the csv files produced during training
  """
  

  # Load data
  train_err = np.loadtxt("{}_train_err.csv".format(path))
  val_err = np.loadtxt("{}_val_err.csv".format(path))
  train_loss = np.loadtxt("{}_train_loss.csv".format(path))
  val_loss = np.loadtxt("{}_val_loss.csv".format(path))

  plt.title("Train vs Validation Error") # Set title
  n = len(train_err) # number of epochs
  plt.plot(range(1,n+1), train_err, label="Train")
  plt.plot(range(1,n+1), val_err, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Error")
  plt.legend(loc='best')
  plt.show()

  plt.title("Train vs Validation Loss")
  plt.plot(range(1,n+1), train_loss, label="Train")
  plt.plot(range(1,n+1), val_loss, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc='best')
  plt.show()

def train_net_with_features(net, batch_size=64, learning_rate=0.01, num_epochs=30):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   net.to(device)
   ########################################################################
   # Fixed PyTorch random seed for reproducible result
   torch.manual_seed(1000)
   ########################################################################
   # Obtain the PyTorch data loader objects to load batches of the datasets
   dir = "./models/vgg/features"
   trainDS = torchvision.datasets.DatasetFolder(f"{dir}/train",
                                               loader=torch.load,
                                               extensions=".tensor")
   valDS = torchvision.datasets.DatasetFolder(f"{dir}/val",
                                               loader=torch.load,
                                               extensions=".tensor")
   testDS = torchvision.datasets.DatasetFolder(f"{dir}/test",
                                               loader=torch.load,
                                               extensions=".tensor")
   
   train_loader = torch.utils.data.DataLoader(trainDS, batch_size, shuffle=True)
   val_loader = torch.utils.data.DataLoader(valDS, batch_size, shuffle=True)
   test_loader = torch.utils.data.DataLoader(testDS, batch_size, shuffle=True)
   
   categories = os.listdir(f"{dir}/train")
   ########################################################################
   # using cross entropy loss as this is a multiclassification problem
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
   
   ########################################################################
   # Set up some numpy arrays to store the training/test loss/erruracy
   train_err = np.zeros(num_epochs)
   train_loss = np.zeros(num_epochs)
   val_err = np.zeros(num_epochs)
   val_loss = np.zeros(num_epochs)
   
   #######################################################################
   # Set up dir
   dir = f"./models/vgg_transfer_classifier"
   if not os.path.isdir(dir):
       os.makedirs(dir, exist_ok=True)
   ########################################################################
   
   # Train the network
   # Loop over the data iterator and sample a new batch of training data
   # Get the output from the network, and optimize our loss function.
   start_time = time.time()
   for epoch in range(num_epochs):  # loop over the dataset multiple times
      total_train_loss = 0.0
      total_train_err = 0.0
      total_epoch = 0
      for i, data in enumerate(train_loader, 0):
         # Get the inputs
         inputs, labels = data
         labels = labels.long()
         inputs, labels = inputs.to(device), labels.to(device)
         # Zero the parameter gradients
         optimizer.zero_grad()
         # Forward pass, backward pass, and optimize
         outputs = net(inputs)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()
         # Calculate the statistics
         corr = outputs.argmax(dim=1) != labels
         total_train_err += int(corr.sum())
         total_train_loss += loss.item()
         total_epoch += len(labels)
      train_err[epoch] = float(total_train_err) / total_epoch
      train_loss[epoch] = float(total_train_loss) / (i+1)
      val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion,
                                                 device)
      print(("Epoch {}: Train err: {}, Train loss: {} |"+
              "Validation err: {}, Validation loss: {}").format(
                  epoch + 1,
                  train_err[epoch],
                  train_loss[epoch],
                  val_err[epoch],
                  val_loss[epoch]))
      # Save the current model (checkpoint) to a file
      model_path = dir + f"/{net.name}_BS={batch_size}_LR={learning_rate}_EP={epoch}"
      torch.save(net.state_dict(), model_path)
   print('Finished Training')
   end_time = time.time()
   elapsed_time = end_time - start_time
   print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
   # Write the train/test loss/err into CSV file for plotting later
   epochs = np.arange(1, num_epochs + 1)
   np.savetxt("{}_train_err.csv".format(model_path), train_err)
   np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
   np.savetxt("{}_val_err.csv".format(model_path), val_err)
   np.savetxt("{}_val_loss.csv".format(model_path), val_loss)