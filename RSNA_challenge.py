import pandas as pd
import pydicom
import os
import math
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to measure overlap between predicted and target bounding boxes by intersection over union (IOU)
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# Path to dataset
root_dir = './rsna-pneumonia-detection-challenge'
img_dir = os.path.join(root_dir, 'stage_2_train_images')
test_img_dir = os.path.join(root_dir, 'stage_2_test_images')
label_file = os.path.join(root_dir, 'stage_2_train_labels.csv')

# Read the CSV files
train_labels = pd.read_csv('rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
detailed_class_info = pd.read_csv('rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')
print("Loaded CSV Files")
print(train_labels.head())
print(detailed_class_info.head())

# Merge the dataframes on 'patientId'
combined_labels = pd.merge(train_labels, detailed_class_info, on='patientId', how='left')
# Remove duplicates based on 'patientId'
combined_labels = combined_labels.drop_duplicates(['patientId'])
# for setting limit: combined_labels = combined_labels.sample(n=5)
print("\nMerged CSV Files")
print(combined_labels.head())


# Function to transform input images for preprocessing
def transform_image(file_path):
    ds = pydicom.dcmread(file_path)
    img = Image.fromarray(ds.pixel_array)  # Convert to PIL image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    return input_tensor

# Function to preprocess all training_images in a directory
print("\nConverting to PIL and transforming to tensor")
def preprocess_images(img_dir):
    patient_ids = [filename[:-4] for filename in os.listdir(img_dir) if filename.endswith('.dcm')]
    training_images = {}
    for filename in tqdm(patient_ids, desc="Processing images", unit="image"):
        file_path = os.path.join(img_dir, filename + '.dcm')
        if os.path.exists(file_path):
            image_tensor = transform_image(file_path)
            training_images[filename + '.dcm'] = image_tensor
    return training_images

# Custom mapping for class column
class_mapping = {'No Lung Opacity / Not Normal': 0, 'Normal': 1, 'Lung Opacity': 2}
combined_labels['class'] = combined_labels['class'].map(class_mapping)


combined_labels.set_index('patientId', inplace=True) # Replacing indices with patientIds
labels = combined_labels.to_dict('index') # Converting the dataframe to a dictionary
print("\nNumber of unique patient IDs (entries in labels): ", len(labels))


# Preprocess training images
train_img_dir = os.path.join(root_dir, 'stage_2_train_images')
training_images = preprocess_images(train_img_dir)

# Generate a list of patient IDs (get rid of .dcm in the filename)
preprocessed_ids = [img_file[:-4] for img_file in training_images.keys()]

# Preprocess test images
max_test_images = None  # Add the limit here
test_img_dir = os.path.join(root_dir, 'stage_2_test_images')
test_images = preprocess_images(test_img_dir)

print(f"Preprocessed {len(training_images)} Train Images")
print(f"Preprocessed {len(test_images)} Test Images")

#handling NaN values for Target = 0 (no bounding box)
def get_label(patientId):
    label_info = labels[patientId]
    if math.isnan(label_info['x']):
        label_info['x'] = 0
    if math.isnan(label_info['y']):
        label_info['y'] = 0
    if math.isnan(label_info['width']):
        label_info['width'] = 0
    if math.isnan(label_info['height']):
        label_info['height'] = 0
    return label_info

# Combine training_images and labels into a list of tuples
# Skip training_images without labels
data = [(training_images[img_file], get_label(img_file[:-4])) for img_file in training_images if img_file[:-4] in labels]
print(f"\nCombined labels and training images into {len(data)} data points")

# Split data into train and val sets
train_data, val_data = train_test_split(data, test_size=0.2)
print(f"\nSplit data into {len(train_data)} training data points and {len(val_data)} validation data points")

# Define a custom Dataset class
class PneumoniaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label_info = self.data[idx]
        label = torch.tensor([label_info['class'], label_info['x'], label_info['y'], label_info['width'], label_info['height']], dtype=torch.float32)
        return image, label

# Create Dataset objects for training and validation
train_dataset = PneumoniaDataset(train_data)
val_dataset = PneumoniaDataset(val_data)

# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the neural network model class
class Net(nn.Module):
    # Initialize network layers
    def __init__(self):
        super(Net, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Load pre-trained ResNet-50 model
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 120)  # Adjust the last fully connected layer
        self.fc2_class = nn.Linear(120, 3)  # 3 outputs for classification
        self.fc2_bbox = nn.Linear(120, 4)  # 4 outputs for bounding box

    # Define forward pass
    def forward(self, x):
        x = self.resnet50(x)  # Pass through ResNet-50
        x_class = self.fc2_class(x)
        x_bbox = self.fc2_bbox(x)
        return x_class, x_bbox

# Instantiate the network
net = Net()

# Use CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)  # Move model to GPU
print(f'The model is using {device}')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training loop
print("\nStarting training ...")
for epoch in range(1):  # loop over the dataset multiple times
    print(f"Starting epoch {epoch}")
    progress_bar = tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), desc="Epoch " + str(epoch))
    for i, data in progress_bar:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        class_outputs, box_outputs = net(inputs) # Unpack the two outputs

        class_labels = labels[:, 0].long()
        box_labels = labels[:, 1:]

        # Get classification loss
        class_loss = criterion(class_outputs, class_labels)

        # Get localization loss
        pos_indices = (class_labels == 2) # Indices where the class is 'Lung Opacity'
        localization_loss = F.smooth_l1_loss(box_outputs[pos_indices], box_labels[pos_indices])

        # Total loss
        loss = class_loss + localization_loss

        progress_bar.set_postfix({"loss": loss.item()})
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

print('Finished Training\n')

# Initializing actual and predicted class labels
true_classes = []
pred_classes = []


# Validation loop
total_iou = 0
total_images = 0
total_positives = 0

for data in val_dataloader:
    training_images, labels = data[0].to(device), data[1].to(device)
    labels = labels.float()

    class_outputs, box_outputs = net(training_images)

    class_labels = labels[:, 0].long()
    box_labels = labels[:, 1:] # Remaining columns in labels are bounding box coordinates

    true_classes.extend(class_labels.cpu().numpy())
    pred_classes.extend(torch.argmax(class_outputs, dim=1).cpu().numpy())

    # Loop through positive (class 1) predictions only for IOU calculation
    for i in range(len(class_labels)):
        if class_labels[i] == 2: # Assuming positive class is labeled as 2
            true_box = box_labels[i].cpu().numpy()
            pred_box = box_outputs[i].cpu().detach().numpy()
            total_iou += bb_intersection_over_union(true_box, pred_box)
            total_positives += 1

    total_images += len(training_images)

if total_positives > 0:
    # print("Average IoU for Positive Cases: ", total_iou / total_positives)
    print(f"Average IoU for Positive Cases: {total_iou} / {total_positives}")  # Set a static IOU for now
else:
    print("No Positive Cases to Compute IoU")

accuracy = accuracy_score(true_classes, pred_classes)
precision = precision_score(true_classes, pred_classes, average='macro')
recall = recall_score(true_classes, pred_classes, average='macro')

print("\nAccuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)


test_outputs = {}
with torch.no_grad(): # Disables gradient computation
    for image_id, processed_image in test_images.items():
        processed_image = processed_image.unsqueeze(0).to(device)
        class_output, bbox_output = net(processed_image) # Forward pass through the model
        test_outputs[image_id] = {
        "class_output": class_output.cpu().numpy(),
        "bbox_output": bbox_output.cpu().numpy()
} # Record the output

# Extracting class and bounding box predictions from test outputs
test_predictions = {}
for image_id, output in test_outputs.items():
    class_output = torch.tensor(output["class_output"])
    bbox_output = torch.tensor(output["bbox_output"])
    class_prediction = torch.argmax(class_output).item()
    test_predictions[image_id] = {'class': class_prediction, 'bbox': bbox_output}


# Output final results
print("\nAverage IoU: ", total_iou / total_images)

def visualize_predictions(image_path, prediction):
    # Read the original image
    ds = pydicom.dcmread(image_path)
    img = ds.pixel_array
    
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
    
    # Get bounding box prediction
    bbox = prediction['bbox']
    print(bbox)
    # Check if the class is 'Lung Opacity' and bounding box dimensions are not zero
    if prediction['class'] == 2 and all(b > 0 for b in bbox):  # Assuming class 'Lung Opacity' is mapped to 2
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

# Visualize predictions on a few test images
for image_id, prediction in list(test_predictions.items())[:5]: # Visualize first 5 predictions
    if not image_id.endswith('.dcm'):
        image_id += '.dcm'
    if prediction['class'] == 2:  # Only visualize if 'Lung Opacity'
        image_path = os.path.join(test_img_dir, image_id)
        visualize_predictions(image_path, prediction)