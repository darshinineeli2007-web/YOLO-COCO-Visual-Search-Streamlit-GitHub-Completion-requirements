# YOLO-COCO-Visual-Search-Streamlit-GitHub-Completion-requirements

## 1. AIM

To develop an AI-based image classification system using PyTorch and Deep Learning techniques to identify and classify images into three categories: Cats, Dogs, and Pandas. The system uses Transfer Learning with the pre-trained ResNet18 architecture to extract image features and classify unseen images.

## 2. OBJECTIVES

1. To collect and organize images of cats, dogs, and pandas.
2. To preprocess and resize images into a standard format.
3. To apply data augmentation to improve model performance.
4. To implement Transfer Learning using ResNet18.
5. To train and validate the image classification model.
6. To evaluate the model using test accuracy and a confusion matrix.
7. To predict the class of new images.
8. To develop an interactive image classification application.

## 3. SOFTWARE REQUIREMENTS

- Programming Language: Python
- Deep Learning Framework: PyTorch
- Model: ResNet18
- Libraries: Torchvision, NumPy, Matplotlib, Seaborn, Scikit-learn, Pillow
- Development Environment: Visual Studio Code
- Environment Manager: Conda
- Application Framework: Streamlit
- Dataset: Cats, Dogs, and Pandas image dataset

## 4. PROCEDURE

1. Collect images of cats, dogs, and pandas from an accessible dataset.
2. Divide the dataset into training, validation, and testing folders.
3. Import the required Python libraries.
4. Check GPU availability for model training.
5. Resize images to 224 × 224 pixels.
6. Apply normalization and data augmentation.
7. Load the pre-trained ResNet18 model.
8. Freeze the pre-trained layers and replace the final classification layer with three output classes.
9. Train the model using the Adam optimizer and Cross Entropy Loss.
10. Validate the model after each epoch.
11. Evaluate the trained model using test accuracy and a confusion matrix.
12. Save the trained model.
13. Develop a Streamlit application for image upload and prediction.
14. Run the final application locally using Visual Studio Code and Conda.
15. Capture screenshots and upload the project to a public GitHub repository.

## 5. DATASET STRUCTURE

The dataset should be organized as follows:

```text
dataset/
    train/
        cats/
        dogs/
        pandas/
    valid/
        cats/
        dogs/
        pandas/
    test/
        cats/
        dogs/
        pandas/
```

## 6. FULL PYTHON CODE

The following code is for training and evaluating the ResNet18 classification model.

### File: train.py

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = "dataset"
MODEL_DIR = "models"
RESULT_DIR = "results"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "valid"),
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_transform
)

classes = train_dataset.classes
print("Classes:", classes)

assert len(classes) == 3, "Dataset must contain exactly three classes."

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(classes))
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

train_losses = []
valid_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()

    val_loss /= len(valid_loader)

    train_losses.append(train_loss)
    valid_losses.append(val_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Validation Loss: {val_loss:.4f}"
    )

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(predictions)
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0
))

cm = confusion_matrix(
    all_labels,
    all_preds,
    labels=list(range(len(classes)))
)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "training_loss.png"))
plt.show()

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes
}, os.path.join(MODEL_DIR, "cat_dog_panda_resnet18.pth"))

print("\nModel saved successfully!")
print("Training and evaluation completed.")
```

### File: app.py

```python
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(
    page_title="Cat Dog Panda Classifier",
    page_icon="🐾"
)

st.title("🐾 Cat, Dog & Panda Image Classifier")
st.write("Upload an image to identify whether it is a Cat, Dog, or Panda.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    "models/cat_dog_panda_resnet18.pth",
    map_location=device,
    weights_only=True
)

classes = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(classes))
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    predicted_class = classes[prediction.item()]

    st.success(f"Prediction: {predicted_class.title()}")
    st.write(f"Confidence: {confidence.item() * 100:.2f}%")
```

### File: requirements.txt

```text
torch
torchvision
streamlit
numpy
matplotlib
seaborn
scikit-learn
Pillow
```

## 7. EXECUTION COMMANDS

After setting up the Conda environment and installing the requirements in Visual Studio Code, run:

```bash
python train.py
```

After training completes and the model is saved, run:

```bash
streamlit run app.py
```

## 8. EXPECTED OUTPUT

1. GPU or CPU device information.
2. Dataset classes and image counts.
3. Training loss and training accuracy for each epoch.
4. Validation loss.
5. Test accuracy and classification report.
6. Confusion matrix.
7. Training and validation loss graph.
8. Saved trained model.
9. Streamlit interface displaying the predicted class and confidence score.

## 9. ENHANCEMENTS

1. Transfer Learning using ResNet18.
2. Image augmentation using random flipping and rotation.
3. Dropout regularization to reduce overfitting.
4. GPU acceleration when available.
5. Interactive Streamlit application.
6. Prediction confidence display.

## 10. OUTPUT
<img width="1552" height="703" alt="image" src="https://github.com/user-attachments/assets/08c3b9f4-f82a-41c6-8da5-da72bcbfb2eb" />
<img width="1063" height="778" alt="image" src="https://github.com/user-attachments/assets/de8389f3-ba4b-4646-bfe9-6647fde26b7f" />
<img width="712" height="571" alt="image" src="https://github.com/user-attachments/assets/4e784a32-c35c-4da3-a4c9-0d27f7f9bef6" />
<img width="1140" height="763" alt="image" src="https://github.com/user-attachments/assets/e7423423-2b8b-47e2-aec1-e47381d8265f" />


## 11. RESULT

The project implements a three-class image classification system using PyTorch and the ResNet18 Transfer Learning architecture. It trains the model to identify cats, dogs, and pandas, evaluates its performance using test data, and provides an interactive Streamlit interface for predicting new images. The final accuracy and performance metrics are recorded after successful execution.
