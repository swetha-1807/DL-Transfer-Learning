# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement

The aim of this experiment is to develop an image classification model using transfer learning with the VGG19 architecture. The model is trained on a given dataset to perform binary classification. A pre-trained VGG19 model is used to utilize features learned from the ImageNet dataset. The final layer of the network is modified to match the required output classes. The model is trained and validated using suitable loss functions and optimization techniques. The performance of the model is evaluated using accuracy, confusion matrix, and classification report.

## Theory

Deep Learning uses neural networks to automatically learn features from data. Convolutional Neural Networks (CNNs) are highly effective for image classification as they capture spatial patterns in images. Transfer learning allows the reuse of pre-trained models to improve performance and reduce training time. VGG19 is a deep CNN with 19 layers known for its strong feature extraction capability. In this model, the final layer is modified for binary classification using a sigmoid-based loss function. The Adam optimizer and BCEWithLogitsLoss are used for efficient training and accurate prediction.

## Neural Network Model

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/36805bf4-5d20-4820-892f-437be3bd9857" />

## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.

## PROGRAM

### Name: SWETHA K

### Register Number: 212224230284

```
# Load Pretrained Model and Modify for Transfer Learning

model=models.vgg19(weights=VGG19_Weights.DEFAULT)


# Modify the final fully connected layer to match the dataset classes

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)


# Include the Loss function and optimizer

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Train the model

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())

            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        train_losses.append(running_loss/len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss=0.0
        with torch.no_grad():
          for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())
            val_loss+=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:SWETHA K")
    print("Register Number: 212224230284")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
# Train the model
# Write your code here
train_model(model,train_loader,test_loader)

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="795" height="851" alt="image" src="https://github.com/user-attachments/assets/a40ff5be-d083-4b5f-9510-821c5eda5314" />


## Confusion Matrix

<img width="788" height="579" alt="image" src="https://github.com/user-attachments/assets/e482b48d-a078-4fb8-8a52-0bfe428d832d" />


## Classification Report

<img width="918" height="281" alt="image" src="https://github.com/user-attachments/assets/18a5bab9-0368-4571-8762-47257118799d" />

### New Sample Data Prediction
<img width="594" height="448" alt="image" src="https://github.com/user-attachments/assets/1363b500-baae-4b57-97b1-48be26d7004d" />
<img width="460" height="443" alt="image" src="https://github.com/user-attachments/assets/2e3686d8-df35-4bf8-9f3e-548acf6a4e1a" />


## RESULT
The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.


