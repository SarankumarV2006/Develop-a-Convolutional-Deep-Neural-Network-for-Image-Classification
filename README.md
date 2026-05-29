# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
The problem is to design and develop a Convolutional Deep Neural Network (CNN) that can automatically classify grayscale images into predefined categories. The model must learn important spatial features such as edges, textures, and shapes from image data and accurately predict the correct class label.Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model
<img width="991" height="481" alt="image" src="https://github.com/user-attachments/assets/ffe5dbfd-8c84-4310-99ac-7246be61f782" />


## DESIGN STEPS

## STEP 1:
Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

## STEP 2:
Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

## STEP 3:
Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

## STEP 4:
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

## STEP 5:
Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

## STEP 6:
Evaluate the trained model on test images and verify the classification accuracy for new unseen images. 


## PROGRAM

### Name: Sarankumar.V

### Register Number: 212224220089

```
class CNNClassifier(nn.Module):
   def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128) # Assuming input_size leads to 3x3 after pooling
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

   def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier(28)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):
for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Name:Sarankumar.V')
        print('Register Number: 212224220089')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```


### OUTPUT

## Training Loss per Epoch

<img width="485" height="305" alt="image" src="https://github.com/user-attachments/assets/c1ce219b-4a14-4f9f-83c2-2e13058733fb" />

## Confusion Matrix

<img width="1228" height="830" alt="image" src="https://github.com/user-attachments/assets/75ac94bc-1691-4361-9673-51c4c204ae85" />

## Classification Report

<img width="922" height="439" alt="image" src="https://github.com/user-attachments/assets/bf413d1b-c81a-461f-b318-5ad684177a89" />


### New Sample Data Prediction
<img width="613" height="620" alt="image" src="https://github.com/user-attachments/assets/206d94fb-5284-4cbb-8eef-5ac069e810c1" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
