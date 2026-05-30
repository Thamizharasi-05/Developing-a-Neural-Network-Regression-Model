# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Neural Network Model
Include the neural network model diagram.
<img width="1037" height="704" alt="image" src="https://github.com/user-attachments/assets/357f8056-69d7-4df1-bce3-d541671eeaa7" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: G. Thamizharasi

### Register Number: 212224100059

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)




def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      Loss=criterion(ai_brain(X_train),y_train)
      Loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(Loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {Loss.item():.6f}')


```

### Dataset Information
<img width="251" height="524" alt="image" src="https://github.com/user-attachments/assets/34b00c20-94c8-4d1c-9d89-8633de71392d" />

### OUTPUT
<img width="407" height="232" alt="image" src="https://github.com/user-attachments/assets/2df5193b-2744-43d4-9e95-4f11f3092633" />

### Training Loss Vs Iteration Plot
<img width="726" height="569" alt="image" src="https://github.com/user-attachments/assets/4e33c7c7-2704-4d85-8cc7-266c05206b65" />

### New Sample Data Prediction
<img width="322" height="43" alt="image" src="https://github.com/user-attachments/assets/d1515ad8-1692-4b48-a5cd-d19fd98b580d" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
