from abc import ABC, abstractmethod
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# Abstract base class for dimensionality reduction objects
class DimRedObject(ABC):
  
    def __init__(self, data: np.ndarray):
        """
        Initialize the DimRedObject with input data.
        
        Parameters:
        - data: np.ndarray - The data to be used for dimensionality reduction.
        """
        self.data = data
    

    @abstractmethod
    def transform(self, new_data: np.ndarray):
        """
        Abstract method for transforming new data.
        Must be implemented by subclasses.
        
        Parameters:
        - new_data: np.ndarray - The new data to be transformed.
        """
        pass

# PCAObject class that implements the DimRedObject interface using PCA
class PCAObject(DimRedObject):
    
    def __init__(self, data: np.ndarray, n_components :int =10):
        """
        Initialize the PCAObject with input data and number of components. Automatically scales data
        
        Parameters:
        - data: np.ndarray - The data to be reduced using PCA.
        - n_components: int - Number of principal components to retain (default is 10).
        """
        super().__init__(data)

        self.scaler = StandardScaler().fit(data)
        data = self.scaler.transform(data)
        self.pca = PCA(n_components).fit(data)  # Initialize the PCA model with n components
        #self.pca.fit(self.data)  # Fit the PCA model to the input data
    
    def transform(self, new_data: np.ndarray):
        """
        Transform the new data using the fitted PCA model.
        
        Parameters:
        - new_data: np.ndarray - The data to transform.
        
        Returns:
        - np.ndarray - The transformed data in the PCA-reduced space.
        """

        print("here")
        return self.pca.transform(self.scaler.transform(new_data))
    
    def get_components(self):
        """
        Get the principal components after fitting the PCA model.
        
        Returns:
        - np.ndarray - The principal components of the PCA model.
        """
        return self.pca.components_


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class AutoEncoderObject(DimRedObject):
    def __init__(self, data: np.ndarray, batch_sizes=[100, 500, 1000], epochs=20):
        super().__init__(data)
        self.data = data
        self.batch_sizes = batch_sizes
        self.epochs = epochs
        self.losses_per_batch_size = {}

        # Start hyperparameter tuning during initialization
        self.best_batch_size, self.best_losses = self.hyperparameter_tuning()
        self.train_with_batch_size(self.best_batch_size)

    def train_with_batch_size(self, batch_size):
        print(f"Training with batch size: {batch_size}")
        
        # Convert data to torch tensor
        data_tensor = torch.tensor(self.data).float()

        # Create DataLoader for batching
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the Autoencoder model
        self.model = AE(self.data.shape[1])

        # Validation using MSE Loss function
        loss_function = torch.nn.MSELoss()

        # Using Adam Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-1, weight_decay=1e-8)

        losses = []
        for epoch in range(self.epochs):
            epoch_losses = []
            for batch in dataloader:
                batch_data = batch[0]

                # Output of Autoencoder
                encoded, reconstructed = self.model(batch_data)

                # Calculating the loss function
                loss = loss_function(reconstructed, batch_data)

                # Zero gradients, perform backward pass, and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.detach().numpy())

            # Store the average loss for the epoch
            epoch_loss = np.mean(epoch_losses)
            losses.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {epoch_loss}")
        
        return losses

    def hyperparameter_tuning(self):
        best_loss = float('inf')
        best_batch_size = None
        
        for batch_size in self.batch_sizes:
            losses = self.train_with_batch_size(batch_size)
            self.losses_per_batch_size[batch_size] = losses
            
            # Track the best batch size based on the final loss
            final_loss = losses[-1]
            if final_loss < best_loss:
                best_loss = final_loss
                best_batch_size = batch_size
            
            # Plotting the losses for each batch size
            plt.plot(losses, label=f"Batch size: {batch_size}")
        
        # Final plot
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss per Epoch for Different Batch Sizes')
        plt.savefig("batch_size_tuning_results")
        plt.show()

        return best_batch_size, self.losses_per_batch_size[best_batch_size]

    def transform(self, new_data: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            transformed_data = []
            for sample in new_data:
                encoded = self.model(torch.tensor(sample).float())[0]
                transformed_data.append(encoded.detach().numpy())
            return np.array(transformed_data)


# 
class AE(torch.nn.Module):
    def __init__(self, input_dim):
        print("Input dimensions:", input_dim)
        super().__init__()


        # Building an encoder with Linear layers followed by ReLU activation function
        # Use int() to convert the float result of np.ceil to an integer
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(np.ceil(input_dim / 2))),
            torch.nn.ReLU(),
            torch.nn.Linear(int(np.ceil(input_dim / 2)), int(np.ceil(input_dim / 3))),
            torch.nn.ReLU(),
            torch.nn.Linear(int(np.ceil(input_dim / 3)), int(np.ceil(input_dim / 4))),
            torch.nn.ReLU(),
            torch.nn.Linear(int(np.ceil(input_dim / 4)), int(np.ceil(input_dim / 5))),
            torch.nn.ReLU(),
        )

         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(int(np.ceil(input_dim / 5)), int(np.ceil(input_dim / 4))),
            torch.nn.ReLU(),
            torch.nn.Linear(int(np.ceil(input_dim / 4)), int(np.ceil(input_dim / 3))),
            torch.nn.ReLU(),
            torch.nn.Linear(int(np.ceil(input_dim / 3)), int(np.ceil(input_dim / 2))),
            torch.nn.ReLU(),
            torch.nn.Linear(int(np.ceil(input_dim / 2)), input_dim),
            torch.nn.Sigmoid()
        )

 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded