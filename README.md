# Deep Neural Networks for Collaborative Filtering Recommendation System

## Overview

This repository implements and evaluates two deep learning models for collaborative filtering using the MovieLens dataset:

- **Multilayer Perceptron (MLP) with embeddings** for user and movie IDs  
- **Autoencoder** for unsupervised latent feature learning

The models predict user ratings on movies by learning latent interactions and reconstructing feature representations. The project includes detailed dataset preprocessing, model training, evaluation metrics, and visualization.

---

## Dataset

We use the **MovieLens dataset** (preprocessed version) consisting of three `.dat` files:

- `users.dat`: user demographic information (UserID, Gender, Age, Occupation, Zip-code)  
- `movies.dat`: movie metadata (MovieID, Title, Genres)  
- `ratings.dat`: user ratings for movies (UserID, MovieID, Rating, Timestamp)

### Preprocessing

- Loaded `.dat` files with correct delimiters and encoding (`latin-1` for movie titles).  
- Merged users, movies, and ratings into a single DataFrame.  
- Encoded categorical features:  
  - One-hot encoding for `Gender`  
  - Label encoding for `Occupation`  
  - Multi-label binarization for `Genres` (split by `|`)  
- Dropped unnecessary columns like `Zip-code` and `Title` for modeling.  
- Normalized numerical features (Age, Rating) using `MinMaxScaler` for autoencoder inputs.  
- Created train-test splits for model evaluation.

---

## Models

### 1. Multilayer Perceptron (MLP) with Embeddings

- Embedding layers for `UserID` and `MovieID` to capture latent factors.  
- Concatenated embeddings with other numeric/categorical features.  
- Fully connected layers with ReLU activations for non-linear modeling.  
- Mean Squared Error (MSE) loss optimized with Adam optimizer.  
- Predicts explicit ratings from 1 to 5.

### 2. Autoencoder

- Encoder-decoder architecture to learn compressed latent representations of features.  
- Uses fully connected layers with ReLU activations and Sigmoid output layer.  
- Trained to reconstruct normalized input features.  
- Provides unsupervised feature embeddings useful for downstream tasks.

---

## Training

- Both models trained for 20 epochs with batch size 256.  
- MLP uses `(UserID, MovieID, Features) -> Rating` supervised training.  
- Autoencoder uses reconstruction loss on normalized features.  
- Loss curves tracked and plotted for both models.

---

## Evaluation

- Used **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)** for MLP rating prediction evaluation.  
- Autoencoder evaluated on reconstruction RMSE over test data.  
- Visualized predicted vs actual ratings and error distributions for MLP.  
- Visualized reconstruction error distribution and latent embeddings (t-SNE) for Autoencoder.

---

## Visualization

- Training loss curves for both models.  
- Scatter plots of predicted vs actual ratings (MLP).  
- Histograms of prediction and reconstruction errors.  
- t-SNE visualization of 32-dimensional autoencoder latent embeddings reduced to 2D, optionally colored by user demographics.

---

## Usage

1. Clone the repository.
2. Prepare dataset by placing `users.dat`, `movies.dat`, and `ratings.dat` in the data folder.  
3. Run preprocessing scripts to merge and encode the dataset.  
4. Train MLP and Autoencoder models using provided training scripts.  
5. Evaluate models and generate visualizations.  
6. Use saved model weights (`mlp_model.pth`, `autoencoder_model.pth`) for inference or further fine-tuning.

---

## Requirements

- Python 3.7+  
- PyTorch  
- pandas  
- numpy  
- scikit-learn  
- matplotlib
