# Landmark Classification Using Convolutional Neural Networks

## Project Overview
This project applies Convolutional Neural Networks (CNNs) to build a landmark classifier that can predict the location of an image based on any discernible landmarks. Photo sharing and storage services benefit from location metadata, but many images lack this information due to missing GPS data or privacy concerns. This project aims to address this issue by developing machine learning models to classify landmarks automatically.

## Project Goals
1. **Build a CNN from Scratch**:
   - Visualize and preprocess the dataset.
   - Design and train a CNN model for landmark classification.
   - Evaluate and export the best network using Torch Script.

2. **Implement Transfer Learning**:
   - Explore different pre-trained CNN models.
   - Choose and fine-tune the best model for landmark classification.
   - Export the best transfer-learned model using Torch Script.

3. **Deploy the Model in an App**:
   - Develop a simple application using the best trained model.
   - Test the app to verify predictions and analyze strengths and weaknesses.
   - Generate an archive file for submission.

## Project Structure
The project consists of three Jupyter Notebooks:

1. **cnn_from_scratch.ipynb** – This notebook guides the process of creating a CNN model from scratch, including data visualization, preprocessing, training, and evaluation.
2. **transfer_learning.ipynb** – In this notebook, transfer learning is applied using pre-trained models, and the best-performing model is identified and fine-tuned.
3. **app.ipynb** – This notebook focuses on deploying the trained model in an application, testing its performance, and generating the final submission archive.

## Getting Started
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- Jupyter Notebook
- PyTorch
- Torchvision
- NumPy
- Matplotlib

### Installation
1. Clone the repository or download the project workspace.
   ```bash
   https://github.com/olugbeminiyi2000/LANDMARKCLASSIFIER/
   cd LANDMARKCLASSIFIER
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements (1).txt
   ```
   
## Evaluation and Reflection
- Compare accuracy, loss, and performance of both the CNN from scratch and the transfer learning model.
- Discuss any limitations and potential improvements for future work.

## Acknowledgments
- Thanks to PyTorch for providing pre-trained models for transfer learning.
- Dataset credits to relevant sources used in training and evaluation.

