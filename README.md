# Fashion MNIST Image Classification

This project demonstrates image classification using the Fashion MNIST dataset. The goal is to build a model that can classify fashion-related images into different categories such as T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

## Dataset

The Fashion MNIST dataset is used for training and evaluation. It consists of 60,000 grayscale images of size 28x28 pixels, divided into 10 categories, along with a test set of 10,000 images. The dataset is preprocessed and split into training, validation, and test sets.

## Model Architecture

The model architecture used for image classification is a convolutional neural network (CNN). The CNN consists of the following layers:

1. Convolutional layer with 32 filters, a kernel size of (3, 3), ReLU activation, and padding set to 'same'.
2. Max pooling layer with a pool size of (2, 2).
3. Convolutional layer with 64 filters, a kernel size of (3, 3), ReLU activation, and padding set to 'same'.
4. Max pooling layer with a pool size of (2, 2).
5. Flatten layer to convert the 2D feature maps into a 1D feature vector.
6. Dense layer with 128 units and ReLU activation.
7. Dropout layer with a dropout rate of 0.4 to reduce overfitting.
8. Dense layer with 10 units (equal to the number of classes) and softmax activation for multi-class classification.

The model is compiled with the Adam optimizer, a learning rate of 0.001, and the sparse categorical cross-entropy loss function. The accuracy metric is used for evaluation.

## Training and Evaluation

Data augmentation techniques are applied to the training set using the ImageDataGenerator class from TensorFlow. This helps improve the model's ability to generalize and handle variations in the input data.

The model is trained for 20 epochs using batches of size 64. The training progress is printed to the console, showing the loss and accuracy values. After training, the model is evaluated on the test set to obtain the final test loss and accuracy.

## Flask Web Application

The project includes a Flask web application that allows users to upload an image and get predictions for the uploaded image. The uploaded image is preprocessed, fed into the trained model, and the top predicted labels with their corresponding probabilities are displayed.

The web application consists of an HTML file (`index.html`) for the user interface, a main Python file (`main.py`) for handling requests and running the Flask application, and a model Python file (`model.py`) for training and saving the model.

## Usage

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Run the main Python file `main.py` to start the Flask web application.
3. Access the web application in your browser at `http://localhost:5000`.
4. Click on the "Choose Image" button to select an image for classification.
5. Click on the "Upload and Classify" button to upload the selected image and view the predictions.
6. The top predicted labels with their probabilities will be displayed, along with a visual representation of the probabilities using a progress bar.
7. To go back to the main page and upload another image, click on the "Back to Main Page" button.

Feel free to modify the code and experiment with different configurations to further improve the model's performance.

## Credits

The Fashion MNIST dataset is originally curated by Zalando Research and can be accessed at [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist).

The code and project structure in this repository were developed by Ali Albalushi.
