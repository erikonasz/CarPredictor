# Image Classification System
This project aims to develop a robust image classification system using machine learning and deep learning techniques. The system incorporates various models such as ResNet50, MobileNetV2, DenseNet121, and a custom CNN model. The models are trained on a dataset of car brand images and evaluated using standard evaluation metrics like accuracy, precision, recall, and F1 score. The trained models can be integrated into a functional application or service for real-world usage.

![image](https://github.com/erikonasz/DeepLearning-CarPrediction/assets/75033114/b34887ac-c9e3-4250-9784-f64f865d25dc)

# Dataset
For this project, the dataset consists of car brand images from two sources:

- [Kaggle: 100 Images of Top 50 Car Brands](https://www.kaggle.com/datasets/yamaerenay/100-images-of-top-50-car-brands)
- [Kaggle: The Car Connection Picture Dataset](https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset)


# Dependencies
To run this project, you need to install the following dependencies:

numpy
tensorflow
matplotlib
You can install them using pip:

**pip install numpy tensorflow matplotlib**

# Image Preprocessing
Before training the models, the input images are preprocessed using the ImageDataGenerator class from the tensorflow.keras.preprocessing.image module. The following preprocessing options are applied:

- Rescale pixel values to the range [0, 1]
- Randomly rotate images by up to 20 degrees
- Randomly shift images horizontally and vertically
- Randomly apply shearing transformation
- Randomly zoom in on images
- Randomly flip images horizontally
- Fill in missing pixels with the nearest value
- Model Comparison

# Models for comparison:

- **ResNet50**: Known for its depth and residual connections, ResNet50 has 50 layers and addresses the problem of vanishing gradients in deep networks.
- **MobileNetV2**: Designed to be efficient and lightweight, MobileNetV2 utilizes depth-wise separable convolutions, reducing computational complexity while maintaining good accuracy.
- **DenseNet121**: Part of the DenseNet family, DenseNet121 has 121 layers and introduces dense connectivity, where each layer is directly connected to every other layer.
- **CNN**: A custom Convolutional Neural Network (CNN) model is created. CNNs are widely used for image recognition tasks, leveraging their ability to identify patterns in images.
The models are trained and evaluated using the provided dataset. For transfer learning models (ResNet50, MobileNetV2, and DenseNet121), the pre-trained weights from the ImageNet dataset are loaded. The CNN model is trained from scratch.

# Usage 

Please run every code block from top to bottom, make sure to adjust epoch number to your needs:
![image](https://github.com/erikonasz/DeepLearning-CarPrediction/assets/75033114/f05e7f44-ae73-4d28-8f90-0c796dfa300e)
Upload any car image to same directory, predict that image with new command block:
predict_image("image_name.jpg")

# Training and Evaluation
The models are compiled with the Adam optimizer and categorical cross-entropy loss. They are trained for 20 epochs with a batch size of 8. During training, the accuracy and loss metrics are recorded for both the training and validation sets.

The training and validation accuracy and loss are plotted for each model using subplots for visual analysis.

# Results 

We evaluated the performance of different models on the car brand classification task using a dataset containing images of various car brands. The models we compared were ResNet50, MobileNetV2, DenseNet121, and a custom CNN.

- ResNet50 achieved an accuracy of ~40%.
- MobileNetV2 achieved an accuracy of ~83%
- DenseNet121 achieved an accuracy of ~87%
- The custom CNN achieved an accuracy of ~50%
- From these results, we can conclude that DenseNet121 performed the best among the pre-trained models, achieving the highest accuracy. It outperformed both ResNet50 and MobileNetV2. The custom CNN, while achieving a slightly lower accuracy, still demonstrated competitive performance.
![image](https://github.com/erikonasz/DeepLearning-CarPrediction/assets/75033114/2819664f-d3ce-42ba-b2e9-5370163b265c)

