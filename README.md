# Handwritten Digit Recognition, the "Hello world" of Computer Vision :)

A pure Python desktop application that allows users to draw multiple digits on a canvas and predicts them using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The application can detect, segment, and predict multiple digits drawn side-by-side from left to right.

## Features
* **Interactive GUI:** Built with Tkinter for drawing digits seamlessly.
* **Multi-Digit Segmentation:** Uses OpenCV to find contours and separate multiple digits drawn on the same canvas.
* **Deep Learning Model:** Utilizes a custom CNN built with TensorFlow/Keras, trained on the MNIST dataset to achieve high accuracy.
* **Real-time Processing:** Automatically resizes, pads, and normalizes the drawn digits to match the model's expected input ($28 \times 28$ pixels).

## Prerequisites
Make sure you have Python 3.x installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
