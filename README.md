Project Overview

This is a demonstration of an end-to-end machine learning solution designed to classify the musical subgenre (e.g., Black Metal, Death Metal, Power Metal) of a band based solely on the visual characteristics of its official logo.

The application uses a transfer learning approach with the PyTorch framework, leveraging the VGG16 architecture to extract features from logo images before classification.


Technology Stack

The application is split into two main components:


Frontend Client (index.html)

HTML, Tailwind CSS, JavaScript

The static web interface for image submission. It runs entirely in the browser and calls the remote API.


Backend API (Container)

Flask, PyTorch (VGG16 Model)

Hosted on Google Cloud Run, this service loads the trained model weights (logo_genre_classifier.pt), performs inference on the uploaded image, and returns the genre prediction.
