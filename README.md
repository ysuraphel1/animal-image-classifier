# Animal Image Classifier

An end-to-end computer vision application that classifies uploaded animal images using a pretrained convolutional neural network built with PyTorch and deployed through an interactive Streamlit interface.

This repository includes the trained model and dataset structure so users can clone the project and run the application immediately without downloading additional files.

---

## Demo

Below is a walkthrough of the application:

![App Demo](demo/demo.gif)

---

## Features

* Upload an image through a local web interface
* Predict animal species from image input
* Display prediction confidence scores
* Show top-3 predicted classes
* Uses transfer learning with ResNet-18
* Includes dataset structure and pretrained model artifacts
* Runs locally with no additional setup beyond dependency installation

---

## Supported Animal Classes

The model currently predicts the following species:

* butterfly
* cat
* chicken
* cow
* dog
* elephant
* horse
* spider
* and many more!

---

## Repository Structure

```
animal-image-classifier/
│
├── app.py
├── predict.py
├── train.py
├── evaluate.py
├── prepare_data.py
├── requirements.txt
├── README.md
│
├── artifacts/
│   ├── best_model.pth
│   ├── class_names.json
│   ├── confusion_matrix.png
│   └── metrics.json
│
├── data/
│   ├── raw/
│   ├── train/
│   ├── val/
│   └── test/
│
└── src/
    ├── config.py
    ├── data_utils.py
    ├── inference_utils.py
    └── model_utils.py
```

---

## Quick Start

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/animal-image-classifier.git
cd animal-image-classifier
```

Create a virtual environment:

### macOS / Linux

```
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

Upload an image in the browser window that opens automatically to receive predictions.

---

## Example Prediction Output

```
Prediction: dog
Confidence: 92.4%

Top Predictions:
dog        92.4%
cat         4.8%
cow         1.3%
```

---

## Running Predictions from the Command Line

You can classify an image directly from the terminal:

```
python predict.py --image path/to/image.jpg --top_k 3
```

Example:

```
python predict.py --image data/test/dog/dog_00001.jpg --top_k 3
```

---

## Model Details

Architecture:

ResNet-18 convolutional neural network (transfer learning)

Framework:

PyTorch

Input:

224 × 224 RGB images

Output:

Probability distribution across supported animal classes with top-k ranking

Pipeline:

```
uploaded image → preprocessing → ResNet-18 model → probability scores → predicted class
```

The model learns directly from pixel values and does not rely on filenames or image metadata.

---

## Dataset

This project uses the **Animal Image Dataset (90 Different Animals)** created by **Sourav Banerjee** and made available on Kaggle:

https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

The dataset is redistributed here solely for demonstration purposes as part of this computer vision classification pipeline.

Full credit for dataset collection and preparation belongs to the original author.

Directory structure:

```
data/raw/<class_name>/image.jpg
```

Example:

```
data/raw/cat/cat_001.jpg
data/raw/dog/dog_014.jpg
data/raw/horse/horse_022.jpg
```

Dataset splits are stored in:

```
data/train/
data/val/
data/test/
```

---

## Re-training the Model (Optional)

To retrain the classifier using the included dataset:

```
python train.py
```

To regenerate evaluation metrics:

```
python evaluate.py
```

This produces:

```
artifacts/best_model.pth
artifacts/class_names.json
artifacts/confusion_matrix.png
artifacts/metrics.json
```

---

## Output Artifacts

The repository includes pretrained model artifacts for immediate inference:

```
artifacts/best_model.pth
artifacts/class_names.json
artifacts/confusion_matrix.png
artifacts/metrics.json
```

These allow the application to run without retraining.

---

## Limitations

The classifier predicts only among the supported animal classes listed above.

If an uploaded image contains an unsupported species, the model will still return the closest match with lower confidence.

---

## Tech Stack

Python
PyTorch
Torchvision
Streamlit
scikit-learn
matplotlib

---

## Example Use Cases

Demonstrations of convolutional neural networks

Computer vision experimentation and transfer learning workflows

Interactive ML model deployment examples

Portfolio projects showcasing applied deep learning pipelines

---

## License

This repository is intended for portfolio use.
