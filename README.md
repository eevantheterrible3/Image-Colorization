
# Image-Colorization

This repository contains code and data used for an image colorization project (grayscale -> RGB) implemented and demonstrated in `training.ipynb`.

The notebook trains and evaluates three different neural-network based approaches for colorization:

- Encoder-Decoder
- Plain CNN
- U-Net

Each model takes a 128x128 grayscale input (single channel) and predicts a 128x128 RGB image (3 channels). Trained models and example outputs produced by the notebook are saved to the repository when training/prediction cells are executed.

## Requirements

Tested with Python 3.8+ and TensorFlow 2.x. Minimal required Python packages used in the notebook:

- tensorflow
- numpy
- pandas
- pillow (PIL)
- matplotlib

Install with pip (PowerShell example):

```powershell
python -m pip install --upgrade pip
python -m pip install tensorflow numpy pandas pillow matplotlib
```

If you use a virtual environment, activate it first (for PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

## Quick start — open the notebook

Start Jupyter (PowerShell):

```powershell
jupyter notebook training.ipynb
```

Open and run the notebook cells in order. The notebook performs these high-level steps:

1. Import libraries
2. Recursively load images from `Data/training_data` and `Data/test_data` and resize them to 128x128
3. Convert and save grayscale versions to `Data/training_data_bw` and `Data/test_data_bw`
4. Prepare NumPy arrays for training/validation (normalizing to [0,1])
5. Define three models: encoder-decoder, CNN, and U-Net
6. Train each model (with EarlyStopping and ModelCheckpoint callbacks)
7. Save the trained `.h5` model files and generate colorized outputs into folders like `colorized_images_results_u_net`
8. Provide helper functions to load a saved model and run single-image predictions

Run-time notes:
- Training can be slow on CPU. If you have a GPU and a proper TensorFlow GPU build + drivers, training will be significantly faster.
- The notebook uses small image sizes (128x128) and modest batch sizes to keep memory requirements reasonable.

## Models and saved filenames

The notebook defines and trains three model types and saves them with these example filenames (cells in the notebook use these names):

- Encoder-Decoder: `colorization_model_encoder_decoder.h5` (checkpoint `best_encoder_decoder_model.keras`)
- CNN: `colorization_model_cnn.h5` (checkpoint `cnn_colorization_best.keras`)
- U-Net: `colorization_model_u_net.h5` (checkpoint `u_net_colorization_best.keras`)

If you re-run the notebook, the model checkpoint callbacks will overwrite these files with the best validation models.

## How to run a single-image prediction (example)

The notebook includes helper functions: `load_trained_model`, `prepare_image_for_prediction`, and `display_grayscale_and_colorized`.

Example usage (run inside a Python REPL, script, or a notebook cell):

```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# helpers (same logic as in the notebook)
def prepare_image_for_prediction(image_path, target_size=(128,128)):
		img = Image.open(image_path).convert('L')
		img = img.resize(target_size)
		arr = np.array(img).astype('float32') / 255.0
		arr = arr[..., np.newaxis]
		return np.expand_dims(arr, axis=0)

def display_grayscale_and_colorized(grayscale_img, colorized_img):
		plt.figure(figsize=(10,5))
		plt.subplot(1,2,1)
		plt.imshow(grayscale_img.squeeze(), cmap='gray')
		plt.axis('off')
		plt.subplot(1,2,2)
		plt.imshow(colorized_img)
		plt.axis('off')
		plt.show()

# Load model and predict
model = load_model('colorization_model_u_net.h5', compile=False)
input_img = prepare_image_for_prediction('Data/test_data_bw/20056_bw.jpg')
pred = model.predict(input_img)[0]
pred = (pred * 255).astype('uint8')
display_grayscale_and_colorized(input_img[0], pred)
```

PowerShell note: use forward/backslashes exactly as shown for paths on Windows, or use raw strings in Python.

## Output files

- Trained models (in the notebook): `*.h5` files saved in the repository root.
- Checkpoints: `*.keras` files created by ModelCheckpoint callbacks.
- Colorized images: saved into folders like `colorized_images_results_u_net` by the notebook.

## Reproducibility and tips

- The notebook normalizes image pixels to [0, 1] before training and multiplies by 255 when saving predicted images.
- If you want to train on larger images / more data, increase memory and batch size and consider using a GPU.
- To evaluate results quantitatively, consider adding PSNR / SSIM metrics to the training/evaluation pipeline.
