# How to Run This Project

This document provides step-by-step instructions to set up and run the Bridge Crack Depth Pipeline.

## 1. Prerequisites

- Python 3.8+
- `pip` for package management

## 2. Setup

### a. Clone the Repository

If you haven't already, clone this project to your local machine.

### b. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Navigate to the project directory
cd bridge_crack_pipeline

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### c. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## 3. Running the Pipeline

This project can be run in two modes:
- **Training Mode:** Train all models from scratch using the provided synthetic data.
- **Inference Mode:** Use pre-trained models to analyze a new image.

---

### Full Pipeline (Training from Scratch)

Follow these steps to run the entire pipeline, including data generation and model training.

#### Step 1: Generate Synthetic Data

This script creates a set of synthetic crack images, masks, and a ground truth CSV file in the `data/demo/` directory.

```bash
python src/make_synth_demo.py
```

#### Step 2: Train the U-Net Segmentation Model

This trains the U-Net model to identify cracks in images. The best model is saved as `outputs/unet_demo/best.pt`.

```bash
python src/seg_unet_train.py --config config.yaml
```

#### Step 3: Run Inference to Generate Masks

This script uses the trained U-Net model to generate predicted crack masks for the images.

```bash
python src/seg_infer.py --config config.yaml --images data/demo/images --out_masks outputs/inferred_masks
```

#### Step 4: Train Depth Regressor and Get Final Output

This is the final step. It uses the generated masks to calculate crack features (width, length) and trains a model to predict the depth. It outputs the final results, including a severity assessment, into a CSV file.

```bash
python src/features_and_depth.py --config config.yaml --images data/demo/images --masks outputs/inferred_masks --nde_csv data/demo/nde_groundtruth.csv --out outputs/regressor
```
The final output CSV will be located at `outputs/regressor/results.csv`.

---
### Inference on a Custom Image

To run the pipeline on your own image:

1.  **Place your image** in the `data/demo/images` folder. It's a good idea to clear out the old synthetic images first.
2.  **Run the inference and feature extraction steps:**
    ```bash
    # Step 1: Generate the crack mask for your image
    python src/seg_infer.py --config config.yaml --images data/demo/images --out_masks outputs/inferred_masks

    # Step 2: Generate the final analysis CSV
    python src/features_and_depth.py --config config.yaml --images data/demo/images --masks outputs/inferred_masks --nde_csv data/demo/nde_groundtruth.csv --out outputs/regressor
    ```
The final output for your image will be in `outputs/regressor/results.csv`.
