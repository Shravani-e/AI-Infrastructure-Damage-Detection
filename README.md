# Bridge Crack Depth Pipeline (U-Net + Regression)

An end-to-end pipeline for crack detection, measurement, and depth estimation in bridge structures. The pipeline uses U-Net for segmentation followed by regression analysis for depth prediction.

## Features

- Crack detection using U-Net segmentation
- Automated measurements:
  - Crack length
  - Crack width (50th, 95th percentile, max)
  - Crack area
  - Depth prediction
- Severity classification
- Repair recommendations
- Visual annotations with measurements

## Prerequisites

```bash
# Required Python packages
opencv-python    # Image processing
numpy           # Numerical operations
scikit-image    # Image analysis
scikit-learn    # Machine learning
pandas          # Data handling
pyyaml          # Configuration
torch           # Deep learning
torchvision     # Computer vision
albumentations  # Data augmentation
xgboost         # Gradient boosting
matplotlib      # Plotting
```

## Quick Start

1. Install Dependencies:
```bash
pip install -r requirements.txt
```

2. Generate Demo Data (Optional):
```bash
python data/demo/make_synth_demo.py
```

3. Train U-Net Model:
```bash
python src/seg_unet_train.py --config config.yaml
```

4. Run Crack Detection:
```bash
python src/seg_infer.py --config config.yaml --images data/demo/images --out_masks outputs/masks
```

5. Generate Measurements:
```bash
python src/features_and_depth.py --config config.yaml --images data/demo/images --masks outputs/masks --nde_csv data/demo/measurements.csv --out outputs/results
```

## Project Structure

```
config.yaml             # Configuration file
requirements.txt        # Python dependencies
data/
  demo/                # Demo data and generation
    make_synth_demo.py # Synthetic data generator
src/
  features_and_depth.py   # Feature extraction and depth prediction
  seg_infer.py           # Segmentation inference
  seg_unet_train.py      # U-Net training
  utils.py               # Utility functions
```

## Code Files Description

### Core Components

1. **seg_unet_train.py**
   - Implements U-Net model training for crack segmentation
   - Key functions:
     - Model architecture definition
     - Training loop implementation
     - Validation and model saving
     - Loss function (BCE-Dice)
   - Outputs trained model weights to `outputs/unet_demo/best.pt`

2. **seg_infer.py**
   - Handles crack detection inference
   - Features:
     - Loads trained U-Net model
     - Processes input images
     - Generates binary masks
     - Applies threshold for crack detection
   - Input: Raw images
   - Output: Binary masks in `outputs/masks/`

3. **features_and_depth.py**
   - Core measurement and analysis module
   - Functionality:
     - Extracts crack measurements
       - Length calculation using skeletonization
       - Width statistics (50th, 95th percentile, max)
       - Area computation
     - Depth prediction using Random Forest
     - Severity classification
     - Repair plan generation
   - Visualization:
     - Draws crack contours (red)
     - Overlays measurements
     - Creates annotated images

### Utility Files

4. **utils.py**
   - Shared utility functions
   - Components:
     - Configuration loading
     - Device setup (CPU/GPU)
     - U-Net model definition
     - Image processing helpers
     - Evaluation metrics

5. **make_synth_demo.py** (in data/demo/)
   - Synthetic data generation
   - Creates:
     - Simulated crack images
     - Ground truth masks
     - Sample measurements
   - Useful for:
     - Testing pipeline
     - Development
     - Demonstration

### Data Flow

The code files work together in the following sequence:

1. `make_synth_demo.py` → Creates demo dataset
2. `seg_unet_train.py` → Trains segmentation model
3. `seg_infer.py` → Detects cracks in images
4. `features_and_depth.py` → Analyzes and measures cracks

Each step uses `utils.py` for common functionality.

## Outputs

The pipeline generates:

1. **Segmentation Masks** (`outputs/masks/`):
   - Binary masks showing detected cracks

2. **Results** (`outputs/results/`):
   - `results.csv`: All measurements
     - Length (mm)
     - Width statistics (mm)
     - Area (mm²)
     - Predicted depth (mm)
     - Severity classification
   - `annot_*.jpg`: Annotated images showing:
     - Crack outlines (red contours)
     - Measurement overlay
     - Classification and predictions

## Configuration

Key parameters in `config.yaml`:

```yaml
gsd_mm_per_px: 0.20    # Ground sampling distance (mm/pixel)
train:
  img_size: 512        # Training image size
  batch_size: 4        # Training batch size
  epochs: 1            # Training epochs
infer:
  thresh: 0.5         # Detection threshold
severity:              # Classification thresholds
  minor:
    depth_mm_lt: 5
  moderate:
    depth_mm_range: [5, 15]
  severe:
    depth_mm_gt: 15
```

## Using Your Own Data

Organize your data as follows:

```
data/
  your_dataset/
    images/          # Your crack images
    masks/           # Ground truth masks (for training)
    measurements.csv # Ground truth depths (for training)
```

Then update paths in `config.yaml` accordingly.

## Notes

- Ensure proper calibration of `gsd_mm_per_px` for accurate measurements
- The severity classification uses configurable thresholds
- Training data should include diverse crack types and sizes
- Visualization includes red contours for detected cracks and measurement overlays
