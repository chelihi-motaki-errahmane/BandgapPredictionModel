# BandgapPredictionModel
A residual MLP model that predicts the bandgap of materials from Density,symmetry,Elemental Properties  using the matbench_mp_gap from matminer

## The Challenge
Working with the `matbench_mp_gap` dataset (106,113 samples) presents significant memory challenges. I implemented a **custom chunking and featurization pipeline** using `matminer` to process data in batches, ensuring the workflow is stable even on limited hardware.

## Tech Stack
- **Data:** Matminer, Pandas, Scikit-Learn (KNNImputer for missing data)
- **Model:** PyTorch (Deep Learning)
- **Architecture:** Feed-forward Neural Network with:
  - Batch Normalization
  - Residual (Skip) Connections
  - Dropout for regularization

## Results
- **Mean Absolute Error (MAE):** 0.3456 eV 
- The model successfully identifies metallic vs. semiconducting behavior with high reliability.

## Project Structure
- `BandGapPrediction.ipynb`: The main workflow (Data cleaning, Scaling, Training).
- `model.py`: The PyTorch model architecture.
- `requirements.txt`: Dependencies.

## How to Run
1. Clone the repo.
2. Install requirements: `pip install -r requirements.txt`
3. Run the notebook. *Note: You will need a GPU with CUDA support for optimal training speed.*
