# Human Activity Recognition with CNN + LSTM Hybrids

Deep Learning project (DSE3120, Manipal University Jaipur) on smartphone-sensor Human Activity Recognition using the UCI HAR Dataset.

The project compares CNN, CNN+LSTM, CNN+Self-Attention, and CNN+BiLSTM models. It also evaluates a hybrid CNN-BiLSTM model using raw sensor signals and engineered features, along with a 5-seed ensemble.

---

## Student Information

| Field | Details |
|---|---|
| Name | Devansh Agarwal |
| Registration No. | YOUR_REGISTRATION_NUMBER |
| Branch | Data Science |
| Batch | F |
| GitHub Username | Devansh-Agarwal47 |
| Course | DSE3120 – Deep Learning |
| University | Manipal University Jaipur |

---

## Project Title

**Human Activity Recognition with CNN + LSTM Hybrids**

---

## Project Overview

Human Activity Recognition (HAR) aims to automatically identify human activities using sensor data collected from smartphones.

This project uses deep learning models to classify six human activities from smartphone inertial sensor data.

The project evaluates and compares:

- CNN baseline
- CNN + LSTM
- CNN + LSTM + Self-Attention
- CNN + BiLSTM
- Hybrid CNN-BiLSTM
- 5-seed ensemble of the hybrid model

---

## Dataset

The project uses the **UCI Human Activity Recognition Using Smartphones Dataset (UCI HAR)**.

### Dataset Characteristics

- 6 activity classes
- 9 raw inertial sensor channels
- Raw input window: `128 × 9`
- Official subject-independent train/test split
- 561 engineered features

### Activities

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

Dataset download and preparation instructions are available in [`data/README.md`](data/README.md).

---

## Model Architectures

### 1. CNN Baseline

A Convolutional Neural Network (CNN) is used to extract local temporal patterns from the raw sensor signals.

### 2. CNN + LSTM

CNN layers extract local features, while LSTM layers model temporal dependencies in the sensor signals.

### 3. CNN + LSTM + Self-Attention

Self-attention is added to the CNN-LSTM architecture to allow the model to focus on important temporal representations.

### 4. CNN + BiLSTM

Bidirectional LSTM processes temporal information in both forward and backward directions.

### 5. Hybrid CNN-BiLSTM

The hybrid model combines:

- CNN processing of raw sensor signals
- BiLSTM temporal modelling
- 561 engineered UCI HAR features

A 5-seed ensemble is also evaluated by combining predictions from five independently trained models.

---

## Notebooks

| Notebook | Models | Description |
|---|---|---|
| `01_HAR_CNN_BiLSTM_Hybrid.ipynb` | CNN, CNN+BiLSTM, Hybrid CNN-BiLSTM, 5-seed Ensemble | Main experiment |
| `02_HAR_CNN_LSTM_Attention.ipynb` | CNN, CNN+LSTM, CNN+LSTM+Self-Attention | Attention comparison |

---

## Results

### UCI HAR Official Test Set

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| CNN baseline (Notebook 01) | 89.96% | 89.77% |
| CNN + BiLSTM, raw (Notebook 01) | 88.80% | 88.45% |
| Hybrid, single seed (Notebook 01) | 94.88% | 94.84% |
| **Hybrid, 5-seed ensemble (Notebook 01)** | **95.69%** | **95.70%** |
| CNN baseline (Notebook 02) | 90.87% | 91.05% |
| CNN + LSTM (Notebook 02) | 91.41% | 91.48% |
| CNN + LSTM + Attention (Notebook 02) | 91.72% | 91.77% |
| *Xia et al. (2020), reported* | *95.80%* | *95.78%* |

The hybrid 5-seed ensemble achieved the best performance among the models evaluated in this project.

---

## Results and Visualizations

All experiment results are organized inside the `results/` directory.

```text
results/
├── figures/
└── tables/
```

### Figures

The `results/figures/` directory contains the original figures generated in the notebooks.

### Tables

The `results/tables/` directory contains CSV files containing important experimental results, including:

- Dataset summary
- Hyperparameter search
- Cross-validation results
- Training history
- Training summary
- Bias-variance results
- Final test evaluation
- Per-class metrics
- Computational cost
- Ensemble results
- Ablation results
- Literature comparison
- Gradient convergence
- Sample predictions

---

## Comparison with Xia et al. (2020)

The project also compares the proposed approach with the LSTM-CNN architecture reported by Xia et al. (2020).

The proposed hybrid model is **not an exact reproduction** of the architecture reported in that paper.

The proposed approach:

- Uses CNN before the recurrent layers
- Uses BiLSTM
- Incorporates 561 engineered features
- Uses a 5-seed ensemble

Therefore, the comparison with Xia et al. should be interpreted as a performance reference rather than an exact reproduction.

The paper used its own train/test split, so the reported numbers are not strictly like-for-like.

---

## Project Structure

```text
har-cnn-lstm-hybrid/
│
├── data/
│   └── README.md
│
├── docs/
│
├── models/
│
├── notebooks/
│   ├── 01_HAR_CNN_BiLSTM_Hybrid.ipynb
│   └── 02_HAR_CNN_LSTM_Attention.ipynb
│
├── results/
│   ├── figures/
│   └── tables/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Devansh-Agarwal47/har-cnn-lstm-hybrid.git
cd har-cnn-lstm-hybrid
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open the notebooks using:

- Jupyter Notebook
- JupyterLab
- VS Code
- Google Colab

The main experiments are available in:

```text
notebooks/01_HAR_CNN_BiLSTM_Hybrid.ipynb
notebooks/02_HAR_CNN_LSTM_Attention.ipynb
```

Run the notebook cells sequentially to reproduce the experiments.

The notebooks save experimental figures and CSV results to the `results/` directory and trained models to the `models/` directory.

---

## Project Contributions

The project includes work on:

- Dataset preparation and analysis
- Deep learning model implementation
- CNN and recurrent model development
- Self-attention implementation
- Hybrid CNN-BiLSTM development
- Hyperparameter experimentation
- Cross-validation
- Ablation analysis
- Ensemble evaluation
- Performance evaluation
- Results analysis
- Documentation and repository organization

---

## Reference

K. Xia, J. Huang, H. Wang, "LSTM-CNN Architecture for Human Activity Recognition," *IEEE Access*, vol. 8, 2020.

DOI: 10.1109/ACCESS.2020.2982225.

The reference paper is available in the [`docs/`](docs/) directory.

---

## Author

**Devansh Agarwal**

GitHub: [Devansh-Agarwal47](https://github.com/Devansh-Agarwal47)

---

## Course Information

**Course:** DSE3120 – Deep Learning  
**University:** Manipal University Jaipur  
**Branch:** Data Science  
**Batch:** E
