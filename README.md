# Human Activity Recognition with CNN + LSTM Hybrids (UCI HAR)

Deep learning project (DSE3120, Manipal University Jaipur) on smartphone-sensor Human Activity Recognition.
It compares CNN, CNN+LSTM, CNN+Self-Attention and CNN+BiLSTM models, and benchmarks them against
the LSTM-CNN architecture of Xia et al. (2020).

## Dataset
UCI HAR: raw 9-channel inertial windows (128 x 9), 6 activities, official subject-independent split
(7,352 train / 2,947 test windows). See [`data/README.md`](data/README.md) for download instructions.

## Notebooks
| Notebook | Models | Notes |
|---|---|---|
| `01_HAR_CNN_BiLSTM_Hybrid.ipynb` | CNN baseline, CNN+BiLSTM (raw), Hybrid (raw CNN-BiLSTM branch + 561 engineered features), 5-seed ensemble | Best results |
| `02_HAR_CNN_LSTM_Attention.ipynb` | CNN baseline, CNN+LSTM, CNN+LSTM+Self-Attention | Raw signals only |

## Results (UCI HAR official test set)
| Model | Accuracy | Macro F1 |
|---|---|---|
| CNN baseline (notebook 01) | 89.96% | 89.77% |
| CNN + BiLSTM, raw (notebook 01) | 88.80% | 88.45% |
| Hybrid, single seed (notebook 01) | 94.88% | 94.84% |
| **Hybrid, 5-seed ensemble (notebook 01)** | **95.69%** | **95.70%** |
| CNN baseline (notebook 02) | 90.87% | 91.05% |
| CNN + LSTM (notebook 02) | 91.41% | 91.48% |
| CNN + LSTM + Attention (notebook 02) | 91.72% | 91.77% |
| *Xia et al. (2020), reported* | *95.80%* | *95.78%* |

Sitting vs. standing is the hardest pair to separate in all models.

## Comparison with Xia et al. (2020)
Xia et al. feed raw signals to two stacked LSTM layers (32 units), then Conv(64) -> max-pool -> Conv(128) -> GAP -> BN
(about 49.6k parameters). Our hybrid ensemble reaches a similar accuracy but is **not a reproduction of the paper's model**: it puts the CNN
before the recurrent layers, adds 561 engineered features and averages five seeds. Note the paper used its own train/test split, so numbers are not strictly like-for-like.

## Run it
```bash
pip install -r requirements.txt
```
Open a notebook in Jupyter, VS Code or Google Colab and run all cells. The notebooks save figures and CSVs to
`results/` and trained models to `models/` (the notebooks were run on Colab; copy those folders here after a run).

## Reference
K. Xia, J. Huang, H. Wang, "LSTM-CNN Architecture for Human Activity Recognition," *IEEE Access*, vol. 8, 2020,
DOI: 10.1109/ACCESS.2020.2982225. PDF in [`docs/`](docs/).

## Author
Devansh Agarwal
