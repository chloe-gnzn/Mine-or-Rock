# Mine-or-Rock: Sonar Classification with a Custom C# Neural Network

A 3-layer multi-layer perceptron (input, hidden, output) with backpropagation, written from scratch in C#,
applied to binary classification of sonar returns (metal mine vs. rock).

## Authors & Contributors
| Name | Role | Primary Contributions |
|---|---|---|
| **Chloe Julia Geonzon** | Custom Neural Engine (`NeuralEngine.cs`), Xavier initialization, backpropagation mathematics, loss & activation derivations. |
| **Iren Nathaleigh Guinita** | Data pipeline (`DataLoader.cs`, `Preprocessing.cs`), evaluation suite (`Evaluator.cs`), and hyperparameter grid sweep (`Trainer.cs`, `HyperparameterSweep.cs`). |

## How to Run
1. Clone or open the repository folder in Visual Studio Code or Visual Studio.
2. Make sure `Data/sonar.csv` is set to "Copy to Output Directory: Copy always".
3. Ensure the .NET 10 SDK is installed (`dotnet --version`).
4. Run `dotnet run`. The program loads the data, trains the baseline network, trains the tuned network,
   and prints the confusion matrix and metrics for each.
5. Set `runSweep = true` in `Program.cs` to regenerate the hyperparameter sweep.

Output files: `training_log.csv` (baseline), `training_log_tuned.csv` (tuned),
`hyperparameter_results.csv` (sweep).

## Architecture & Mathematical Foundations
- **Topology:** 60 Input Nodes → 24 Hidden Nodes → 1 Output Node (Sigmoid).
- **Weight Initialization:** Xavier (Glorot) uniform initialization with bounds ±√(6 / (fan_in + fan_out)). Biases initialized to 0.0.
- **Activation Function:** Clamped Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ bounded at $[-45, 45]$ to prevent exponential floating-point overflow.
  - Derivative: $\sigma'(a) = a(1.0 - a)$
- **Loss Function:** Sample Mean Squared Error (MSE): $E = \frac{1}{2}(y - \hat{y})^2$.
- **Optimization:** Online Backpropagation (Stochastic Gradient Descent) with analytical gradients:
- Output Delta: $\delta_o = (y - \hat{y}) \cdot \sigma'(\hat{y}) = (y - \hat{y}) \cdot \hat{y}(1.0 - \hat{y})$
- Hidden Delta: $\delta_h = (\delta_o \cdot W_{ho}) \cdot \sigma'(a_h) = (\delta_o \cdot W_{ho}) \cdot a_h(1.0 - a_h)$
- Weight Updates: $\Delta W = \eta \cdot \delta \cdot \text{activation}$
  
## Data Description
- **Source:** UCI Machine Learning Repository, "Connectionist Bench (Sonar, Mines vs. Rocks)".
- **Size:** 208 samples, 60 numeric features, 1 label.
- **Features:** each feature is the energy of a sonar return within one frequency band, already scaled to roughly 0 to 1.
- **Classes:** `M` (metal cylinder / mine) = 1 with 111 samples, `R` (rock) = 0 with 97 samples.
- **Why this dataset:** it is small and hard enough that hyperparameters visibly change results,
  and the high feature-to-sample ratio makes overfitting easy to observe and discuss.

## Pipeline
1. **Parsing (`DataLoader.cs`):** reads the CSV, skips malformed rows, encodes M = 1 and R = 0.
2. **Split (`Preprocessing.cs`):** stratified 80/20 split with a fixed seed: 167 train (89 mines, 78 rocks)
   and 41 test (22 mines, 19 rocks).
3. **Scaling (`Preprocessing.cs`):** min-max scaling to [0, 1], with min and max learned from the training set only,
   so no information from the test set leaks into training.
4. **Training (`Trainer.cs`):** online (per-sample) backpropagation with the training order reshuffled every epoch.
   Train and test loss and accuracy are logged to CSV.
5. **Evaluation (`Evaluator.cs`):** confusion matrix, accuracy, precision, recall and F1, with "mine" as the positive class.

## Hyperparameter Tuning
Tuning used a validation set carved out of the training data (about 133 train / 34 validation rows),
so the test set was never used to choose settings. The grid was 4 learning rates (0.01, 0.05, 0.15, 0.3)
x 4 hidden sizes (8, 16, 24, 48) x 4 epoch checkpoints (100, 300, 500, 1000), each averaged over 3 random initializations.
Full results are in `hyperparameter_results.csv`.

Findings:
- **Learning rate had the strongest effect.** At 0.3 the network reached 100% training accuracy by epoch 300
  while validation loss rose. At 0.01, training was slower and validation loss stayed near 0.066-0.071.
- **More epochs helped only at low learning rates.** At 0.05 and above, validation loss was lowest around epoch 100
  and increased afterward, which indicates overfitting.
- **Hidden size barely mattered.** At lr 0.01 and 300 epochs, validation loss was 0.0708, 0.0700, 0.0664 and 0.0673
  for 8, 16, 24 and 48 hidden nodes.
- **Chosen configuration:** lr 0.01, 24 hidden nodes, 1000 epochs. Its validation loss was about equal to the best
  configurations (0.0685 vs 0.0664), but with higher validation accuracy (82.4% vs 74.5%).
  This was decided before looking at the test set.

## Results

| Model | LR | Hidden | Epochs | Train Acc | Test Acc | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.15 | 24 | 1000 | 99.40% | 82.93% | 80.00% | 90.91% | 85.11% |
| Tuned | 0.01 | 24 | 1000 | 92.81% | 82.93% | 80.00% | 90.91% | 85.11% |

Test confusion matrix (identical for both models):

| | Predicted Rock | Predicted Mine |
|---|---|---|
| Actual Rock | 14 | 5 |
| Actual Mine | 2 | 20 |

Final test loss at epoch 1000: baseline 0.0814, tuned 0.0712.

## Analysis
- **Backpropagation works.** Training loss decreases steadily in both logs, and accuracy starts near chance (about 46%)
  with an untrained network.
- **The baseline overfits.** It reaches 99.4% training accuracy but 82.9% test accuracy, and its test loss is lowest
  around epoch 100 and then rises while training loss keeps falling.
- **Tuning reduced overfitting but did not change test accuracy.** The train/test gap shrank from about 16.5 to about
  10 percentage points, and validation loss fell from 0.0906 to 0.0685. The test confusion matrix stayed the same.
- **Errors lean toward false alarms.** The model made 5 false alarms (rock called a mine) and 2 misses (mine called a rock).
  For mine detection, misses are the more dangerous error, so the 90.91% recall is the encouraging figure.
  Precision of 80% means about 1 in 5 "mine" predictions is wrong.

## Limitations
- The test set has only 41 samples, so one sample is worth about 2.4 percentage points. Differences of one or two
  samples should not be treated as real improvements.
- Results come from one train/test split. Cross-validation would give a more reliable estimate.
- With only 167 training samples, no tested configuration reduced validation loss much below 0.066.
