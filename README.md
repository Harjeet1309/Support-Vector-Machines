# Breast Cancer Classification using Support Vector Machines (SVM)

## Dataset Description

- **Source:** `breast.csv`
- **Target variable:** `diagnosis` (B = Benign, M = Malignant)
- **Features:** Various numerical measurements derived from images of breast masses (e.g., radius, texture, perimeter, area, etc.)
- **Label Encoding:** 'B' → 0, 'M' → 1

---

## 🛠️ Technologies Used

- **Python**
- **Pandas, NumPy** – data manipulation
- **Matplotlib, Seaborn** – data visualization
- **scikit-learn** – machine learning (SVM, scaling, grid search, cross-validation, evaluation)

---

## Methodology

### Load and Preprocess Data
- Removed null values and unnecessary unnamed columns.
- Encoded the categorical target (`diagnosis`) using LabelEncoder.
- Standardized features for SVM training using `StandardScaler`.

### Feature Selection for 2D Visualization
- Selected two key features: `radius_mean` and `texture_mean`.
- These were used to visualize decision boundaries of the trained models.

### Model Training
- Trained two SVM models:
  - **Linear Kernel SVM**
  - **RBF Kernel SVM**
- Models were trained on both:
  - Full feature set (for real prediction performance)
  - 2D feature set (for visualization)

### Decision Boundary Visualization
- Plotted decision boundaries for both linear and RBF SVMs using the 2D feature set.
- Clear separation of classes visualized using contour plots.

### Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold cross-validation to find best values of:
  - `C` (penalty parameter)
  - `gamma` (kernel coefficient for RBF)
- Best parameters printed to console.

### Cross-Validation Evaluation
- Applied `cross_val_score` on full dataset using the best RBF model.
- Computed and displayed accuracy for each fold and mean accuracy.

