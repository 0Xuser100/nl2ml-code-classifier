# SVM Training for Code Block Classification

This project trains an SVM model to classify code blocks into different categories (class, function, variable, loop, condition) using TF-IDF features.
it is based on this paper https://peerj.com/articles/cs-1230/

## Project Structure

```
project/
├── data/
│   └── markup_data.csv       # Original dataset with code blocks and labels
├── graph/                    # Graph definition files
├── models/                   # Trained model files
├── common/                   # Common utility functions
├── svm_train.py              # Main SVM training script
├── running_svm.ipynb         # Jupyter notebook to run the training
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Required Folders

The following folders are needed:
- `data/` - Contains the input data files
- `graph/` - Contains graph definition files
- `models/` - Where trained models will be saved
- `common/` - Contains utility functions used by the scripts

The notebook will create these folders if they don't exist.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have the `markup_data.csv` file in the `data/` directory. This file should contain at least the following columns:
   - `code_block` - The code snippet to classify
   - `graph_vertex_id` - The label/category of the code block

## Running the Code

### Using the Jupyter Notebook

1. Start Jupyter Notebook:

```bash
jupyter notebook
```

2. Open the `running_svm.ipynb` notebook.

3. Run all cells in the notebook by clicking "Cell" > "Run All" or by running each cell individually.

The notebook will:
- Read the original `markup_data.csv` file
- Select only the `code_block` and `graph_vertex_id` columns
- Create a temporary file with these columns using semicolon as the delimiter
- Modify the SVM training script for the dataset
- Run the SVM training script
- Verify that the model files were created
- Clean up temporary files

### Changes Made to the Original Code

The main changes made to the original code are:

1. **Direct Column Selection**: Instead of creating a separate sample file, the notebook selects only the required columns (`code_block` and `graph_vertex_id`) during runtime.

2. **Temporary File Handling**: The selected columns are saved to a temporary file that is automatically deleted after training.

3. **Error Handling**: If there's an error reading the original dataset, the notebook creates a small sample dataset instead.

4. **Hyperparameter Adjustments**: The SVM hyperparameters are adjusted for better performance on the dataset:
   - `tfidf__min_df` set to 1 for small datasets
   - `tfidf__max_df` set to 1.0 (100%) for small datasets
   - KFold cross-validation splits reduced to 2 for small datasets

5. **Visualization**: Added data visualization to understand the distribution of labels.

## Output Files

After running the notebook, the following files will be created:

1. `../models/svm_linear_search_graph_v1.sav` - The trained SVM model
2. `../models/tfidf_hyper_svm_graph_v1.pickle` - The TF-IDF vectorizer

These files can be used for making predictions on new code blocks.

## Code Explanation

### SVM Training Process

The SVM training process involves:

1. **Data Preparation**: Reading the dataset and selecting the required columns.

2. **TF-IDF Vectorization**: Converting the code blocks into numerical features using TF-IDF.

3. **Hyperparameter Selection**: Using either default hyperparameters or Optuna for hyperparameter optimization.

4. **Model Training**: Training an SVM model with the selected hyperparameters.

5. **Cross-Validation**: Evaluating the model using K-fold cross-validation.

6. **Model Saving**: Saving the trained model and TF-IDF vectorizer for later use.

### Key Components

- **svm_train.py**: The main script that trains the SVM model.
- **running_svm.ipynb**: The Jupyter notebook that orchestrates the training process.
- **common/tools.py**: Contains utility functions for data loading, TF-IDF transformation, and cross-validation.

### Alternative SVM Implementations

The codebase includes several SVM variants for different use cases:

1. **svm_hyperparam_train.py**: Uses Bagging Classifier with SVM as the base estimator for improved performance.

2. **svm_augment_train.py**: Implements data augmentation by masking variable names in code blocks, which can help the model generalize better.

3. **svm_for_semi.py**: Implements semi-supervised learning, allowing the model to learn from both labeled and unlabeled data.

4. **hierarchy_svm_svm.py**: Implements a hierarchical SVM approach with two levels of classification, which can be useful for complex classification tasks.

Each implementation has its own advantages and can be used depending on the specific requirements of your task.

## Using the Trained Model for Prediction

Once you have trained the SVM model, you can use it to make predictions on new code blocks. The following Python code demonstrates how to load the trained model and make predictions:

```python
import pickle
import pandas as pd
from common.tools import tfidf_transform

# Paths to the trained model and TF-IDF vectorizer
MODEL_PATH = "../models/svm_linear_search_graph_v1.sav"
TFIDF_PATH = "../models/tfidf_hyper_svm_graph_v1.pickle"

# Load the trained model
clf = pickle.load(open(MODEL_PATH, 'rb'))
print("Model loaded successfully")

# Example new code blocks to classify
new_code_blocks = [
    "class MyClass:\n    def __init__(self, name):\n        self.name = name",
    "def calculate_sum(a, b):\n    return a + b",
    "x = 10\ny = 20\nz = x + y"
]

# Convert to DataFrame
df_new = pd.DataFrame({"code_block": new_code_blocks})

# Transform the new code blocks using the trained TF-IDF vectorizer
# Note: We don't need to pass tfidf_params here as the vectorizer is already trained
features = tfidf_transform(df_new["code_block"], {}, TFIDF_PATH)

# Make predictions
predictions = clf.predict(features)
print("Predictions:", predictions)

# Map predictions to class names if needed
# For example, if your classes are: 0=class, 1=function, 2=variable, etc.
class_names = ["class", "function", "variable", "loop", "condition"]
predicted_classes = [class_names[pred] if pred < len(class_names) else "unknown" for pred in predictions]
print("Predicted classes:", predicted_classes)
```

### Alternative Method Using get_metrics Function

You can also use the `get_metrics` function from `common/tools.py` if you have ground truth labels for evaluation:

```python
from common.tools import get_metrics, tfidf_transform

# Load the model and make predictions
X = tfidf_transform(df_new["code_block"], {}, TFIDF_PATH)
y = df_new["graph_vertex_id"].values  # If you have ground truth labels
TAGS_TO_PREDICT = ["class", "function", "variable", "loop", "condition"]  # Your class labels

# Get predictions and evaluation metrics
X, y, y_pred, metrics = get_metrics(X, y, TAGS_TO_PREDICT, MODEL_PATH)
print(f"Accuracy: {metrics['test_accuracy']*100:.2f}%")
print(f"F1 Score: {metrics['test_f1_score']*100:.2f}%")
```

### Related Files

The following files are involved in the prediction process:

- `common/tools.py`: Contains utility functions for loading models and transforming data
  - `tfidf_transform()`: Transforms new code blocks using the trained TF-IDF vectorizer
  - `get_metrics()`: Makes predictions and calculates evaluation metrics

## Troubleshooting

If you encounter any issues:

1. Make sure all required dependencies are installed.
2. Check that the `markup_data.csv` file exists in the `data/` directory.
3. Verify that the `markup_data.csv` file contains the required columns.
4. If the dataset is too large, try running with a smaller subset.
5. When making predictions, ensure that the model and TF-IDF vectorizer files exist in the specified paths.
6. If you get errors during prediction, check that your new code blocks have similar formatting to the training data.

## License

This project is provided as-is with no warranty. Use at your own risk.
