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

## Troubleshooting

If you encounter any issues:

1. Make sure all required dependencies are installed.
2. Check that the `markup_data.csv` file exists in the `data/` directory.
3. Verify that the `markup_data.csv` file contains the required columns.
4. If the dataset is too large, try running with a smaller subset.

## License

This project is provided as-is with no warranty. Use at your own risk.
