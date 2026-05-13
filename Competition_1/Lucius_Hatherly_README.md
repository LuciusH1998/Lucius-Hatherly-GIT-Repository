# Competition 1: Beer Quality Prediction 🍺

**Author:** Lucius Hatherly  
**Team Name:** `ift6390_LuciusHatherly_20294880`  
**Institution:** Université de Montréal  (MILA)
**Course:** Machine Learning (IFT6390)  

## Overview/Introduction

This submitted repository contains all code and documentation for the **Beer Quality Prediction** Kaggle competition. The objective is to predict the **beer quality score (0–10)** using the inputted fermentation and chemical features.  

This project includes:
- Data cleaning, outlier handling, and encoding  
- Feature engineering and correlation analysis  
- Model training and hyperparameter tuning  
- Model evaluation (accuracy, F1, precision, recall, error)  
- Kaggle submission file generation  

---

# Note on Requirements 
To properly execute the code, it is recommended you have Python 3.10+ and Jupyter Notebook installed

## Structure of the Repository 

```
Competition_1/
├── Competition_1_Lucius_Hatherly.ipynb # Code used for processing, training, tuning, and evaluation
├── rf_additional_feat_eng.csv # Kaggle submission (for feature engineering after split)
├── submission_best_random_forest.csv # Kaggle submission (best tuned Random Forest)
├── data/                      # Dataset folder
│   │── train.csv              # Training data in csv format
│   │── test.csv               # Test data in csv format
│   └── sample_submission.csv  # Sample submission file
├── Lucius Hatherly Beer Quality Kaggle Report # Final report 
├── requirements.txt           # Dependencies needed for code to run 

```

## How to Run 

Step 1: Data Preparation 

Download the official Kaggle data files including: 1. train.csv and 2. test.csv

Place these two files inside in a folder called data located in the project root 

/BeerQuality_Kaggle_Submission/
├── data/
│   ├── train.csv
│   └── test.csv
├── Competition_1_Lucius_Hatherly.ipynb
├── requirements.txt
├── README.md
├── ...

Step 2: Install dependencies (Optional as dependencies can be downloaded in the code blocks of Competition_1_Lucius_Hatherly.ipynb where they are imported and run in Visual Studio.) 

1. Create and activate a Python environment (recommended Python 3.10+):
python -m venv venv
venv\Scripts\activate      # on Windows  
source venv/bin/activate   # on macOS / Linux

2. Install all required libraries 

pip install -r requirements.txt

This ensures that all dependencies (NumPy, Pandas, jupyter, ipython, ipykernel, Matplotlib, Seaborn, and scikit-learn) are correctly installed. 

Step 3: Run the Notebook

Open the notebook directly in Jupyter Notebook or VS Code and click Run All.
This will:

Load and preprocess the dataset

Train and evaluate all models

Produce validation results and Kaggle submission files automatically

Note, if you open the notebook and run it directly in VS Code, you do not need to create an environment, just ensure you have Python 3.10 + installed and selected as the active kernel. 

Optional (Command-Line Execution)

If you prefer running it headlessly:

python -m jupyter nbconvert --to notebook --execute Competition_1_Lucius_Hatherly.ipynb --output=Competition_1_Lucius_Hatherly_executed.ipynb

These two approaches ensure that the code is properly executed. 

Step 4: Output Viewing 

Upon completion, the ipynb code will:

Clean and preprocess the dataset
Perform feature engineering and label encoding
Split the data into 80% training and 20% validation sets
Train multiple supervised learning models (Logistic Regression, SVM, kNN, Naïve Bayes, Random Forest)
Tune model hyperparameters via 5-fold stratified cross-validation
Evaluate all models using accuracy, precision, recall, F1-score, and confusion matrices
Generate Kaggle-ready CSV submission files, including:

submission_best_random_forest.csv

rf_additional_feat_eng.csv

## Notes
- All random seeds were fixed (`random_state=42`) to ensure reproducibility.
- The notebook automatically produces the submission files:
  1. LogisticRegression_before_October_25.csv
  2. Random_forest_before_Oct25.csv
  3. submission_best_random_forest.csv
  4. Post_October_25_Random_Forest_AFE.csv
- All of these files can be uploaded directly to Kaggle.






