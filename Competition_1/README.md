# Competition 1: Beer Quality Prediction 🍺

Welcome to your first machine learning competition! In this challenge, you will work with a dataset containing **beer samples** and their **chemical properties**. Your goal is to train a **classification** model to predict the **quality** of each beer based on those properties.

## Introduction

Beer is one of the most widely consumed alcoholic beverages in the world for almost 4,000 years. This beverage is obtained from the fermentation of starches, mainly derived from cereal grains such as malted barley, wheat, corn, and rice. The brewing process involves several steps, including malting, boiling, fermenting, conditioning, and packaging. Each step can significantly influence the final product's flavor, aroma, and overall quality.

## Problem Statement

### Dataset Description

You are provided with a dataset of beer samples where each sample is characterized by:
- **Chemical attributes**: Various measurable chemical properties of the beer, either categorical or numerical (bitterness_IBU, beer_style, diacetyl_concentration...)
- **Quality score**: A target **integer** variable representing the overall quality rating of the beverage

### Mathematical Formulation

Let's formalize this as a **supervised learning problem**:

- **Input features**: $\mathbf{x} = [x_1, x_2, ..., x_d] \in \mathbb{R}^d$ where $d$ is the number of chemical attributes
- **Target variable**: $y \in \{1, 2, 3, ..., 10\}$ representing **discrete** quality scores from 1 (poor) to 10 (excellent)
- **Training dataset**: $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $n$ training samples

⚠️ This is a **multi-class classification problem** where we aim to predict the quality score, as a **discrete class**, and not a continuous value.

**Evaluation Metric**: Your model's performance will be evaluated using **accuracy**, defined as:

$$\text{Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}_{[\hat{y}_i = y_i]}$$

where $\mathbb{1}_{[\hat{y}_i = y_i]}$ is the indicator function that equals 1 when the prediction $\hat{y}_i$ matches the true label $y_i$, and 0 otherwise.

### Competition Format 🏆

This competition follows a **two-phase evaluation system**: You will submit your predictions on [Kaggle](https://www.kaggle.com/competitions/ift-6390-ift-3395-beer-quality-prediction/). Your model will be evaluated on the provided test set (test.csv). This test set is divided into two parts: a **public test set** and a **private test set**. You can make up to 3 submissions per day.

**Public Leaderboard**: Your performance on the public test set will be displayed at any time on the public leaderboard, which gives you immediate feedback on how well your model is performing. Use this to iterate and improve your approach.

**Private Leaderboard**: Your performance on the private test set will determine your final ranking in the competition. The leaderboard is hidden during the competition to prevent overfitting and displayed only at the end of the competition.

**Important**: Your final score will be determined by performance on the **private test set** only, so focus on building robust, generalizable models rather than just optimizing for the public leaderboard!

## Important Deadlines

- **October 25, 2025**: Beat the different baselines using **only methods seen in class**
- **November 8, 2025**: Final submission - submit your best model and compete against each other

## Getting Started
Before you begin exploring the data and building your models, you'll need to set up your python environment with the required dependencies.

### 📋 Environment Setup

Make sure you have Python 3.9 or higher installed and pip. You can use either Conda or Virtual Environment (venv) to manage your dependencies.

#### Option 1: Using Conda 🐍
```bash
# Create a new conda environment
conda create -n .venv python=3.9

# Activate the environment
conda activate .venv

# Install requirements
pip install -r requirements.txt
```

#### Option 2: Using Virtual Environment (venv) 📦
```bash
# Create a virtual environment
python3.9 -m venv .venv

# Activate the environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 📁 Project Structure

```
Competition_1/
├── README.md                  # This file
├── requirements.txt           # Required packages
├── notebook.ipynb             # Main notebook for training your model
├── data/                      # Dataset folder
│   │── train.csv              # Training data in csv format
│   │── test.csv               # Test data in csv format
│   └── sample_submission.csv  # Sample submission file
```

## Your Mission 

1. **Explore** both the train and test datasets and understand the properties of each attribute (missing data, redundant or unbalanced features)
2. **Preprocess** the data (scaling, normalization, encoding of non-numerical features, etc. be creative!)
3. **Train** various machine learning models seen during class
4. **Evaluate** model performance using appropriate metrics
5. **Generate** predictions for the test set and submit them to Kaggle

### Data Exploration and Visualization

The first step in any machine learning project is to load and take a look at the data. You can for example:

- Check for features types (numerical, categorical) and min/max values
- Visualize the distribution of certain features, conditional distributions, correlations, etc.
- Visually check for outliers or anomalies, and linear separability of classes

### Data Preprocessing

Depending on the method you opt for, we might need to preprocess our data (cleaning, transforming, etc.). Here are some common preprocessing steps that you might want to implement:

1. **Handling Missing Values**: Identify and appropriately handle any missing values in the dataset, either by imputation or removal.
2. **Encoding Categorical Variables**: Convert categorical variables into a numerical format using techniques such as one-hot encoding or label encoding.
3. **Feature Scaling**: Normalize or standardize numerical features to ensure they are on a similar scale, which is crucial for many machine learning algorithms.
4. **Feature Selection**: Identify and retain the most relevant features that contribute significantly to the prediction task, potentially using techniques like correlation analysis or feature importance from models.
5. **Any additional preprocessing steps you like want to try, be creative!**

### Training and Validation Split

No matter the approach, we suggest you to always held out some data from the training set to evaluate your model's performance and perform hyper-parameter tuning:

$$\mathcal{D} = \mathcal{D}_{train} \cup \mathcal{D}_{val}$$

Where:
- $\mathcal{D}_{train}$: Used for training models (80% of data)
- $\mathcal{D}_{val}$: Used for evaluation and model selection (20% of data)

### Model Training and Evaluation

Train multiple machine learning models and compare their performance. Each model learns a different type of decision boundary.

Once that you have trained and evaluated different models, you should pick the best one and analyze it in more detail. Here are some suggestions on what to include in your analysis:

1. **Model Performance**: Report the performance metrics (accuracy, precision, recall, F1-score) on both the training and validation sets. Discuss any discrepancies between training and validation performance.
2. **Confusion Matrix**: Present a confusion matrix to visualize the model's performance across different quality classes. Identify which classes are most frequently misclassified.
3. **Feature Importance**: If applicable, analyze the importance of different features in the model. You can for example, discuss which chemical properties are most influential in predicting beer quality.
4. **Error Analysis**: Investigate the instances where the model made incorrect predictions. Look for patterns in the misclassifications and consider potential reasons for these errors.

## Submission Instructions

### Kaggle Submission Format

Your submission file must follow the format of `sample_submission.csv`, which is a CSV file with two columns and a header row:
- **id**: The identifier for each test sample
- **quality**: Your predicted quality score (integer from 1 to 10)

Example format:
```
id,quality
1,7
2,5
3,8
...
```

Make sure your predictions are integers in the range [1, 10] and that you include predictions for all test samples.

## Team Formation and Kaggle Setup

### Team Size Requirements
- **IFT3395 students**: Can work in teams of **up to 2 students**
- **IFT6390 students**: Must work **individually**

### Kaggle Team Name Format

Your Kaggle team name **must** follow this format:

**For IFT3395 teams:**
- Team of 1: `ift3395_firstname_lastname`
- Team of 2: `ift3395_firstname1_lastname1_firstname2_lastname2`

**For IFT6390 students:**
- `ift6390_firstname_lastname`

**Examples:**
- `ift3395_mehdi-inane_ahmed`
- `ift3395_mehdi-inane_ahmed_tom_marty`
- `ift6390_charlie_tremblay`

⚠️ **Important**: Using the correct naming format is mandatory for proper grade attribution.

## Report Requirements

In addition to implementing your method, you must write a short report that details your methodology for solving this problem and provides the results of your method. Specifically, your report must contain the following information:

1. **Kaggle Team Name** (following the format specified above: `ift3395_...` or `ift6390_...`), along with the list of team members (full name and student ID)
2. **Feature Preprocessing (Feature Design)**: Describe and justify your feature preprocessing steps and indicate which features you selected for your model.
3. **Methodology**: Describe and justify all decisions regarding the split of data into training and validation sets, as well as the techniques used to improve the performance (regularization strategy, hyperparameter tuning, etc.)
4. **Results**: Present a concise analysis of your results using tables or graphs.
5. **Discussion**: Comment on your results and indicate the advantages and disadvantages of your approach and methodology.

The report must not exceed **2 pages**. You are free to structure the report as you wish as long as you include the elements mentioned above. Introduction, problem description, and conclusion sections are not mandatory.

### Submission Deliverables

Submit the following on **Gradescope** (link will be provided on Piazza) before the final deadline (November 5, 2025):
1. **Report** (PDF format, max 2 pages)
2. **Code** (Jupyter notebook or Python scripts)

⚠️ **Important**: Make sure your Kaggle team name matches what you write in your report!

## Grading (Total 100 points)

### **Data Competition (60 points)**
- **20 points**: Beat the Random baseline model
- **30 points**: Beat the Strong baseline model (This should be achieved only using methods seen in class)
- **5 points**: Rank above the median performance
- **5 points**: Achieve top 3 ranking

**Note**: Baseline model predictions are available in the `Leaderboard` section on Kaggle.

### **Written Report (30 points)**
- **6 points**: Format and presentation
- **8 points**: Algorithms description and justification
- **8 points**: Methodology and experimental design
- **8 points**: Discussion of results and analysis

### **Code Submission (10 points)**
- **2 points**: Well-commented code
- **4 points**: Code readability and organization
- **4 points**: Documentation on how to run the code (README, instructions, etc.)

##  Tips for Success

- Start early 
- Don't forget about data exploration, preprocessing and feature engineering (this is the most important part)
- Try different models and compare their performance 
- Document your approach and findings 
- Have fun and learn something new! 


## Competition Rules 

The goal for this competition, is for you to have the opportunity to learn the key aspects and subleties of what makes a good classification methods, from data-analysis, preprocessing, to model training and hyperparameter selection.

- Use generative AI responsibly, your ranking only account for 10% of the total grade!
- All UdeM rules on plagiarism applies.
---

Good luck with the competition! 🏅
