# Duplicate Question Detection Using Machine Learning

This project focuses on building a machine learning pipeline to detect duplicate questions using advanced preprocessing, feature engineering, and state-of-the-art classification algorithms. The dataset used for this project contains 400,000+ rows, providing a challenging and rich environment to test the robustness of the pipeline.

# Key Features
`1. Dataset`

- The dataset contains question pairs and their labels (1 for duplicates, 0 for non-duplicates).

- Size: 400,000+ rows, making it suitable for testing scalability and performance.

`2. Preprocessing Techniques`

- Basic Text Cleaning:
    - Removed unnecessary symbols, lowercased text, and stripped whitespace.

- Stopword Removal:
    - Reduced noise in the data by removing common words using the NLTK library.

- Tokenization:
    - Split questions into tokens for further analysis.


`3. Feature Engineering`

a. Length-Based Features

- Length of Questions: Captures the total and absolute difference in length.

- Word Counts: Counts and compares the number of words in both questions.

- Longest Substring Ratio: Measures the ratio of the longest common substring.


b. Token Features

- Common words, stopwords, and token counts (e.g., common words divided by the minimum and maximum token counts).

- Binary features indicating if the first/last words are the same.


c. Fuzzy Features

Used the FuzzyWuzzy library to calculate:
- Fuzzy Ratio
- Partial Ratio
- Token Sort Ratio
- Token Set Ratio
  
Calculated shared words and common token metrics.

`4. Algorithms Used`

- Random Forest Classifier
  
Trained on the engineered features using batch-wise training to handle large data efficiently.
Performed better on confusion matrix analysis, especially in identifying false negatives.

- XGBoost Classifier
  
Gradient boosting technique for efficient handling of large-scale data.
Provided 80% accuracy, comparable to Random Forest.

`5. Evaluation`
- Achieved 80% accuracy with both Random Forest and XGBoost.
- Confusion Matrix Analysis:
Random Forest had better precision and recall for duplicate questions.
