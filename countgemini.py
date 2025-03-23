<<<<<<< HEAD
from dataprep import *
from gemini import *
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the balanced dataset
balanced_data = pd.read_csv('balanced_sampled_with_gemini_sentiment.csv')

# Clean the 'gemini_sentiment' column to remove any extra text if needed
balanced_data['gemini_sentiment_cleaned'] = (
    balanced_data['gemini_sentiment']
    .str.replace(r'^\d+\.\s*', '', regex=True)  # Remove leading numbers and dots
    .str.strip()  # Remove any extra whitespace
    .str.lower()  # Convert to lowercase for consistency
)

# Filter out invalid or non-standard values in the cleaned 'gemini_sentiment' column
valid_gemini_sentiments = balanced_data['gemini_sentiment_cleaned'].isin(['positive', 'negative'])
filtered_data = balanced_data[valid_gemini_sentiments]

# Count missed sentiments (invalid or non-standard Gemini sentiments)
missed_sentiments_count = len(balanced_data) - len(filtered_data)

# Count the cleaned and filtered sentiment values
gemini_sentiment_counts = filtered_data['gemini_sentiment_cleaned'].value_counts()
sentiment_counts = balanced_data['sentiment'].value_counts()

# Calculate the total number of valid Gemini sentiments and original sentiments
total_gemini_sentiments = gemini_sentiment_counts.sum()
total_sentiments = sentiment_counts.sum()

# Print the sentiment counts and totals
print("Filtered Gemini Sentiment Counts (positive/negative only):")
print(gemini_sentiment_counts)

print("\nOriginal Sentiment Counts:")
print(sentiment_counts)

print(f"\nTotal valid Gemini sentiments analyzed: {total_gemini_sentiments}")
print(f"Total original sentiments analyzed: {total_sentiments}")
print(f"\nNumber of missed sentiments (invalid or non-standard): {missed_sentiments_count}")

# Extract ground truth and predictions
y_true = filtered_data['sentiment'].str.lower()  # Original sentiments
y_pred = filtered_data['gemini_sentiment_cleaned']  # Gemini-generated sentiments

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=['positive', 'negative'])

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary', pos_label='positive')
recall = recall_score(y_true, y_pred, average='binary', pos_label='positive')
f1 = f1_score(y_true, y_pred, average='binary', pos_label='positive')

# Binary encoding for ROC curve calculation
y_true_binary = (y_true == 'positive').astype(int)  # Encode 'positive' as 1, 'negative' as 0
y_pred_binary = (y_pred == 'positive').astype(int)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_binary)
roc_auc = roc_auc_score(y_true_binary, y_pred_binary)

# Print metrics
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}\n")
print(f"ROC-AUC: {roc_auc:.4f}\n")

print("Confusion Matrix:")
print(f"True Positive (TP): {conf_matrix[0, 0]}")
print(f"False Positive (FP): {conf_matrix[1, 0]}")
print(f"True Negative (TN): {conf_matrix[1, 1]}")
print(f"False Negative (FN): {conf_matrix[0, 1]}")

# Plotting the confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Misclassification Analysis
filtered_data['misclassified'] = (y_true != y_pred)

# Print sample misclassified rows
print("\nSample Misclassified Examples:")
print(filtered_data[filtered_data['misclassified']][['sentiment', 'gemini_sentiment_cleaned']].head(10))
=======
from dataprep import *
from gemini import *
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the balanced dataset
balanced_data = pd.read_csv('balanced_sampled_with_gemini_sentiment.csv')

# Clean the 'gemini_sentiment' column to remove any extra text if needed
balanced_data['gemini_sentiment_cleaned'] = (
    balanced_data['gemini_sentiment']
    .str.replace(r'^\d+\.\s*', '', regex=True)  # Remove leading numbers and dots
    .str.strip()  # Remove any extra whitespace
    .str.lower()  # Convert to lowercase for consistency
)

# Filter out invalid or non-standard values in the cleaned 'gemini_sentiment' column
valid_gemini_sentiments = balanced_data['gemini_sentiment_cleaned'].isin(['positive', 'negative'])
filtered_data = balanced_data[valid_gemini_sentiments]

# Count missed sentiments (invalid or non-standard Gemini sentiments)
missed_sentiments_count = len(balanced_data) - len(filtered_data)

# Count the cleaned and filtered sentiment values
gemini_sentiment_counts = filtered_data['gemini_sentiment_cleaned'].value_counts()
sentiment_counts = balanced_data['sentiment'].value_counts()

# Calculate the total number of valid Gemini sentiments and original sentiments
total_gemini_sentiments = gemini_sentiment_counts.sum()
total_sentiments = sentiment_counts.sum()

# Print the sentiment counts and totals
print("Filtered Gemini Sentiment Counts (positive/negative only):")
print(gemini_sentiment_counts)

print("\nOriginal Sentiment Counts:")
print(sentiment_counts)

print(f"\nTotal valid Gemini sentiments analyzed: {total_gemini_sentiments}")
print(f"Total original sentiments analyzed: {total_sentiments}")
print(f"\nNumber of missed sentiments (invalid or non-standard): {missed_sentiments_count}")

# Extract ground truth and predictions
y_true = filtered_data['sentiment'].str.lower()  # Original sentiments
y_pred = filtered_data['gemini_sentiment_cleaned']  # Gemini-generated sentiments

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=['positive', 'negative'])

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary', pos_label='positive')
recall = recall_score(y_true, y_pred, average='binary', pos_label='positive')
f1 = f1_score(y_true, y_pred, average='binary', pos_label='positive')

# Binary encoding for ROC curve calculation
y_true_binary = (y_true == 'positive').astype(int)  # Encode 'positive' as 1, 'negative' as 0
y_pred_binary = (y_pred == 'positive').astype(int)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_binary)
roc_auc = roc_auc_score(y_true_binary, y_pred_binary)

# Print metrics
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}\n")
print(f"ROC-AUC: {roc_auc:.4f}\n")

print("Confusion Matrix:")
print(f"True Positive (TP): {conf_matrix[0, 0]}")
print(f"False Positive (FP): {conf_matrix[1, 0]}")
print(f"True Negative (TN): {conf_matrix[1, 1]}")
print(f"False Negative (FN): {conf_matrix[0, 1]}")

# Plotting the confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Misclassification Analysis
filtered_data['misclassified'] = (y_true != y_pred)

# Print sample misclassified rows
print("\nSample Misclassified Examples:")
print(filtered_data[filtered_data['misclassified']][['sentiment', 'gemini_sentiment_cleaned']].head(10))
>>>>>>> 23ed1d090e8796675969bde3c102083c8670e7a6
