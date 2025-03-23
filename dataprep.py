import pandas as pd

# Load the IMDb dataset
def load_data(dataset):
    data = pd.read_csv(dataset)
    return data

# Preprocess the data
def preprocess_data(data):
    # Clean the text (basic cleaning)
    data['review'] = data['review'].str.replace(r'http\S+|@\S+|#\S+', '', case=False)  # Remove URLs, mentions, hashtags
    data['review'] = data['review'].str.replace(r'[^a-zA-Z\s]', '', case=False)  # Remove special characters
    data['review'] = data['review'].str.lower()  # Convert to lowercase
    return data

# Main execution
def main():
    # Load the dataset
    data = load_data('dataset.csv')
    
    # Preprocess the data
    data = preprocess_data(data)

    # Show the first few rows of the dataset after loading and preprocessing
    print("Sample data after preprocessing:")
    print(data.head(10))  # Display the first 10 rows

    # Display total count of labels and sentiments
    print("\nTotal Count of Sentiments (positive/negative):")
    print(f"Positive labels: {data['sentiment'].value_counts().get('positive', 0)}")
    print(f"Negative labels: {data['sentiment'].value_counts().get('negative', 0)}")
    
    # Randomly select 5000 reviews
    data_sampled = data.sample(n=5000, random_state=42)

    # Save the sampled dataset to a new CSV file
    data_sampled.to_csv('sampled_imdb.csv', index=False)
    
    print("Data preparation complete! Sampled 5000 reviews saved to 'sampled_imdb.csv'.")

    # Count positive and negative reviews in the sampled data
    positive_reviews = data_sampled[data_sampled['sentiment'] == 'positive']
    negative_reviews = data_sampled[data_sampled['sentiment'] == 'negative']
    
    # Display counts before undersampling
    print("\nSampled Data Sentiment Counts (Before Undersampling):")
    print(f"Positive Reviews: {len(positive_reviews)}")
    print(f"Negative Reviews: {len(negative_reviews)}")
    
    # Undersample the larger class
    min_count = min(len(positive_reviews), len(negative_reviews))
    
    positive_reviews_undersampled = positive_reviews.sample(n=min_count, random_state=42)
    negative_reviews_undersampled = negative_reviews.sample(n=min_count, random_state=42)
    
    # Combine the undersampled positive and negative reviews
    balanced_data = pd.concat([positive_reviews_undersampled, negative_reviews_undersampled])

    # Shuffle the balanced dataset
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the balanced dataset to a new CSV file
    balanced_data.to_csv('balanced_sampled_imdb.csv', index=False)
    
    print("Undersampling complete! Balanced data saved to 'balanced_sampled_imdb.csv'.")

    # Display the counts after undersampling
    print("\nBalanced Data Sentiment Counts (After Undersampling):")
    print(balanced_data['sentiment'].value_counts())

if __name__ == "__main__":
    main()
