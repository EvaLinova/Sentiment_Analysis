import pandas as pd
import google.generativeai as genai
import time
import random
import psutil  # For monitoring memory usage
import os
from dataprep import *  # Import your preprocessing functions
import re

# Configure the Gemini API with your API key
genai.configure(api_key="Here should be the key.")

# Function to parse and validate the sentiments in the response
def parse_sentiments(response_text, expected_count):
    # Extract lines starting with a number and a period (e.g., "1. positive")
    pattern = r"^\d+\.\s*(positive|negative)"
    matches = re.findall(pattern, response_text, re.MULTILINE)

    # Handle extra or missing sentiments
    if len(matches) > expected_count:
        print(f"Warning: Received {len(matches)} sentiments but expected {expected_count}. Truncating extra entries.")
        matches = matches[:expected_count]
    elif len(matches) < expected_count:
        print(f"Warning: Missing {expected_count - len(matches)} sentiments. Filling with None.")
        matches.extend([None] * (expected_count - len(matches)))

    return [match.lower() if match else None for match in matches]

# Function to process sentiments for a batch of reviews
def get_batch_sentiments(reviews_batch, batch_index):
    batch_reviews = "\n".join([f"{i + 1}. {review}" for i, review in enumerate(reviews_batch)])
    prompt = (
        "You are a sentiment analysis assistant. "
        "Please classify each of the following movie reviews as 'positive' or 'negative'. "
        "Ensure there is one sentiment for each review in the format:\n"
        "1. positive\n2. negative\n...\n"
        f"{batch_reviews}\nSentiments:"
    )
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        
        # Parse and validate the response
        sentiments = parse_sentiments(response.text, len(reviews_batch))
        return sentiments
    except Exception as e:
        print(f"Error processing batch {batch_index}: {e}")
        return [None] * len(reviews_batch)

# Retry function for mismatched or failed batches
def retry_batch(reviews_batch, batch_index, retries=3):
    for attempt in range(retries):
        print(f"Retrying batch {batch_index}... Attempt {attempt + 1}")
        sentiments = get_batch_sentiments(reviews_batch, batch_index)
        if all(sentiments):
            return sentiments
        time.sleep(1)  # Add a small delay between retries
    return [None] * len(reviews_batch)

# Function to add rate limiting and delay
def process_reviews_with_rate_limit(balanced_data, batch_size=10, delay=1, retries=3):
    sentiments = []
    start_time = time.time()  # Start the timer

    print("Starting batch sentiment processing...\n")

    # Track initial memory usage
    initial_memory = psutil.virtual_memory().percent
    print(f"Initial memory usage: {initial_memory}%")

    total_processed = 0  # To keep track of the total number of reviews processed

    for i in range(0, len(balanced_data), batch_size):
        reviews_batch = balanced_data['review'].iloc[i:i + batch_size].tolist()
        batch_index = i // batch_size + 1

        # Process the batch with retry logic
        batch_sentiments = retry_batch(reviews_batch, batch_index, retries=retries)
        sentiments.extend(batch_sentiments)

        # Increment total processed count
        total_processed += len(reviews_batch)

        # Apply rate limiting
        time.sleep(delay + random.uniform(0, 2))  # Add random jitter

        # Check memory usage after each batch
        current_memory = psutil.virtual_memory().percent
        print(f"Current memory usage after batch {batch_index}: {current_memory}%")

    # End the timer
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nBatch processing complete. Total time taken: {total_time:.2f} seconds.")

    # Print final memory usage
    final_memory = psutil.virtual_memory().percent
    print(f"Final memory usage: {final_memory}%")

    return sentiments

# Main execution
def main():
    # Load the balanced dataset
    try:
        balanced_data = pd.read_csv('balanced_sampled_imdb.csv')
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return

    # Process the reviews with rate limiting
    sentiments = process_reviews_with_rate_limit(balanced_data, batch_size=10, delay=2, retries=3)

    # Add the sentiments to the dataframe
    balanced_data['gemini_sentiment'] = sentiments

    # Final review processing check
    total_reviews = len(balanced_data)
    total_sentiments = len(sentiments)

    # Summary message
    if total_sentiments == total_reviews:
        print(f"\nAll {total_reviews} reviews have been evaluated successfully.")
    else:
        print(f"\nWarning: Some reviews were not evaluated. Expected {total_reviews} reviews, but got {total_sentiments} sentiments.")

    # Ensure there are no 'None' values (this means no errors during processing)
    missing_sentiments = balanced_data['gemini_sentiment'].isnull().sum()
    if missing_sentiments > 0:
        print(f"Warning: {missing_sentiments} reviews have missing sentiments.")
    else:
        print("All reviews have valid sentiment values.")

    print("\nSample data with Gemini sentiment:")
    print(balanced_data[['review', 'gemini_sentiment']].head())

    # Save the updated dataset
    output_file = 'balanced_sampled_with_gemini_sentiment.csv'
    balanced_data.to_csv(output_file, index=False)
    print(f"\nSentiment analysis complete! Saved to '{output_file}'.")

if __name__ == "__main__":
    main()
