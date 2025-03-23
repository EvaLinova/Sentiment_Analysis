<<<<<<< HEAD
<<<<<<< HEAD
import pandas as pd
import openai
import time
import random
import psutil # For monitoring memory usage
from dataprep import *  # Import your preprocessing functions

# Set your OpenAI API key
openai.api_key = 'Here should be the key.'

# Function to process sentiments for a batch of reviews
def get_batch_sentiments(reviews_batch):
    # Prepare a single prompt with multiple reviews
    batch_reviews = "\n".join([f"{i + 1}. {review}" for i, review in enumerate(reviews_batch)])
    prompt = f"Classify the sentiment of the following movie reviews as 'positive' or 'negative':\n{batch_reviews}\nSentiments:"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5 Turbo model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that does sentiment analysis. You classify every review as 'positive' or 'negative' only. There's no neutral sentiment."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.0,  # Ensures consistent and reliable response
        )
        sentiments = response['choices'][0]['message']['content'].strip().split('\n')
        return [sentiment.strip().lower() for sentiment in sentiments] #pro každý sentiment v poli sentiments se provede strip a lower funkce
    except Exception as e:
        print(f"Error processing batch of reviews: {e}")
        return [None] * len(reviews_batch)

# Function to add rate limiting and delay
def process_reviews_with_rate_limit(balanced_data, batch_size=10, delay=1):
    sentiments = []
    start_time = time.time()  # Start the timer

    print("Starting batch sentiment processing...\n")

    # Track initial memory usage
    initial_memory = psutil.virtual_memory().percent
    print(f"Initial memory usage: {initial_memory}%")

    for i in range(0, len(balanced_data), batch_size):
        # Get the batch of reviews
        reviews_batch = balanced_data['review'].iloc[i:i + batch_size].tolist()
        batch_sentiments = get_batch_sentiments(reviews_batch)
        sentiments.extend(batch_sentiments)
        
        # Print progress
        print(f"Processed batch {i // batch_size + 1}...")

        # Apply rate limiting by introducing a delay between API calls
        print(f"Sleeping for {delay} seconds to avoid rate limit...")
        time.sleep(delay + random.uniform(0, 2))  # Adding random jitter to avoid fixed delays
        
        # Check memory usage after each batch
        current_memory = psutil.virtual_memory().percent
        print(f"Current memory usage after batch {i // batch_size + 1}: {current_memory}%")

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
    balanced_data = pd.read_csv('balanced_sampled_imdb.csv')

    # Preprocess the data
    balanced_data = preprocess_data(balanced_data)

    # Process the reviews with rate limiting
    sentiments = process_reviews_with_rate_limit(balanced_data, batch_size=10, delay=2)

    # Add the sentiments to the dataframe
    balanced_data['gpt_sentiment'] = sentiments

    # Verify that all reviews were evaluated
    if len(sentiments) == len(balanced_data):
        print(f"All {len(balanced_data)} reviews have been evaluated successfully.")
    else:
        print(f"Warning: Some reviews were not evaluated. Expected {len(balanced_data)} reviews, but got {len(sentiments)} sentiments.")

    # Ensure there are no 'None' values (this means no errors during processing)
    if balanced_data['gpt_sentiment'].isnull().sum() > 0:
        print(f"Warning: There are {balanced_data['gpt_sentiment'].isnull().sum()} reviews with no sentiment assigned.")
    else:
        print("All reviews have valid sentiment values.")

    # Display the first few rows with the new column
    print("\nSample data with GPT sentiment:")
    print(balanced_data[['review', 'gpt_sentiment']].head())

    # Save the updated dataset with GPT sentiment
    balanced_data.to_csv('balanced_sampled_with_gpt_sentiment.csv', index=False)
    
    print("\nSentiment analysis complete! Saved to 'balanced_sampled_with_gpt_sentiment.csv'.")

if __name__ == "__main__":
=======
import pandas as pd
import openai
import time
import random
import psutil # For monitoring memory usage
from dataprep import *  # Import your preprocessing functions

# Set your OpenAI API key
openai.api_key = 'Here should be the key.'

# Function to process sentiments for a batch of reviews
def get_batch_sentiments(reviews_batch):
    # Prepare a single prompt with multiple reviews
    batch_reviews = "\n".join([f"{i + 1}. {review}" for i, review in enumerate(reviews_batch)])
    prompt = f"Classify the sentiment of the following movie reviews as 'positive' or 'negative':\n{batch_reviews}\nSentiments:"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5 Turbo model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that does sentiment analysis. You classify every review as 'positive' or 'negative' only. There's no neutral sentiment."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.0,  # Ensures consistent and reliable response
        )
        sentiments = response['choices'][0]['message']['content'].strip().split('\n')
        return [sentiment.strip().lower() for sentiment in sentiments] #pro každý sentiment v poli sentiments se provede strip a lower funkce
    except Exception as e:
        print(f"Error processing batch of reviews: {e}")
        return [None] * len(reviews_batch)

# Function to add rate limiting and delay
def process_reviews_with_rate_limit(balanced_data, batch_size=10, delay=1):
    sentiments = []
    start_time = time.time()  # Start the timer

    print("Starting batch sentiment processing...\n")

    # Track initial memory usage
    initial_memory = psutil.virtual_memory().percent
    print(f"Initial memory usage: {initial_memory}%")

    for i in range(0, len(balanced_data), batch_size):
        # Get the batch of reviews
        reviews_batch = balanced_data['review'].iloc[i:i + batch_size].tolist()
        batch_sentiments = get_batch_sentiments(reviews_batch)
        sentiments.extend(batch_sentiments)
        
        # Print progress
        print(f"Processed batch {i // batch_size + 1}...")

        # Apply rate limiting by introducing a delay between API calls
        print(f"Sleeping for {delay} seconds to avoid rate limit...")
        time.sleep(delay + random.uniform(0, 2))  # Adding random jitter to avoid fixed delays
        
        # Check memory usage after each batch
        current_memory = psutil.virtual_memory().percent
        print(f"Current memory usage after batch {i // batch_size + 1}: {current_memory}%")

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
    balanced_data = pd.read_csv('balanced_sampled_imdb.csv')

    # Preprocess the data
    balanced_data = preprocess_data(balanced_data)

    # Process the reviews with rate limiting
    sentiments = process_reviews_with_rate_limit(balanced_data, batch_size=10, delay=2)

    # Add the sentiments to the dataframe
    balanced_data['gpt_sentiment'] = sentiments

    # Verify that all reviews were evaluated
    if len(sentiments) == len(balanced_data):
        print(f"All {len(balanced_data)} reviews have been evaluated successfully.")
    else:
        print(f"Warning: Some reviews were not evaluated. Expected {len(balanced_data)} reviews, but got {len(sentiments)} sentiments.")

    # Ensure there are no 'None' values (this means no errors during processing)
    if balanced_data['gpt_sentiment'].isnull().sum() > 0:
        print(f"Warning: There are {balanced_data['gpt_sentiment'].isnull().sum()} reviews with no sentiment assigned.")
    else:
        print("All reviews have valid sentiment values.")

    # Display the first few rows with the new column
    print("\nSample data with GPT sentiment:")
    print(balanced_data[['review', 'gpt_sentiment']].head())

    # Save the updated dataset with GPT sentiment
    balanced_data.to_csv('balanced_sampled_with_gpt_sentiment.csv', index=False)
    
    print("\nSentiment analysis complete! Saved to 'balanced_sampled_with_gpt_sentiment.csv'.")

if __name__ == "__main__":
>>>>>>> 23ed1d090e8796675969bde3c102083c8670e7a6
=======
import pandas as pd
import openai
import time
import random
import psutil # For monitoring memory usage
from dataprep import *  # Import your preprocessing functions

# Set your OpenAI API key
openai.api_key = 'Here should be the key.'

# Function to process sentiments for a batch of reviews
def get_batch_sentiments(reviews_batch):
    # Prepare a single prompt with multiple reviews
    batch_reviews = "\n".join([f"{i + 1}. {review}" for i, review in enumerate(reviews_batch)])
    prompt = f"Classify the sentiment of the following movie reviews as 'positive' or 'negative':\n{batch_reviews}\nSentiments:"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5 Turbo model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that does sentiment analysis. You classify every review as 'positive' or 'negative' only. There's no neutral sentiment."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.0,  # Ensures consistent and reliable response
        )
        sentiments = response['choices'][0]['message']['content'].strip().split('\n')
        return [sentiment.strip().lower() for sentiment in sentiments] #pro každý sentiment v poli sentiments se provede strip a lower funkce
    except Exception as e:
        print(f"Error processing batch of reviews: {e}")
        return [None] * len(reviews_batch)

# Function to add rate limiting and delay
def process_reviews_with_rate_limit(balanced_data, batch_size=10, delay=1):
    sentiments = []
    start_time = time.time()  # Start the timer

    print("Starting batch sentiment processing...\n")

    # Track initial memory usage
    initial_memory = psutil.virtual_memory().percent
    print(f"Initial memory usage: {initial_memory}%")

    for i in range(0, len(balanced_data), batch_size):
        # Get the batch of reviews
        reviews_batch = balanced_data['review'].iloc[i:i + batch_size].tolist()
        batch_sentiments = get_batch_sentiments(reviews_batch)
        sentiments.extend(batch_sentiments)
        
        # Print progress
        print(f"Processed batch {i // batch_size + 1}...")

        # Apply rate limiting by introducing a delay between API calls
        print(f"Sleeping for {delay} seconds to avoid rate limit...")
        time.sleep(delay + random.uniform(0, 2))  # Adding random jitter to avoid fixed delays
        
        # Check memory usage after each batch
        current_memory = psutil.virtual_memory().percent
        print(f"Current memory usage after batch {i // batch_size + 1}: {current_memory}%")

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
    balanced_data = pd.read_csv('balanced_sampled_imdb.csv')

    # Preprocess the data
    balanced_data = preprocess_data(balanced_data)

    # Process the reviews with rate limiting
    sentiments = process_reviews_with_rate_limit(balanced_data, batch_size=10, delay=2)

    # Add the sentiments to the dataframe
    balanced_data['gpt_sentiment'] = sentiments

    # Verify that all reviews were evaluated
    if len(sentiments) == len(balanced_data):
        print(f"All {len(balanced_data)} reviews have been evaluated successfully.")
    else:
        print(f"Warning: Some reviews were not evaluated. Expected {len(balanced_data)} reviews, but got {len(sentiments)} sentiments.")

    # Ensure there are no 'None' values (this means no errors during processing)
    if balanced_data['gpt_sentiment'].isnull().sum() > 0:
        print(f"Warning: There are {balanced_data['gpt_sentiment'].isnull().sum()} reviews with no sentiment assigned.")
    else:
        print("All reviews have valid sentiment values.")

    # Display the first few rows with the new column
    print("\nSample data with GPT sentiment:")
    print(balanced_data[['review', 'gpt_sentiment']].head())

    # Save the updated dataset with GPT sentiment
    balanced_data.to_csv('balanced_sampled_with_gpt_sentiment.csv', index=False)
    
    print("\nSentiment analysis complete! Saved to 'balanced_sampled_with_gpt_sentiment.csv'.")

if __name__ == "__main__":
>>>>>>> 23ed1d090e8796675969bde3c102083c8670e7a6
    main()