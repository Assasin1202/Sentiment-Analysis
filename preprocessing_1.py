# %%
import os
import pandas as pd
import spacy
import torch
from tqdm import tqdm
import re

# %%
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# Initialize spaCy model with disabling unnecessary components for efficiency
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

# Add custom stop words if needed
# nlp.Defaults.stop_words |= {"additional", "stopword"}

# %%
def load_imdb_data(pos_dir, neg_dir):
    """
    Load IMDB reviews from positive and negative directories.

    Args:
        pos_dir (str): Path to positive reviews directory.
        neg_dir (str): Path to negative reviews directory.

    Returns:
        pd.DataFrame: DataFrame containing reviews and labels.
    """
    data = []
    labels = []

    # Load positive reviews
    pos_files = [file for file in os.listdir(pos_dir) if file.endswith(".txt")]
    for filename in tqdm(pos_files, desc="Loading positive reviews"):
        file_path = os.path.join(pos_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            review = file.read().strip()
            data.append(review)
            labels.append(1)  # Positive label

    # Load negative reviews
    neg_files = [file for file in os.listdir(neg_dir) if file.endswith(".txt")]
    for filename in tqdm(neg_files, desc="Loading negative reviews"):
        file_path = os.path.join(neg_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            review = file.read().strip()
            data.append(review)
            labels.append(0)  # Negative label

    df = pd.DataFrame({'review': data, 'label': labels})
    print(f"Total reviews loaded: {len(df)}")
    return df

# %%
def preprocess_text(text):
    """
    Preprocess the input text by cleaning and removing unnecessary components.

    Args:
        text (str): Raw review text.

    Returns:
        str: Cleaned text.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text

def tokenize_text(text):
    """
    Tokenize and lemmatize the preprocessed text using spaCy.

    Args:
        text (str): Cleaned text.

    Returns:
        list: List of tokens.
    """
    doc = nlp(text)
    tokens = [
        token.text for token in doc
        if not token.is_stop and not token.is_punct and token.text.isalpha()  # Check for alphabetic characters only
    ]
    return tokens

def preprocess_dataframe(df):
    """
    Apply preprocessing and tokenization to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with raw reviews.

    Returns:
        pd.DataFrame: DataFrame with tokenized reviews.
    """
    # Preprocess text
    tqdm.pandas(desc="Preprocessing text")
    df['cleaned_review'] = df['review'].progress_apply(preprocess_text)

    # Tokenize text
    tqdm.pandas(desc="Tokenizing text")
    df['tokenized_review'] = df['cleaned_review'].progress_apply(tokenize_text)

    return df

# %%
def save_dataframe(df, filepath):
    """
    Save the DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        filepath (str): Path to the output CSV file.
    """
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

# %%
# Define directories
train_pos_dir = "C:/Users/Pranav/Desktop/Sem_7/DL/DL_lab/Lab 3/aclImdb_v1/aclImdb/train/pos"
train_neg_dir = "C:/Users/Pranav/Desktop/Sem_7/DL/DL_lab/Lab 3/aclImdb_v1/aclImdb/train/neg"
test_pos_dir = "C:/Users/Pranav/Desktop/Sem_7/DL/DL_lab/Lab 3/aclImdb_v1/aclImdb/test/pos"
test_neg_dir = "C:/Users/Pranav/Desktop/Sem_7/DL/DL_lab/Lab 3/aclImdb_v1/aclImdb/test/neg"

# %%
# Load datasets
print("Loading training data...")
train_data = load_imdb_data(train_pos_dir, train_neg_dir)

print("Loading testing data...")
test_data = load_imdb_data(test_pos_dir, test_neg_dir)

# %%
# Preprocess datasets
print("Preprocessing training data...")
train_data = preprocess_dataframe(train_data)

print("Preprocessing testing data...")
test_data = preprocess_dataframe(test_data)

# %%
# Select relevant columns
train_data_final = train_data[['tokenized_review', 'label']]
test_data_final = test_data[['tokenized_review', 'label']]

# %%
# Save preprocessed data to CSV
save_dataframe(train_data_final, "train_data_preprocessed_new.csv")
save_dataframe(test_data_final, "test_data_preprocessed_new.csv")

# %%
# Optional: Display sample data
print("Sample preprocessed training data:")
print(train_data_final.head())

print("Sample preprocessed testing data:")
print(test_data_final.head())


# # %%
# print(train_data_final['tokenized_review'][0])

# # %%
# import pandas as pd
# from collections import Counter
# import numpy as np

# # Load preprocessed data
# train_data = pd.read_csv("train_data_preprocessed_new.csv")
# test_data = pd.read_csv("test_data_preprocessed_new.csv")

# # %%
# # Build vocabulary from training data
# def build_vocab(tokenized_reviews, min_freq=2):
#     counter = Counter()
#     for tokens in tokenized_reviews:
#         counter.update(tokens)
#     # Include words with frequency >= min_freq
#     vocab = {word for word, freq in counter.items() if freq >= min_freq}
#     # Add special tokens
#     vocab = sorted(vocab)
#     vocab = ['PAD', 'UNK'] + vocab
#     word_to_idx = {word: idx for idx, word in enumerate(vocab)}
#     idx_to_word = {idx: word for word, idx in word_to_idx.items()}
#     print(vocab)
#     print(f"Vocabulary size: {len(vocab)}")
#     return word_to_idx, idx_to_word

# word_to_idx, idx_to_word = build_vocab(train_data['tokenized_review'],1)

# # %%
# def tokens_to_indices(tokenized_reviews, word_to_idx):
#     indices = []
#     for tokens in tokenized_reviews:
#         idx = [word_to_idx.get(token, word_to_idx['UNK']) for token in tokens]
#         indices.append(idx)
#     return indices

# train_indices = tokens_to_indices(train_data['tokenized_review'], word_to_idx)
# test_indices = tokens_to_indices(test_data['tokenized_review'], word_to_idx)

# # %%
# def calculate_average_length(indices):
#     total_words = sum(len(seq) for seq in indices)
#     total_sentences = len(indices)
#     average_length = total_words / total_sentences
#     return int(round(average_length))

# average_length = calculate_average_length(train_indices)
# print(f"Average sentence length: {average_length} tokens")

# # %%
# max_length = average_length  # e.g., 60
# pad_idx = word_to_idx['PAD']

# def pad_truncate(indices, max_length, pad_idx):
#     padded = []
#     for seq in indices:
#         if len(seq) > max_length:
#             padded_seq = seq[:max_length]
#         else:
#             padded_seq = seq + [pad_idx] * (max_length - len(seq))
#         padded.append(padded_seq)
#     return padded

# train_padded = pad_truncate(train_indices, max_length, pad_idx)
# test_padded = pad_truncate(test_indices, max_length, pad_idx)


# # %%
# train_data['padded_review'] = train_padded
# test_data['padded_review'] = test_padded

# # Save the updated DataFrame if needed
# train_data.to_csv("train_data_final.csv", index=False)
# test_data.to_csv("test_data_final.csv", index=False)


# # %%
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import ast  # For parsing string representations of lists

# # %%
# # Load preprocessed data
# train_df = pd.read_csv("train_data_final.csv")
# test_df = pd.read_csv("test_data_final.csv")

# # %%
# # Inspect the first few rows
# print("Training Data:")
# print(train_df.head())

# print("\nTest Data:")
# print(test_df.head())
