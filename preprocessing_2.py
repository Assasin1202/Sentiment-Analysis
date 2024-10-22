

# %%
# Load preprocessed data
import pandas as pd
from collections import Counter
import ast  # To safely evaluate strings containing Python literals


train_data = pd.read_csv("train_preprocessed_without_lemma.csv")
test_data = pd.read_csv("test_data_preprocessed_new.csv")


# Define minimum frequency
min_freq = 5

def build_vocab(tokenized_reviews, min_freq=5):
    """
    Build a vocabulary dictionary mapping words to indices based on tokenized reviews.

    Args:
        tokenized_reviews (pd.Series): Series containing strings of list-formatted tokens.
        min_freq (int): Minimum frequency threshold for including a word in the vocabulary.

    Returns:
        word_to_idx (dict): Mapping from word to index.
        idx_to_word (dict): Mapping from index to word.
    """
    counter = Counter()

    # Convert string representations of lists to actual lists
    for review in tokenized_reviews:
        tokens = ast.literal_eval(review)
        counter.update(tokens)
    
    # Include words with frequency >= min_freq
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    
    # Sort the vocabulary
    vocab = sorted(vocab)
    
    # Add special tokens
    vocab = ['PAD', 'UNK'] + vocab

    # Create mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    print(vocab[:20])
    print(f"Vocabulary size: {len(vocab)}")
    return word_to_idx, idx_to_word

# Build the vocabulary using training data
word_to_idx, idx_to_word = build_vocab(train_data['tokenized_review'], min_freq=min_freq)

# %%
vocab_size = len(word_to_idx)
print(f"Vocabulary size: {vocab_size}")

# %%
# print(train_multihot[0].shape)
# # Print the indices of the non-zero elements
# print(np.nonzero(train_multihot[0])[0])
def tokens_to_indices(tokenized_reviews, word_to_idx):
    indices = []
    for tokens in tokenized_reviews:
        idx = [word_to_idx.get(token, word_to_idx['UNK']) for token in tokens]
        indices.append(idx)
    return indices

train_indices = tokens_to_indices(train_data['tokenized_review'], word_to_idx)
test_indices = tokens_to_indices(test_data['tokenized_review'], word_to_idx)

print(f"Train indices shape: {len(train_indices)}")
print(f"Test indices shape: {len(test_indices)}")
# %% 
train_indices[0]
# %%
def calculate_average_length(indices):
    total_words = sum(len(seq) for seq in indices)
    total_sentences = len(indices)
    average_length = total_words / total_sentences
    return int(round(average_length))

average_length = calculate_average_length(train_indices)
print(f"Average sentence length: {average_length} tokens")

# %%
max_length = average_length  # e.g., 60
pad_idx = word_to_idx['PAD']

def pad_truncate(indices, max_length, pad_idx):
    padded = []
    for seq in indices:
        if len(seq) > max_length:
            padded_seq = seq[:max_length]
        else:
            padded_seq = seq + [pad_idx] * (max_length - len(seq))
        padded.append(padded_seq)
    return padded

train_padded = pad_truncate(train_indices, max_length, pad_idx)
test_padded = pad_truncate(test_indices, max_length, pad_idx)


# %%
train_data['padded_review'] = train_padded
test_data['padded_review'] = test_padded

# Save the updated DataFrame if needed
train_data.to_csv("train_data_final.csv", index=False)
test_data.to_csv("test_data_final.csv", index=False)

