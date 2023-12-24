import torch
from torch.utils.data import Dataset
from collections import Counter
import string

class ShakespeareDataset(Dataset):
    def __init__(self, data_file, sequence_length=30):
        self.sequence_length = sequence_length
        self.vocab, self.char_to_index, self.index_to_char = self.build_vocab(data_file)
        self.data = self.load_and_tokenize(data_file)

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        sequence = self.data[index:index + self.sequence_length]
        target = self.data[index + self.sequence_length]

        sequence_indices = [self.char_to_index[char] for char in sequence]
        target_index = self.char_to_index[target]

        # Convert target to one-hot encoding
        target_one_hot = torch.zeros(len(self.vocab))
        target_one_hot[target_index] = 1

        return torch.tensor(sequence_indices), target_one_hot.long()  # Convert to torch.long

    def build_vocab(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            text = f.read().lower()  # Read and convert to lowercase for simplicity

        # Tokenize by characters
        tokens = list(text)

        # Create vocabulary
        vocab = set(tokens)
        vocab_size = len(vocab)

        # Create dictionaries for mapping characters to indices and vice versa
        char_to_index = {char: i for i, char in enumerate(vocab)}
        index_to_char = {i: char for i, char in enumerate(vocab)}

        return vocab, char_to_index, index_to_char

    def load_and_tokenize(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            text = f.read().lower()  # Read and convert to lowercase for simplicity

        # Tokenize by characters
        tokens = list(text)

        # Filter out non-alphanumeric characters and punctuation
        tokens = [char for char in tokens if char.isalnum() or char in string.whitespace]

        return tokens


