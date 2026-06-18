#!/usr/bin/env python3
"""Dataset module for machine translation transformer applications."""

import transformers
from setup import load_pt2en


class Dataset:
    """Loads and prepares a Portuguese to English translation dataset."""

    def __init__(self):
        """Initialize the training/validation data and tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create sub-word tokenizers trained from the given dataset."""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def get_training_corpus(index):
            """Generate batches of sentences for tokenizer training."""
            for pt, en in data.batch(1000):
                if index == 0:
                    yield [sentence.decode('utf-8') for sentence in pt.numpy()]
                else:
                    yield [sentence.decode('utf-8') for sentence in en.numpy()]

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            get_training_corpus(0),
            vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            get_training_corpus(1),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
