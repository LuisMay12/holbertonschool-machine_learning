#!/usr/bin/env python3
"""Dataset module for machine translation transformer applications."""

import transformers
import tensorflow_datasets as tfds


class Dataset:
    """Loads and prepares a Portuguese to English translation dataset."""

    def __init__(self):
        """Initialize the training/validation data and tokenizers."""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )
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
            """Generate sentences from the dataset for tokenizer training."""
            for pt, en in data:
                if index == 0:
                    yield pt.numpy().decode('utf-8')
                else:
                    yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            get_training_corpus(0),
            vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            get_training_corpus(1),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
