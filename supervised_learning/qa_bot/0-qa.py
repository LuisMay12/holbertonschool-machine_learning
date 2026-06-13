#!/usr/bin/env python3
"""Question answering with a pre-trained BERT model."""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Find a text snippet in reference that answers question.

    Args:
        question (str): question to answer
        reference (str): reference document containing the answer

    Returns:
        str: answer found in the reference, or None if no answer is found
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + question_tokens + ['[SEP]']
    input_type_ids = [0] * len(tokens)

    tokens += reference_tokens + ['[SEP]']
    input_type_ids += [1] * (len(reference_tokens) + 1)

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)

    inputs = [
        tf.expand_dims(tf.constant(input_word_ids), 0),
        tf.expand_dims(tf.constant(input_mask), 0),
        tf.expand_dims(tf.constant(input_type_ids), 0)
    ]

    outputs = model(inputs)
    start = int(tf.argmax(outputs[0][0][1:]) + 1)
    end = int(tf.argmax(outputs[1][0][1:]) + 1)

    if start == 0 or end == 0 or start > end:
        return None

    answer_tokens = tokens[start:end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if answer == '' or answer in ('[CLS]', '[SEP]'):
        return None

    return answer
