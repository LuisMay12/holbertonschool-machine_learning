#!/usr/bin/env python3
"""Train a transformer model for Portuguese to English translation."""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule from the original Transformer paper."""

    def __init__(self, dm, warmup_steps=4000):
        """Initialize the schedule."""
        super(CustomSchedule, self).__init__()

        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Calculate the learning rate for a training step."""
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """Calculate sparse categorical crossentropy ignoring padding."""
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """Calculate sparse categorical accuracy ignoring padding."""
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Create and train a transformer for Portuguese to English translation."""
    dataset = Dataset(batch_size, max_len)
    input_vocab = dataset.tokenizer_pt.vocab_size + 2
    target_vocab = dataset.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_len,
        max_len
    )

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()

        for batch, (inputs, target) in enumerate(dataset.data_train):
            target_input = target[:, :-1]
            target_real = target[:, 1:]
            encoder_mask, combined_mask, decoder_mask = create_masks(
                inputs,
                target_input
            )

            with tf.GradientTape() as tape:
                predictions = transformer(
                    inputs,
                    target_input,
                    True,
                    encoder_mask,
                    combined_mask,
                    decoder_mask
                )
                loss = loss_function(target_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables)
            )

            train_loss(loss)
            train_accuracy(accuracy_function(target_real, predictions))

            if batch % 50 == 0:
                print(
                    'Epoch {}, Batch {}: Loss {}, Accuracy {}'.format(
                        epoch + 1,
                        batch,
                        train_loss.result(),
                        train_accuracy.result()
                    )
                )

        print(
            'Epoch {}: Loss {}, Accuracy {}'.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result()
            )
        )

    return transformer
