#!/usr/bin/env python3
"""Train and validate an LSTM model for BTC price forecasting."""

import numpy as np
import tensorflow as tf
import tensorflow.keras as K


DATA_FILE = "btc_preprocessed.npz"
MODEL_FILE = "btc_model.keras"
BATCH_SIZE = 64
EPOCHS = 50
SHUFFLE_BUFFER = 10000
LEARNING_RATE = 0.001
SEED = 0


def load_data(path):
    """Load preprocessed BTC training, validation, and test data."""
    data = np.load(path)

    x_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    x_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    x_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test, data


def make_dataset(x_values, y_values, batch_size, shuffle=False):
    """Create a tf.data.Dataset from input and target arrays."""
    dataset = tf.data.Dataset.from_tensor_slices((x_values, y_values))

    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER, seed=SEED)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_model(input_shape):
    """Build an LSTM model for next-hour BTC close prediction."""
    model = K.models.Sequential([
        K.layers.Input(shape=input_shape),
        K.layers.LSTM(64, return_sequences=True),
        K.layers.Dropout(0.2),  # i am using this to prevent overfitting
        K.layers.LSTM(32),
        K.layers.Dropout(0.2),
        K.layers.Dense(32, activation="relu"),
        K.layers.Dense(1)
    ])

    optimizer = K.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model


def get_callbacks():
    """Create training callbacks to improve validation performance."""
    callbacks = [
        K.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        K.callbacks.ModelCheckpoint(
            MODEL_FILE,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    return callbacks


def denormalize_close(values, data):
    """Convert normalized close values back to USD."""
    feature_columns = list(data["feature_columns"])
    close_index = feature_columns.index("Close")

    close_mean = data["feature_mean"][close_index]
    close_std = data["feature_std"][close_index]

    return values * close_std + close_mean


def show_sample_prediction(model, x_test, y_test, data):
    """Display one denormalized prediction example."""
    prediction = model.predict(x_test[:1], verbose=0).reshape(-1)
    expected = y_test[:1]

    prediction_usd = denormalize_close(prediction, data)
    expected_usd = denormalize_close(expected, data)

    print("Sample prediction:")
    print("Predicted close: ${:.2f}".format(prediction_usd[0]))
    print("Expected close:  ${:.2f}".format(expected_usd[0]))


def main():
    """Train, validate, evaluate, and save the BTC forecasting model."""
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    x_train, y_train, x_val, y_val, x_test, y_test, data = load_data(
        DATA_FILE
    )

    train_ds = make_dataset(
        x_train,
        y_train,
        BATCH_SIZE,
        shuffle=True
    )
    val_ds = make_dataset(
        x_val,
        y_val,
        BATCH_SIZE
    )
    test_ds = make_dataset(
        x_test,
        y_test,
        BATCH_SIZE
    )

    model = build_model(x_train.shape[1:])
    model.summary()

    callbacks = get_callbacks()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    loss, mae = model.evaluate(test_ds, verbose=0)
    print("Test MSE: {:.6f}".format(loss))
    print("Test MAE: {:.6f}".format(mae))

    model.save(MODEL_FILE)
    print("Saved model to {}".format(MODEL_FILE))

    if x_test.shape[0] > 0:
        show_sample_prediction(model, x_test, y_test, data)


if __name__ == "__main__":
    main()
