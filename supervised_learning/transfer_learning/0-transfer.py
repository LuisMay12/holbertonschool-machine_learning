#!/usr/bin/env python3
"""
Trains a convolutional neural network using transfer learning to classify
CIFAR-10.
Saves the trained (compiled) model as cifar10.h5 in the current working
directory.

Also defines preprocess_data(X, Y) to preprocess CIFAR-10 data.
"""

from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Preprocess CIFAR-10 data for the model.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3), CIFAR-10 images
        Y: numpy.ndarray of shape (m,) or (m, 1), CIFAR-10 labels

    Returns:
        X_p: numpy.ndarray, preprocessed images (float32)
        Y_p: numpy.ndarray, one-hot labels of shape (m, 10)
    """
    X_p = X.astype("float32")

    # cifar10 labels often come as (m, 1); flatten to (m,)
    if len(Y.shape) == 2 and Y.shape[1] == 1:
        Y = Y.reshape(-1)

    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def _build_feature_extractor(input_shape=(32, 32, 3), target_size=(96, 96)):
    """
    Builds a frozen feature extractor model:
    input (32x32) -> resize (Lambda) -> preprocess
    -> EfficientNetB0(include_top=False) -> GAP

    Returns:
        feature_model: Keras Model that outputs feature vectors
        base: the EfficientNet backbone (frozen)
    """
    inputs = K.Input(shape=input_shape)

    # Hint requires Lambda first layer to resize up to application size
    resize_layer = K.layers.Resizing(target_size[0], target_size[1], name="resizer")

    x = K.layers.Lambda(
        lambda img: resize_layer(img),
        output_shape=(target_size[0], target_size[1], 3),
        name="resize_to_app_size"
    )(inputs)

    # EfficientNet expects specific preprocessing
    x = K.applications.efficientnet.preprocess_input(x)

    base = K.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(target_size[0], target_size[1], 3)
    )
    base.trainable = False

    x = base(x, training=False)
    x = K.layers.GlobalAveragePooling2D(name="gap")(x)

    feature_model = K.Model(inputs=inputs, outputs=x, name="feature_extractor")
    return feature_model, base


def _build_classifier_head(feature_dim):
    """
    Builds a small classifier head to train on bottleneck features.

    Returns:
        head_model: Keras Model mapping features -> CIFAR-10 probabilities
    """
    feat_in = K.Input(shape=(feature_dim,), name="features_input")
    x = K.layers.BatchNormalization()(feat_in)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Dense(512, activation="relu")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.4)(x)
    outputs = K.layers.Dense(10, activation="softmax", name="predictions")(x)

    name = "classifier_head"
    head_model = K.Model(inputs=feat_in, outputs=outputs, name=name)
    return head_model


def _assemble_full_model(feature_model, head_model):
    """
    Creates the final end-to-end model:
    raw images -> feature_model -> head_model
    """
    inputs = feature_model.input
    features = feature_model(inputs)
    outputs = head_model(features)
    model = K.Model(inputs=inputs, outputs=outputs, name="cifar10_transfer")
    return model


def main():
    """
    Trains the transfer learning model and saves it as cifar10.h5.
    """
    # Load CIFAR-10
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Train/val split (manual, reproducible)
    # 50k total -> 45k train, 5k val
    X_tr, Y_tr = X_train[:45000], Y_train[:45000]
    X_val, Y_val = X_train[45000:], Y_train[45000:]

    # Build frozen feature extractor
    target_size = (96, 96)
    feature_model, _ = _build_feature_extractor(target_size=target_size)

    # Precompute bottleneck features ONCE (hint)
    # This is the biggest time saver.
    batch_size = 256
    F_tr = feature_model.predict(X_tr, batch_size=batch_size, verbose=1)
    F_val = feature_model.predict(X_val, batch_size=batch_size, verbose=1)

    # Build and train classifier head on features
    head_model = _build_classifier_head(F_tr.shape[1])

    # Optimizer & loss
    # AdamW exists in TF 2.15; good regularization for transfer learning heads
    optimizer = K.optimizers.AdamW(learning_rate=2e-3, weight_decay=1e-4)

    head_model.compile(
        optimizer=optimizer,
        loss=K.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )

    callbacks = [
        K.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )
    ]

    head_model.fit(
        F_tr, Y_tr,
        validation_data=(F_val, Y_val),
        epochs=40,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )

    # Assemble full compiled model
    # (raw images -> resize/preprocess -> backbone -> head)
    full_model = _assemble_full_model(feature_model, head_model)

    # Compile full model so the saved .h5 is compiled
    full_model.compile(
        optimizer=K.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
        loss=K.losses.CategoricalCrossentropy(label_smoothing=0.0),
        metrics=["accuracy"]
    )

    # Quick sanity evaluation on test
    full_model.evaluate(X_test, Y_test, batch_size=128, verbose=1)

    # Save compiled model
    full_model.save("cifar10.h5")


if __name__ == "__main__":
    main()
