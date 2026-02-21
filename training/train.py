import os
import tensorflow as tf
from config.config import *
from data.dataset_loader import load_data
from preprocessing.vectorizer import build_vectorizer
from preprocessing.label_encoder import build_label_lookup
from models.triage_model import build_model
from utils.metrics import evaluate_model


def prepare_dataset(texts, labels, vectorizer, label_lookup):
    """
    Creates tf.data pipeline with vectorization and label encoding
    """
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    ds = ds.shuffle(len(texts))

    ds = ds.map(
        lambda x, y: (
            vectorizer(x),
            label_lookup(y)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def train_model(data_path):
    """
    Full training pipeline
    """

    print("Loading dataset...")
    texts, labels = load_data(data_path)

    print("Building vectorizer...")
    vectorizer = build_vectorizer(texts)

    print("Building label encoder...")
    label_lookup = build_label_lookup(labels)

    print("Preparing dataset...")
    dataset = prepare_dataset(texts, labels, vectorizer, label_lookup)

    # Split dataset
    total_batches = tf.data.experimental.cardinality(dataset).numpy()
    train_batches = int(0.8 * total_batches)

    train_ds = dataset.take(train_batches)
    val_ds = dataset.skip(train_batches)

    print("Building model...")
    model = build_model(num_classes=len(label_lookup.get_vocabulary()))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Create folders
    os.makedirs("saved_model", exist_ok=True)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="saved_model/best_model",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            save_format="tf" 
        )
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\nFinal Evaluation on Validation Set:")
    evaluate_model(model, val_ds, label_lookup)

    print("\nSaving final model...")
    model.save("saved_model/final_model", save_format="tf")

    print("Training complete.")

    return model, vectorizer, label_lookup