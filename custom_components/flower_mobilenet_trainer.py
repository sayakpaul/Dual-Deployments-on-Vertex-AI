from typing import List
from absl import logging
from tensorflow import keras
from tfx import v1 as tfx
import tensorflow as tf


_IMAGE_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "class": tf.io.FixedLenFeature([], tf.int64),
    "one_hot_class": tf.io.VarLenFeature(tf.float32),
}

_INPUT_SHAPE = (224, 224, 3)
_TRAIN_BATCH_SIZE = 64
_EVAL_BATCH_SIZE = 64
_EPOCHS = 2


def _parse_fn(example):
    example = tf.io.parse_single_example(example, _IMAGE_FEATURES)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    class_label = tf.cast(example["class"], tf.int32)
    return image, class_label


def _input_fn(file_pattern: List[str], batch_size: int) -> tf.data.Dataset:
    """Generates features and label for training.

    Args:
        file_pattern: List of paths or patterns of input tfrecord files.
        batch_size: representing the number of consecutive elements of returned
            dataset to combine in a single batch.

    Returns:
        A dataset that contains (features, indices) tuple where features is a
            dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    logging.info(f"Reading data from: {file_pattern}")
    tfrecord_filenames = tf.io.gfile.glob(file_pattern[0] + ".gz")
    dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
    dataset = dataset.map(_parse_fn).batch(batch_size)
    return dataset.repeat()


def _make_keras_model() -> tf.keras.Model:
    """Creates a MobileNetV3-based model for classifying flowers data.

    Returns:
    A Keras Model.
    """
    inputs = keras.Input(shape=_INPUT_SHAPE)
    base_model = keras.applications.MobileNetV3Small(
        include_top=False, input_shape=_INPUT_SHAPE, pooling="avg"
    )
    base_model.trainable = False
    x = keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(
        x, training=False
    )  # Ensures BatchNorm runs in inference model in this model
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    model.summary(print_fn=logging.info)
    return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.

    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    """
    train_dataset = _input_fn(fn_args.train_files, batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, batch_size=_EVAL_BATCH_SIZE)

    model = _make_keras_model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=_EPOCHS,
    )
    _, acc = model.evaluate(eval_dataset, steps=fn_args.eval_steps)
    logging.info(f"Validation accuracy: {round(acc * 100, 2)}%")

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model.
    # The result of the training should be saved in `fn_args.serving_model_dir` directory.
    with tf.io.gfile.GFile(fn_args.serving_model_dir + "/model.tflite", "wb") as f:
        f.write(tflite_model)
