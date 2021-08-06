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

_CONCRETE_INPUT = "numpy_inputs"
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
    """Creates a DenseNet121-based model for classifying flowers data.

    Returns:
    A Keras Model.
    """
    inputs = keras.Input(shape=_INPUT_SHAPE)
    base_model = keras.applications.DenseNet121(
        include_top=False, input_shape=_INPUT_SHAPE, pooling="avg"
    )
    base_model.trainable = False
    x = keras.applications.densenet.preprocess_input(inputs)
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


def _preprocess(bytes_input):
    decoded = tf.io.decode_jpeg(bytes_input, channels=3)
    resized = tf.image.resize(decoded, size=(224, 224))
    return resized


@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def preprocess_fn(bytes_inputs):
    decoded_images = tf.map_fn(
        _preprocess, bytes_inputs, dtype=tf.float32, back_prop=False
    )
    return {_CONCRETE_INPUT: decoded_images}


def _model_exporter(model: tf.keras.Model):
    m_call = tf.function(model.call).get_concrete_function(
        [
            tf.TensorSpec(
                shape=[None, 224, 224, 3], dtype=tf.float32, name=_CONCRETE_INPUT
            )
        ]
    )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(bytes_inputs):
        # This function comes from the Computer Vision book from O'Reilly.
        labels = tf.constant(
            ["daisy", "dandelion", "roses", "sunflowers", "tulips"], dtype=tf.string
        )
        images = preprocess_fn(bytes_inputs)

        probs = m_call(**images)
        indices = tf.argmax(probs, axis=1)
        pred_source = tf.gather(params=labels, indices=indices)
        pred_confidence = tf.reduce_max(probs, axis=1)
        return {"label": pred_source, "confidence": pred_confidence}

    return serving_fn


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
    # The result of the training should be saved in `fn_args.serving_model_dir`
    # directory.
    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures={"serving_default": _model_exporter(model)},
    )
