import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

NUM_CLASSES=10

def preprocess(feature, label=None):
    feature = tf.expand_dims(feature, axis=-1)
    feature = tf.divide(tf.cast(feature, tf.float32), 255.0)
    if label is not None:
        label = tf.one_hot(label, NUM_CLASSES)
    return feature, label

def make_input_fn(features, labels, mode, hparams):
    def _input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).map(map_func = preprocess)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuffle(buffer_size = hparams["buffer_size"] * hparams["batch_size"]).repeat(None)
        else:
            num_epochs = 1

        dataset = dataset.batch(batch_size = hparams["batch_size"])
        return dataset

    return _input_fn

def cnn_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape=(28, 28, 1), name="image"))
    model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax', name='output'))
    return model

def train_and_evaluate(output_dir, hparams):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    model = cnn_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(hparams["learning_rate"]),
		  loss=tf.keras.losses.CategoricalCrossentropy(),
		  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    estimator = tf.keras.estimator.model_to_estimator(
      keras_model = model, model_dir = output_dir)
    train_spec = tf.estimator.TrainSpec(input_fn=make_input_fn(X_train, y_train, tf.estimator.ModeKeys.TRAIN, hparams),
					max_steps=hparams["train_steps"])

    input_column = tf.feature_column.numeric_column("image_input", shape=(28,28,1), dtype=tf.float32)
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(tf.feature_column.make_parse_example_spec([input_column]))
    exporter = tf.estimator.LatestExporter(
                 name = "exporter",
                 serving_input_receiver_fn = serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=make_input_fn(X_test, y_test, tf.estimator.ModeKeys.EVAL, hparams),
                                      exporters=exporter,
				      steps=None)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    estimator.saved_model()
