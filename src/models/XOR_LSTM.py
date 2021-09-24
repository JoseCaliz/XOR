import numpy as np
import tensorflow as tf
import logging
import click
import os
import csv
from time import time
import wandb
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

@click.command()
@click.option('--use_variable_length/--no_variable_length', default=False)
@click.option('--use_wandb/--no_wandb', default=False)
@click.option('--tensorboard/--no_tensorboard', default=False)
def main(use_variable_length, use_wandb, tensorboard):

    start_time = time()
    logger = logging.getLogger(__name__)
    logger.info('Initializing Model')

    X = list()
    y = list()

    data_dir = os.path.join('data', 'processed')
    file = os.path.join(data_dir, 'binary_sequences_fixed_length.csv')

    if use_variable_length:
        file = os.path.join(data_dir, 'binary_sequences_variable_length.csv')

    logger.info(f'Reading Dataset: {file}')

    with open(file, 'r') as f:
        dataset_reader = csv.reader(f, delimiter=',')
        for row in dataset_reader:
            x_temp, y_temp = row
            X.append(list(x_temp))
            y.append(y_temp)

    logger.info(f'Splitting Dataset: {file}')

    X_train,  X_test, y_train, y_test = train_test_split(
        np.array(X)[:, :, np.newaxis].astype(np.float32),
        np.array(y).astype(np.int32),
        random_state=0,
        shuffle=False
    )

    # %% hypeparameters
    early_stopping_delta = 1e-3
    early_stopping_patience = 20
    lstm_units = 2
    batch_size = 500
    validation_split = 0.2
    epochs = 100
    callbacks = list()
    model_name = 'LSTM'

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(
            units=lstm_units, kernel_initializer='random_normal',
        ),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy', 'mse'],
        loss = 'binary_crossentropy'
    )

    logger.info(f'Begginig Training')

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=early_stopping_delta,
        patience=early_stopping_patience,
        restore_best_weights=True
    )

    if(use_wandb):
        current_run = wandb.init(project='XOR', entity='jcaliz')

        wandb.config.early_stopping_delta = early_stopping_delta
        wandb.config.early_stopping_patience = early_stopping_patience
        wandb.config.lstm_units = lstm_units
        wandb.config.batch_size = batch_size
        wandb.config.validation_split = validation_split
        wandb.config.epochs = epochs
        wandb.config.callbacks = callbacks
        wandb.config.use_variable_length = use_variable_length

        callbacks.append(wandb.keras.WandbCallback())
        model_name = current_run.name

    if(tensorboard):
        log_dir = "src/models/logs/" + model_name
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        callbacks.append(tensorboard_callback)

    callbacks.append(es)
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
        epochs=epochs
    )

    if(use_wandb):
        model.save('./models/'+current_run.name)

    end_time = time()
    execution_time = str(timedelta(seconds=end_time - start_time))
    logger.info(f'Ending Execution. Time taken: {execution_time}')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

