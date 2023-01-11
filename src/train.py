import tensorflow as tf
import numpy as np
from tqdm import tqdm

import math

from test import test_step
from autoencoder_model import Encoder, Decoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

def train(train_ds, validation_ds, features_dim, args):
    """
    Training the framework. For now, only training the anomaly detection estimator (AE).

    Parameters
    ----------
    train_ds: TF's Dataset. The training set
    validation_ds: TF's Dataset. The validation set
    features_dim: int. The dimensionality of the dataset
    args: argprase args. The args to the programa
    """
    
    train_X = []
    for x_batch_train, y_batch_train in train_ds:
        train_X.append(x_batch_train)
    train_X = np.concatenate(train_X, axis=0)
    # train_X = train_X[np.random.choice(np.arange(len(train_X)), size=10000)]

    if_estimator = if_training(train_X)
    # lof_estiamtor = lof_training(train_X)
    # ocs_estimator = ocs_training(train_X)
    AE_estimator = train_estimator(train_ds, validation_ds, features_dim, args)

    return if_estimator, AE_estimator

def train_estimator(train_ds, validation_ds, features_dim, args):
    """
    Training an Autoencoder as a anomaly detection estimator

    Parameters
    ----------
    train_ds: TF's Dataset. The training set
    validation_ds: TF's Dataset. The validation set
    features_dim: int. The dimensionality of the dataset
    args: argparse args. The args to the program
    """

    print("--- Training Autoencoder ---")
    # setting up training configurations
    optimizer = tf.keras.optimizers.Adam()
    loss_func = tf.keras.losses.MeanSquaredError()
    encoder = Encoder(input_shape=features_dim)
    decoder = Decoder(original_dim=features_dim)

    # early stopping variables
    patience = args.early_stopping_patience
    wait = 0
    best = math.inf

    # training loop
    tqdm_total_bar = args.num_epochs
    for epoch in tqdm(range(args.num_epochs), total=tqdm_total_bar):
        epoch_loss_mean = tf.keras.metrics.Mean()

        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            loss = train_step(x_batch_train, encoder, decoder, loss_func, optimizer)
            # keep track of the metrics
            epoch_loss_mean.update_state(loss)
            if step % int(0.1 * len(train_ds)) == 0 :
                print(f"{str(step+1)+'/'+str(len(train_ds)+1):8s} batch train loss: {epoch_loss_mean.result():3.10f}")

        # validation evaluation
        validation_loss_mean = tf.keras.metrics.Mean()
        for step, (x_batch_validation, y_batch_validation) in enumerate(validation_ds):
            last_validation_loss = test_step(x_batch_validation, encoder, decoder, loss_func).numpy()
            validation_loss_mean.update_state(last_validation_loss)
        
        train_mean_loss = epoch_loss_mean.result()
        validation_mean_loss = validation_loss_mean.result()
        # update metrics after each epoch
        print(f"Epoch {epoch+1:4d}. Train Average Loss: {train_mean_loss:3.10f}. Validation Average Loss: {validation_mean_loss:3.10f}")
        
        # early stopping logic
        wait+=1
        if validation_mean_loss < best:
            best = validation_mean_loss
            wait = 0
        if wait >= patience:
            print("**EARLY STOPPING**")
            break

    return encoder, decoder

@tf.function
def train_step(inputs, encoder, decoder, loss_func, optimizer):
    with tf.GradientTape() as tape:
        latent_var = encoder(inputs)
        outputs = decoder(latent_var)
        loss = loss_func(inputs, outputs)

        trainable_vars = encoder.trainable_variables \
                            + decoder.trainable_variables

    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))

    return loss

def if_training(train_X):
    """
    The training of an Isolation Forest as anomaly detector
    Parameters
    ----------
    train_ds: TF's Dataset. The trainind dataset
    """
    
    print("--- Training Isolation Forest ---")
    if_clf = IsolationForest(max_samples=10000, n_jobs=-1).fit(train_X)

    return if_clf

def lof_training(train_X):
    """
    The training of an Local Outlier Factor (LOF) as anomaly detector
    Parameters
    ----------
    train_ds: TF's Datase.t The training dataset
    """

    print("--- Training Local Outlier Factor ---")
    lof_clf = LocalOutlierFactor(novelty=True, n_jobs=-1).fit(train_X)

    return lof_clf

def ocs_training(train_X):
    """
    The training of an one-class SVM as anomaly detector
    Parameters
    ----------
    train_ds: TF's Dataset. The training dataset
    """

    print("--- Training One-Class SVM ---")
    ocs_clf = OneClassSVM().fit(train_X)

    return ocs_clf