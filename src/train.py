import tensorflow as tf

from autoencoder_model import Encoder, Decoder

def train(train_ds, features_dim, args):
    """
    Training the framework. For now, only training the anomaly detection estimator (AE).

    Parameters
    ----------
    train_ds: TF's Dataset. The training set
    features_dim: int. The dimensionality of the dataset
    args: argprase args. The args to the programa
    """
    
    trained_estimator = train_estimator(train_ds, features_dim, args)

    return trained_estimator

def train_estimator(train_ds, features_dim, args):
    """
    Training an Autoencoder as a anomaly detection estimator

    Parameters
    ----------
    train_ds: TF's Dataset. The training set
    features_dim: int. The dimensionality of the dataset
    args: argparse args. The args to the program
    """

    # setting up training configurations
    optimizer = tf.keras.optimizers.Adam()
    loss_func = tf.keras.losses.MeanSquaredError()
    encoder = Encoder(input_shape=features_dim)
    decoder = Decoder(original_dim=features_dim)

    # training loop
    for epoch in range(args.num_epochs):
        epoch_loss_mean = tf.keras.metrics.Mean()

        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            loss = train_step(x_batch_train, encoder, decoder, loss_func, optimizer)

            # keep track of the metrics
            epoch_loss_mean.update_state(loss)

        # update metrics after each epoch
        print(f'Epoch {epoch+1} loss mean: {epoch_loss_mean.result()}')

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