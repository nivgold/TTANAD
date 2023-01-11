import tensorflow as tf


def create_tf_datasets(train_features,
                       train_labels,
                       validation_features,
                       validation_labels,
                       test_features,
                       test_labels,
                       tta_features,
                       tta_labels,
                       args):
    '''
    Creating tensorflow's datasets.

    Parameters
    ----------
    train_features: The training set features of shape (#training,n)
    train_labels: The training set labels of shape (#training,)
    validation_features: The validation set features of shape (#validation,n)
    validation_labels: The validation set labels of shape (#validation,)
    test_features: The test set features of shape (#test, n)
    test_labels: The test set labels of shape (#test,)
    tta_features: The test-time-augmentation features of shape (#test, #augmentations, n)
    tta_labels: The test-time-augmentation labels of shape (#test, #augmentations)
    '''
    # creating training dataset
    train_ds = (tf.data.Dataset.from_tensor_slices((train_features, train_labels))
                .cache()
                .batch(args.batch_size)
                # .map(train_pack_features_vector)
    )

    # creating validation dataset
    validation_ds = (tf.data.Dataset.from_tensor_slices((validation_features, validation_labels))
                .cache()
                .batch(args.batch_size)
                # .map(train_pack_features_vector)      
    )

    # creating test dataset
    test_ds = (tf.data.Dataset.from_tensor_slices((test_features, test_labels, tta_features, tta_labels))
                .cache()
                .batch(args.batch_size)
                # .map(test_pack_features_vector)
    )

    return train_ds, validation_ds, test_ds