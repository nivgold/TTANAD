import numpy as np
import tensorflow as tf
import utils

def load_dataset(args):
    """
    Loading dataset to be ready for training and testing

    Parameters
    ----------
    args: argparse args. The args to the program    
    """

    # loading preprocssed data
    preproceseed_data, features_dim = load_preprocessed(args.dataset_name, args.window_size)

    datasets = make_dataset(preproceseed_data, args)

    return datasets, features_dim

def load_preprocessed(dataset_name, window_size):
    """
    Loading saved preprocessed dataset files (training set, test set, TTAs)

    Parameters
    ----------
    dataset_name: str. The name of the preprocessed dataset to load
    window_size: int. The size of the sliding window.
    """
    
    disk_path = '../data/' + dataset_name + "/"

    preprocessed_zip = np.load(disk_path + f"{dataset_name}_preprocessed_{window_size}_window_size.npz")

    # extracting training set features and labels
    train_features, train_labels = preprocessed_zip['train_features'], preprocessed_zip['train_labels']
    
    # extracting test set features and labels
    test_features, test_labels = preprocessed_zip['test_features'], preprocessed_zip['test_labels']

    # extracting TTAs features and labels
    tta_features, tta_labels = preprocessed_zip['tta_features'], preprocessed_zip['tta_labels']

    # getting the dimensionality of the dataset
    features_dim = train_features.shape[-1]
    
    # validating that the dimensionality of the trainint set is the same as in the test set
    assert train_features.shape[-1] == test_features.shape[-1]

    return (train_features, train_labels, test_features, test_labels, tta_features, tta_labels), features_dim

def make_dataset(preprocessed_data, args):
    """
    Creating a Tensorflow's Dataset to hold the training set and the test set with the TTAs

    Parameters
    ----------
    preprocessed_data. tuple. Containing the preprocsssed train features, train labels, test features, test labes and tta features, tta labels
    args: argparse args. The args to the program
    """

    # extractingthe preprocessed data
    train_features, train_labels = preprocessed_data[0], preprocessed_data[1]
    test_features, test_labels = preprocessed_data[2], preprocessed_data[3]
    tta_features, tta_labels = preprocessed_data[4], preprocessed_data[5]

    end_training_idx =  int(train_features.shape[0] * 0.7)
    train_train_features, train_train_labels = train_features[:end_training_idx, :], train_labels[:end_training_idx]
    validation_features, validation_labels = train_features[end_training_idx:, :], train_labels[end_training_idx:]
    train_ds, validation_ds, test_ds = utils.create_tf_datasets(train_train_features, train_train_labels, 
                                                                validation_features, validation_labels,
                                                                test_features, test_labels,
                                                                tta_features, tta_labels, args)
 
    return train_train_features, train_train_labels, \
           validation_features, validation_labels, \
           test_features, test_labels, \
           tta_features, tta_labels, \
           train_ds, validation_ds, test_ds

def train_pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


def test_pack_features_vector(features, labels, tta_features, tta_labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels, tta_features, tta_labels