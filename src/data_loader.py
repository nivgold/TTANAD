import numpy as np
import tensorflow as tf

def load_dataset(args):
    """
    Loading dataset to be ready for training and testing

    Parameters
    ----------
    args: argparse args. The args to the program    
    """

    # loading preprocssed data
    preproceseed_data, features_dim = load_preprocessed(args.dataset_name)

    train_ds, test_ds = make_dataset(preproceseed_data, args)

    return train_ds, test_ds, features_dim

def load_preprocessed(dataset_name):
    """
    Loading saved preprocessed dataset files (training set, test set, TTAs)

    Parameters
    ----------
    dataset_name: str. The name of the preprocessed dataset to load
    """
    
    disk_path = '../data/' + dataset_name + "/"

    # # loading training set features and labels
    # train_features = np.load(disk_path + f"{dataset_name}_train_features.npy")
    # train_labels = np.load(disk_path + f"{dataset_name}_train_labels.npy")

    # # loading test set features and labels
    # test_features = np.load(disk_path + f"{dataset_name}_test_features.npy")
    # test_labels = np.load(disk_path + f"{dataset_name}_test_labels.npy")

    # # loading TTAs
    # tta_features = np.load(disk_path + f"{dataset_name}_tta_features.npy")
    # tta_labels = np.load(disk_path + f"{dataset_name}_tta_labels.npy")

    preprocessed_zip = np.load(disk_path + f"{dataset_name}_preprocessed.npz")

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

    # creating training dataset
    train_ds = (tf.data.Dataset.from_tensor_slices((train_features, train_labels))
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

    return train_ds, test_ds

def train_pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


def test_pack_features_vector(features, labels, tta_features, tta_labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels, tta_features, tta_labels