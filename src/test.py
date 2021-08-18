import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

evaluated_algorithms = [
    'WO_TTA_Baseline',
    'Sliding_Window_TTA',
]


def test(test_ds, trained_estimator, args):
    """
    Performing test phase on the test set with all of the compared algorithms described in the Experiments section

    Parameters
    ----------
    test_ds: TF's Dataset. The test set
    trained_estimator: tuple. The trained estimator. In that case, containing the trained encoder and decoder
    args: argparse args. The args to the program
    """

    # testing
    algorithms_metrics = test_loop(test_ds, trained_estimator, args)

    # printing experiments results
    print_test_results(algorithms_metrics, args)

def test_loop(test_ds, trained_estimator, args):
    """
    The test loop implementation on every evaluated algorithm

    Parameters
    ----------
    test_ds: TF's Dataset. The test set
    trained_estimator: tuple. The trained estimator. In that case, containing the trained encoder and decoder
    args: argparse args. The args to the program
    """

    # extracting encoder & decoder
    encoder, decoder = trained_estimator

    # setting up training configurations
    loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    algorithms_test_loss = {algorithm: [] for algorithm in evaluated_algorithms}    
    algorithms_metrics = {algorithm: {} for algorithm in evaluated_algorithms}

    test_labels = []
    tqdm_total_bar = test_ds.cardinality().numpy()
    for step, (x_batch_test, y_batch_test, tta_features_batch, tta_labels_batch) in tqdm(enumerate(test_ds), total=tqdm_total_bar):
        reconstruction_loss = test_step(x_batch_test, encoder, decoder, loss_func).numpy()

        # saving ground-truth
        test_labels.append(y_batch_test.numpy())

        # saving baseline test loss
        algorithms_test_loss['WO_TTA_Baseline'].append(reconstruction_loss)

        # saving TTA test loss
        tta_reconstructions = test_step(tta_features_batch, encoder, decoder, loss_func).numpy()
        for original_loss, tta_loss in list(zip(reconstruction_loss, tta_reconstructions)):
            aggregated_loss = np.concatenate([[original_loss], tta_loss], axis=0)
            algorithms_test_loss['Sliding_Window_TTA'].append(np.mean(aggregated_loss))

    # flatten
    algorithms_test_loss['WO_TTA_Baseline'] = np.concatenate(algorithms_test_loss['WO_TTA_Baseline'], axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    y_true = np.asarray(test_labels).astype(int)

    # calculating AUC
    for algorithm in evaluated_algorithms:
        algorithms_metrics[algorithm]['AUC'] = roc_auc_score(y_true, algorithms_test_loss[algorithm])

    return algorithms_metrics

@tf.function
def test_step(inputs, encoder, decoder, loss_func):
    latent_var = encoder(inputs)
    reconstructed = decoder(latent_var)
    reconstruction_loss = loss_func(inputs, reconstructed)

    return reconstruction_loss

def print_test_results(algorithms_metrics_dict, args):
    """
    Printing the results metrics of all of the evaluated algorithms

    Parameters
    ----------
    algorithms_metrics_dict: dict. Containing the evaluaed metrics for each evaluated algorithm
    args: argparse args. The args to the program    
    """
    
    print(f"--- Reults of {args.dataset_name} ---")
    for algorithm, metrics in algorithms_metrics_dict.items():
        algorithm_name = algorithm.replace("_", " ")
        print("*"*100)
        print(f"--- {algorithm_name} ---")
        # print_list = []
        # for i in range(len(folds_metrics.mean(axis=0))):
        #     print_list.append(folds_metrics.mean(axis=0)[i])
        #     print_list.append(folds_metrics.std(axis=0)[i])
        # print("AUC : {:0.3f}+-{:0.3f}".format(*print_list))
        print("AUC: {AUC}".format(**algorithms_metrics_dict[algorithm]))
    
    print("*"*100)
