import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve

from prettytable import PrettyTable

evaluated_algorithms = [
    'WO_TTA_Baseline',
    'Sliding_Window_TTA',
]

AUTOENCODER = "Autoencoder"
ISOLATION_FOREST = "Isolation Forest"
LOCAL_OUTLIER_FACTOR = "Local Outlier Factor"
ONE_CLASS_SVM = "One-Class SVM"

evaluated_estimators = [
    # 'Isolation Forest',
    # 'Local Outlier Factor',
    # 'One-Class SVM',
    # 'Autoencoder'
]


def test(test_ds, trained_models, args):
    """
    Performing test phase on the test set with all of the compared algorithms described in the Experiments section

    Parameters
    ----------
    test_ds: TF's Dataset. The test set
    if_estimator: scikit-learn's IsolationForest. The trained Isolation Forest model.
    lof_estiamtor: scikit-learn's LocalOutlierFactor. The trained Local Outlier Factor model.
    ocs_estimator: scikit-learn's OneClassSVM. The trained One Class SVM model.
    AE_estimator: tuple of tensorflow's Dataset. The trained AE estimator. In that case, containing the trained encoder and decoder
    args: argparse args. The args to the program
    """

    # testing
    algorithms_metrics = test_loop(test_ds, **trained_models)

    # printing experiments results
    print_test_results(algorithms_metrics, args)

def test_loop(test_ds, AE_estimator=None, if_estimator=None, lof_estimator=None, ocs_estimator=None):
    """
    The test loop implementation on every evaluated algorithm

    Parameters
    ----------
    test_ds: TF's Dataset. The test set
    if_estimator: scikit-learn's IsolationForest. The trained Isolation Forest model.
    lof_estiamtor: scikit-learn's LocalOutlierFactor. The trained Local Outlier Factor model.
    ocs_estimator: scikit-learn's OneClassSVM. The trained One Class SVM model.
    AE_estimator: tuple of tensorflow's Dataset. The trained AE estimator. In that case, containing the trained encoder and decoder
    args: argparse args. The args to the program
    """
    global evaluated_estimators
    if AE_estimator is not None:
        evaluated_estimators.append(AUTOENCODER)
    if if_estimator is not None:
        evaluated_estimators.append(ISOLATION_FOREST)
    if lof_estimator is not None:
        evaluated_estimators.append(LOCAL_OUTLIER_FACTOR)
    if ocs_estimator is not None:
        evaluated_estimators.append(ONE_CLASS_SVM)

    # extracting encoder & decoder
    if AE_estimator is not None:
        encoder, decoder = AE_estimator

    # setting up training configurations
    loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    algorithms_test_loss = {algorithm: {estimator: [] for estimator in evaluated_estimators} for algorithm in evaluated_algorithms}
    algorithms_metrics = {algorithm: {estimator: {} for estimator in evaluated_estimators} for algorithm in evaluated_algorithms}

    test_labels = []
    test_X = []
    tta_X = []
    tqdm_total_bar = test_ds.cardinality().numpy()
    for step, (x_batch_test, y_batch_test, tta_features_batch, tta_labels_batch) in tqdm(enumerate(test_ds), total=tqdm_total_bar):
        test_X.append(x_batch_test)
        tta_X.append(tta_features_batch)
        # saving ground-truth
        test_labels.append(y_batch_test.numpy())

        
        if AE_estimator is not None:
            # saving anomaly score on original test instace
            AE_reconstruction_loss = test_step(x_batch_test, encoder, decoder, loss_func).numpy() # type: ignore
            # saving baseline test loss
            algorithms_test_loss['WO_TTA_Baseline']['Autoencoder'].append(AE_reconstruction_loss)
            # saving TTA test loss
            AE_tta_reconstructions = test_step(tta_features_batch, encoder, decoder, loss_func).numpy() # type: ignore

            for original_loss, tta_loss in list(zip(AE_reconstruction_loss, AE_tta_reconstructions)):
                aggregated_loss = np.concatenate([[original_loss], tta_loss], axis=0)
                algorithms_test_loss['Sliding_Window_TTA']['Autoencoder'].append(np.mean(aggregated_loss))
        

        # for original_loss, tta_loss in list(zip(if_anomaly_score, if_tta_anomaly_score)):
        #     aggregated_loss = np.concatenate([[original_loss], tta_loss], axis=0)
        #     algorithms_test_loss['Sliding_Window_TTA']['Isolation Forest'].append(np.mean(aggregated_loss))

        # for original_loss, tta_loss in list(zip(lof_anomaly_score, lof_tta_anomaly_score)):
        #     aggregated_loss = np.concatenate([[original_loss], tta_loss], axis=0)
        #     algorithms_test_loss['Sliding_Window_TTA']['Local Outlier Factor'].append(np.mean(aggregated_loss))

        # for original_loss, tta_loss in list(zip(ocs_anomaly_score, ocs_tta_anomaly_score)):
        #     aggregated_loss = np.concatenate([[original_loss], tta_loss], axis=0)
        #     algorithms_test_loss['Sliding_Window_TTA']['One-Class SVM'].append(np.mean(aggregated_loss))


    # flatten
    test_X = np.concatenate(test_X, axis=0)
    tta_X = np.concatenate(tta_X, axis=0)

    num_samples, num_tta, num_features = tta_X.shape

    if if_estimator is not None:

        if_anomaly_score = if_test_step(test_X, if_estimator)
        algorithms_test_loss['WO_TTA_Baseline']['Isolation Forest'] = if_anomaly_score # type: ignore
        if_tta_anomaly_score = if_test_step(tta_X.reshape(num_samples*num_tta, num_features), if_estimator).reshape(num_samples, num_tta)
        if_total_anomaly_score = np.concatenate([if_tta_anomaly_score, np.expand_dims(if_anomaly_score, axis=1)], axis=1)
        algorithms_test_loss['Sliding_Window_TTA']['Isolation Forest'] = np.mean(if_total_anomaly_score, axis=1)
    


    if lof_estimator is not None:

        lof_anomaly_score = lof_test_step(test_X, lof_estimator)
        algorithms_test_loss['WO_TTA_Baseline']['Local Outlier Factor'] = lof_anomaly_score # type: ignore
        lof_tta_anomaly_score = lof_test_step(tta_X.reshape(num_samples*num_tta, num_features), lof_estimator).reshape(num_samples, num_tta)
        lof_total_anomaly_score = np.concatenate([lof_tta_anomaly_score, np.expand_dims(lof_anomaly_score, axis=1)], axis=1)
        algorithms_test_loss['Sliding_Window_TTA']['Local Outlier Factor'] = np.mean(lof_total_anomaly_score, axis=1)


    if ocs_estimator is not None:

        ocs_anomaly_score = ocs_test_step(test_X, ocs_estimator)
        algorithms_test_loss['WO_TTA_Baseline']['One-Class SVM'] = ocs_anomaly_score # type: ignore
        ocs_tta_anomaly_score = ocs_test_step(tta_X.reshape(num_samples*num_tta, num_features), ocs_estimator).reshape(num_samples, num_tta)
        ocs_total_anomaly_score = np.concatenate([ocs_tta_anomaly_score, np.expand_dims(ocs_anomaly_score, axis=1)], axis=1)
        algorithms_test_loss['Sliding_Window_TTA']['One-Class SVM'] = np.mean(ocs_total_anomaly_score, axis=1)


    # algorithms_test_loss['WO_TTA_Baseline']['Isolation Forest'] = np.concatenate(algorithms_test_loss['WO_TTA_Baseline']['Isolation Forest'], axis=0)
    # algorithms_test_loss['WO_TTA_Baseline']['Local Outlier Factor'] = np.concatenate(algorithms_test_loss['WO_TTA_Baseline']['Local Outlier Factor'], axis=0)
    # algorithms_test_loss['WO_TTA_Baseline']['One-Class SVM'] = np.concatenate(algorithms_test_loss['WO_TTA_Baseline']['One-Class SVM'], axis=0)

    if AE_estimator is not None:
        algorithms_test_loss['WO_TTA_Baseline']['Autoencoder'] = np.concatenate(algorithms_test_loss['WO_TTA_Baseline']['Autoencoder'], axis=0) # type: ignore

    test_labels = np.concatenate(test_labels, axis=0)

    y_true = np.asarray(test_labels).astype(int)

    # calculating AUC
    for algorithm in evaluated_algorithms:
        for estimator in evaluated_estimators:
            algorithms_metrics[algorithm][estimator]['AUC'] = roc_auc_score(y_true, algorithms_test_loss[algorithm][estimator])
            optimal_f1, optimal_precision, optimal_recall = get_optimal_f1_precision_recall(y_true, algorithms_test_loss[algorithm][estimator])
            algorithms_metrics[algorithm][estimator]['F1'] = optimal_f1
            algorithms_metrics[algorithm][estimator]['Precision'] = optimal_precision
            algorithms_metrics[algorithm][estimator]['Recall'] = optimal_recall


    return algorithms_metrics

@tf.function
def test_step(inputs, encoder, decoder, loss_func):
    latent_var = encoder(inputs)
    reconstructed = decoder(latent_var)
    reconstruction_loss = loss_func(inputs, reconstructed)

    return reconstruction_loss

def get_optimal_f1_precision_recall(y_true, y_pred):
    '''
    Maximizing F1 score to find the optimal threshold
    '''

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    fscore = 2 * (precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.nanargmax(fscore)
    optimal_thr = thresholds[ix]
    optimal_f1 = fscore[ix]
    optimal_precision = precision[ix]
    optimal_recall = recall[ix]

    return optimal_f1, optimal_precision, optimal_recall


def if_test_step(test_X, trained_if):
    """
    A test phase with Isolation Forest on the given test set
    Parameters
    ----------
    test_X: numpy ndarray of shape (batch_size, num_augmentations, dataset's features dim) or (batch_size, dataset's features dim). The batch test set
    trained_if: scikit-learn's IsolationForest. The trained Isolation Forest as anomaly detection estimator
    """

    if len(test_X.shape) == 3:
        batch_anomaly_score = []
        for one_test_tta_samples in test_X:
            anomaly_score = -1 * trained_if.score_samples(one_test_tta_samples)
            batch_anomaly_score.append(anomaly_score)

        anomaly_score = np.array(batch_anomaly_score)
    else:
        anomaly_score = -1 * trained_if.score_samples(test_X)

    return anomaly_score

def lof_test_step(test_X, trained_lof):
    """
    A test phase with Local Outlier Factor on the given test set
    Parameters
    ----------
    test_X: numpy ndarray of shape (batch_size, dataset's features dim). The batch test set
    trained_lof: scikit-learn's LocalOutlierFactor. The trained Local Outlier Factor as anomaly detection estimator
    """

    if len(test_X.shape) == 3:
        batch_anomaly_score = []
        for one_test_tta_samples in test_X:
            anomaly_score = -1 * trained_lof.score_samples(one_test_tta_samples)
            batch_anomaly_score.append(anomaly_score)

        anomaly_score = np.array(batch_anomaly_score)
    else:
        anomaly_score = -1 * trained_lof.score_samples(test_X)

    return anomaly_score

def ocs_test_step(test_X, trained_ocs):
    """
    A test phase with One-Class SVM on the given test set
    Parameters
    ----------
    test_X: numpy ndarray of shape (batch_size, dataset's features dim). The batch test set
    trained_ocs: scikit-learn's OneClassSVM. The trained One-Class SVM as anomaly detection estimator
    """

    if len(test_X.shape) == 3:
        batch_anomaly_score = []
        for one_test_tta_samples in test_X:
            anomaly_score = -1 * trained_ocs.score_samples(one_test_tta_samples)
            batch_anomaly_score.append(anomaly_score)

        anomaly_score = np.array(batch_anomaly_score)
    else:
        anomaly_score = -1 * trained_ocs.score_samples(test_X)

    return anomaly_score

def print_test_results(algorithms_metrics_dict, args):
    """
    Printing the results metrics of all of the evaluated algorithms

    Parameters
    ----------
    algorithms_metrics_dict: dict. Containing the evaluaed metrics for each evaluated algorithm and each evaluated estimator
    args: argparse args. The args to the program    
    """

    print(f"--- Reults of {args.dataset_name} with window size {args.window_size}---")

    # AUC
    print_metric_table(algorithms_metrics_dict, "AUC")
    # F1
    print_metric_table(algorithms_metrics_dict, "F1")
    # Precision
    print_metric_table(algorithms_metrics_dict, "Precision")
    # Recall
    print_metric_table(algorithms_metrics_dict, "Recall")

def print_metric_table(algorithms_metrics_dict, metric):
    table = PrettyTable()
    table.field_names = ['Algorithm'] + evaluated_estimators.copy()

    print(f"--- {metric} ---")
    for algorithm in algorithms_metrics_dict:
        estimator_row = []
        estimator_row.append(algorithm)
        for estimator in evaluated_estimators:
            current_auc = algorithms_metrics_dict[algorithm][estimator][metric]
            estimator_row.append(current_auc)
        
        table.add_row(estimator_row)

    print(table)
    print("*"*100)