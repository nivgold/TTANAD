import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm

DATASET_DATA_FILE = {
    'ids17': {
        'data': 'ids2017_data.csv',
        'normal_class': 'BENIGN',
        'label_col': ' Label',
        'features': [' Flow Duration',
                            ' Total Fwd Packets', ' Total Backward Packets',
                            'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
                            ' Fwd Packet Length Max', ' Fwd Packet Length Min',
                            ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
                            'Bwd Packet Length Max', ' Bwd Packet Length Min',
                            ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
                            ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
                            ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
                            ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
                            ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
                            ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags',
                            ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s',
                            ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',
                            ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
                            'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count',
                            ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
                            ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
                            ' Average Packet Size', ' Avg Fwd Segment Size',
                            ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
                            ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
                            ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
                            ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
                            'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
                            ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean',
                            ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
                            ' Idle Max', ' Idle Min']
    },
    'ids18': {
        'data': 'ids2018_data.csv',
        'normal_class': 'Benign',
        'label_col': 'Label',
        'features': ['Flow Duration', 'Tot Fwd Pkts',
                               'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
                               'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
                               'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
                               'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
                               'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
                               'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                               'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
                               'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
                               'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
                               'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
                               'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
                               'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
                               'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
                               'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
                               'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
                               'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
                               'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
                               'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
                               'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
                               'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
    }
}

def reduce_mem_usage(df):
    """ 
    iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    Parameters:
    df: pandas Dataframe. A preprocessed dataset
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('--- Memory usage of dataframe is {:.2f} MB ---'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('--- Memory usage after optimization is: {:.2f} MB ---'.format(end_mem))
    print('--- Decreased by {:.1f}% ---'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def preprocessing(dataset_name, args):
    """
    Preprocessing the given dataset and producing `num_augmentations` TTA for each 5'th test instance

    Paramaters
    ----------
    dataset_name: str. The name of the given dataset
    args: argparse ArgumentParser. The arguments given to the program
    """

    window_size = args.window_size
    test_ratio = args.test_ratio

    # retrieving essential info on the given dataset
    data_file = DATASET_DATA_FILE[dataset_name]['data']
    normal_class = DATASET_DATA_FILE[dataset_name]['normal_class']
    label_col = DATASET_DATA_FILE[dataset_name]['label_col']
    features = DATASET_DATA_FILE[dataset_name]['features']

    data_file_path = '../data/' + dataset_name + '/' + data_file

    print(f"--- Openning {dataset_name} ---")
    
    if args.nrows != -1:
        df = pd.read_csv(data_file_path, nrows=args.nrows)
    else:
        df = pd.read_csv(data_file_path)
    
    # reduce memort consumption
    df = reduce_mem_usage(df)

    # change target to be 0 for normal and 1 for anomaly
    print(f"--- Converting Target to int dtype ---")
    df = convert_target_to_int(df, label_col, normal_class)

    # splitting to train and test
    print(f"--- Train-Test Split ---")
    train_df, test_df = train_test_split(df, test_ratio)

    # filter out anomalous instances from training set
    print(f"--- Filtering out anomalous samples from training set ---")
    train_df = train_df[train_df[label_col] == 0]

    # rolling the dataset to obtain statistics every 5 'ticks'
    print(f'--- Rolling training set ---')
    train_df = roll_dataset(train_df, features, label_col, window_size)
    print(f'--- Rolling test set ---')
    test_df = roll_dataset(test_df, features, label_col, window_size)

    # normalizing
    print(f'--- Normalizing training set ---')
    train_df = minmax_normalizer(train_df, label_col)
    print(f'--- Normalizing test set ---')
    test_df = minmax_normalizer(test_df, label_col)

    # create tta
    print(f'--- Creating TTAs ---')
    tta_features, tta_labels = create_TTA(test_df, label_col, window_size)

    # extracting features and labels from training set
    train_labels = train_df[label_col]
    train_features = train_df.drop(columns=[label_col], axis=1)

    # extracting features and labels from test set
    test_labels = test_df[window_size:-window_size:window_size][label_col]
    test_features = test_df[window_size:-window_size:window_size].drop(columns=[label_col], axis=1)

    # saving preprocessed dataset to disk
    print(f'--- Saving Preprocessed files to disk ---')
    save_to_disk(train_features, train_labels, test_features, test_labels, tta_features, tta_labels, dataset_name, window_size)

def convert_target_to_int(df, label_col, normal_class):
    """
    Converting the label column to be int dtype. 0 denote normal while 1 denote anomaly

    Parameters
    ----------
    df: pandas' DataFrame. The dataset
    label_col: str. The target column name
    normal_class: The name of the class that indicate "not anomalous"
    """

    df[label_col] = np.where(df[label_col] == normal_class, 0, 1)
    
    return df

def train_test_split(df, test_ratio):
    """
    Splitting the given dataset to train and test by preserving the order between the samples (it is time series data).
    
    Parameters
    ----------
    df: pandas' DataFrame. The dataset
    test_ratio: float. Indicate the ratio of the test after the split
    """

    train_ratio = 1 - test_ratio
    
    train_last_idx = int(len(df) * train_ratio)
    train_df = df.iloc[train_last_idx:, :].reset_index(drop=True)
    test_df = df.iloc[:train_last_idx, :].reset_index(drop=True)
    
    return train_df, test_df

def roll_dataset(df, features, label_col, window_size):
    """
    Convolve a window over the given data and extact statistics (min, max, std).

    Parameters
    ----------
    df: pandas' DataFrame. The given data
    features: list. The features of the given data
    label_col: str. The name of the target column of the given data
    window_size: int. The size of the window that does the aggregation
    """

    df_features = df[features]
    df_label = df[label_col]

    # sliding the window
    rolled_max = df_features.rolling(window_size).max()
    rolled_max.columns = [f'MAX - {column}' for column in list(rolled_max.columns)]

    rolled_min = df_features.rolling(window_size).min()
    rolled_min.columns = [f'MIN - {column}' for column in list(rolled_min.columns)]

    rolled_std = df_features.rolling(window_size).std()
    rolled_std.columns = [f'STD - {column}' for column in list(rolled_std.columns)]

    rolled_labels = df_label.rolling(window_size).max()

    rolled = pd.concat([rolled_max, rolled_min, rolled_std, rolled_labels], axis=1)
    rolled = rolled.fillna(rolled.mean())
    rolled[label_col] = rolled[label_col].astype(np.int16)

    return rolled

def minmax_normalizer(df, label_col, scaler=None):
    """
    Performing Max-Min scaling

    Parameters
    ----------
    df: pandas' DataFrame. The given data
    label_col: str. The name of the label column of the given data
    scaler: scikit-learn's MinMaxScaler. If exist, then the normalization is done using the given scaler
    """

    features = list(set(df.columns) - set(label_col))

    if scaler is None:
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
        
        return df
    
    df[features] = scaler.transform(df[features])

    return df

def create_TTA(test_df, label_col, window_size):
    """
    Creating the synthetic samples which will be the TTA later in the test phase

    Parameters
    ----------
    test_df: pandas' DataFrame. The test set from which the TTA is created
    label_col: str. Denote the name of the target column
    window_size: int. The size of the window.
    """

    test_norm_df = test_df[window_size:-window_size:window_size]

    tta_labels = []
    tta_features = []

    # making the TTA data
    for index in tqdm(test_norm_df.index):
        index = int(index)
        # take 4-before current index
        before_tta_samples = test_df.iloc[index - (window_size-1): index]
        # take 4-after current index
        after_tta_samples = test_df.iloc[index + 1: index + window_size]
        # concat 4-before and 4-after df
        current_tta = pd.concat([before_tta_samples, after_tta_samples], axis=0)

        tta_labels.append(current_tta[label_col].values)
        tta_features.append(current_tta.drop(columns=[label_col], axis=1).values)
    
    return tta_features, tta_labels

def save_to_disk(train_features, train_labels, test_features, test_labels, tta_features, tta_labels, dataset_name, num_TTAs):
    """
    Saving the preprocessed data to the disk

    Parameters
    ----------
    train_features: pandas' DataFrame. The features of the training set
    train_labels: pandas' Series. The labels of the training set
    test_features:  pandas' DataFrame. The features of the test set
    test_labels: pandas' Series. The labels of the test set
    tta_features:  pandas' DataFrame. The features of the tta data
    tta_labels: pandas' Series. The labels of the tta data
    dataset_name: str. The name of the current preprocessed datasetet
    num_TTAs: int. The number of TTAs to create which is also the size of the sliding window
    """

    disk_path = "../data/" + dataset_name + "/"

    # saving in a compressed zip file
    np.savez_compressed(disk_path + f"{dataset_name}_preprocessed_{num_TTAs}_TTA", train_features=train_features.values, train_labels=train_labels.values, test_features=test_features.values, test_labels=test_labels.values, tta_features=tta_features, tta_labels=tta_labels)


if __name__ == '__main__':

    # getting the dataset to preprocess
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True, dest='dataset_name', type=str, help='the dataset to preprocess and save to disk for later use')
    parser.add_argument('-n', '--nrows', dest='nrows', type=int, default=-1, help='The number of rows to read from the dataset')
    parser.add_argument('-w', '--windowsize', dest='window_size', type=int, default=5, help='The size of the window')
    parser.add_argument('-t', '--testratio', dest='test_ratio', type=float, default=0.3, help='The test ratio in the train-test split')
    args = parser.parse_args()

    dataset_name = args.dataset_name.lower()

    if dataset_name not in DATASET_DATA_FILE.keys():
        raise ValueError("Provided dataset is not supported")

    # checking if the given dataset is IDS18 so we shouldn't load all the data
    if dataset_name == 'ids18' and args.nrows == -1:
        # if IDS18 was selected for preprocessing and nrows remained -1, the default number of rows to load is 3M.
        args.nrows = 3000000
    print(f"--- Starting preprocess {dataset_name} dataset ---")
    preprocessing(dataset_name, args)