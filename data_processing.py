import random
import numpy as np
import pandas as pd
import pickledb
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def longterm_preprocessor(data_path, time_max_scale, train_ratio=0.5, valid_ratio=0.25):
    # load processed data
    data_name = data_path.split('/')[2].replace('.csv', '')
    db = pickledb.load(f'./dataset/{data_name}.db', auto_dump=False)
    
    db_key = '|'.join([data_name, str(train_ratio), str(valid_ratio), str(time_max_scale)])
    if db.exists(db_key):
        data = db.get(db_key)
        train_data, valid_data, test_data, num_features = data['train'], data['valid'], data['test'], data[
            'num_features']
        train_data, valid_data, test_data = list2np_in_zip(train_data), list2np_in_zip(valid_data), list2np_in_zip(
            test_data)
        return train_data, valid_data, test_data, num_features

    # read raw data
    df_total = pd.read_csv(data_path)
    df_raw = df_total.drop(columns = ['year', 'month', 'day', 'hour', 'minute', 'second'])
    # create time and feature indices
    df_coord = pd.DataFrame(index=df_raw.index, columns=df_raw.columns)
    time_indices, feature_indices = np.where(df_raw != df_coord)

    # train/valid/test data indices
    data_indices = list(range(0, time_indices.size))
    random.shuffle(data_indices)
    train_valid_border = int(len(data_indices) * train_ratio)
    valid_test_border = train_valid_border + int(len(data_indices) * valid_ratio)
    train_indices = data_indices[:train_valid_border]
    valid_indices = data_indices[train_valid_border:valid_test_border]
    test_indices = data_indices[valid_test_border:]
    
    # period time index encoding 
    min_year = df_total.year.unique().min()
    df_total.year = df_total.year - min_year
    max_val_year = df_total.year.unique().max()
    df_total.year = df_total.year / max_val_year # 0 ~ max_val_year
    
    df_total.month = (df_total.month - 1) / 11 # 1 ~ 12 
    df_total.day = (df_total.day - 1) / 30 # 1 ~ 31 
    df_total.hour = df_total.hour / 23 # 0 ~ 23 
    df_total.minute = df_total.minute / 59 # 0 ~ 59 
    df_total.second = df_total.second / 59 # 0 ~ 59 
    
    time_columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
    df_time = df_total[time_columns] * time_max_scale
    df_total['time_lst'] = df_time.values.tolist()
    df_total.time_lst = df_total.time_lst.apply(lambda x: np.array(x)).values

    time_indices = np.repeat(df_total.time_lst.values.flatten(), len(df_raw.columns))
    
    train_time_indices = time_indices[train_indices]
    valid_time_indices = time_indices[valid_indices]
    test_time_indices = time_indices[test_indices]
    
    # feature index encoding
    num_features = len(df_raw.columns)
    fie = OneHotEncoder(sparse=False)
    feature_fitting_data = np.array(list(range(0, num_features))).reshape(-1, 1)
    fie.fit(feature_fitting_data)
    feature_indices = fie.transform(feature_indices.reshape(-1, 1)).astype(np.float32)
    train_feature_indices = feature_indices[train_indices]
    valid_feature_indices = feature_indices[valid_indices]
    test_feature_indices = feature_indices[test_indices]
    
    # new value scaling
    data_scaler = StandardScaler()
    df_scaled = data_scaler.fit_transform(df_raw)
    values = df_scaled.flatten().astype(np.float32)

    train_values = values[train_indices]
    valid_values = values[valid_indices]
    test_values = values[test_indices]

    # zip train/test/valid data
    train_data = list(zip(train_time_indices, train_feature_indices, train_values))
    valid_data = list(zip(valid_time_indices, valid_feature_indices, valid_values))
    test_data = list(zip(test_time_indices, test_feature_indices, test_values))

    # save preprocessed data
    data = {'train': np2list_in_zip(train_data), 'valid': np2list_in_zip(valid_data), 'test': np2list_in_zip(test_data),
            'num_features': num_features}
    db.set(db_key, data)
    db.dump()

    return train_data, valid_data, test_data, num_features

def list2np_in_zip(zipped_data):
    list1, list2, list3 = zip(*zipped_data)
    list1, list2, list3 = np.array(list1, dtype=np.float32), np.array(list2, dtype=np.float32), \
        np.array(list3, dtype=np.float32)

    return list(zip(list1, list2, list3))


def np2list_in_zip(zipped_data):
    list1, list2, list3 = zip(*zipped_data)
    list1, list2, list3 = list(list1), list(list2), list(list3)

    list1 = [ele.tolist() for ele in list1]
    list2 = [ele.tolist() for ele in list2]
    list3 = [ele.tolist() for ele in list3]

    return list(zip(list1, list2, list3))
