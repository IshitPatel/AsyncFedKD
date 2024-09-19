
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import tensorflow as tf
import os


def get_index_for_client(num_clients, client_id=0):

    if client_id > num_clients:
        client_id = 0

    calcifications = pd.read_csv('calcifications.csv')
    y = calcifications['severity'].copy()
    X = calcifications.drop(columns=['severity'])

    skf = StratifiedKFold(n_splits=num_clients)

    indexes = None
    idx = 0

    for train_index, test_index in skf.split(X, y):

        if idx == client_id:
            indexes = test_index
            break

        idx += 1

    return indexes


def get_data(client_id=0, data_file='./calcifications.csv'):

    num_clients = 4
    idxs = get_index_for_client(num_clients, client_id=client_id)

    # Loading the preprocessed data.
    calcifications = pd.read_csv(data_file)
    calcifications = calcifications.iloc[idxs]

    print(len(calcifications))

    # Generate image data.
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=.25, height_shift_range=.10, width_shift_range=.10, rotation_range=30, rescale=1/255.)

    # Get training data.
    train_data = data_generator.flow_from_dataframe(calcifications, x_col="subsample_path", y_col="severity", class_mode="categorical", target_size=(48,48), subset="training", color_mode="rgb", shuffle=True)

    # Get test data.
    test_data = data_generator.flow_from_dataframe(calcifications, x_col="subsample_path", y_col="severity", class_mode="categorical", target_size=(48,48), subset="validation", color_mode="rgb", shuffle=True)

    return train_data, test_data


# import tensorflow as tf
# import pandas as pd

# def get_data(data_file='./calcifications.csv'):

#     # Loading the preprocessed data.
#     calcifications = pd.read_csv(data_file)

#     # Generate image data.
#     data_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=.25, height_shift_range=.10, width_shift_range=.10, rotation_range=30, rescale=1/255.)

#     # Get training data.
#     train_data = data_generator.flow_from_dataframe(calcifications, x_col="subsample_path", y_col="severity", class_mode="categorical", target_size=(48,48), subset="training", color_mode="rgb", shuffle=True)

#     # Get test data.
#     test_data = data_generator.flow_from_dataframe(calcifications, x_col="subsample_path", y_col="severity", class_mode="categorical", target_size=(48,48), subset="validation", color_mode="rgb", shuffle=True)

#     return train_data, test_data

