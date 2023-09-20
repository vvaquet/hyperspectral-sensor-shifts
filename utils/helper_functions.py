import numpy as np

def get_label_indices(labels):
    unique_labels = np.unique(labels).astype(int)
    indices_labels = {l : np.where(labels == l)[0] for l in unique_labels}
    return indices_labels, unique_labels


def get_sensor_indices(sensors):
    unique_sensors = np.unique(sensors).astype(int)
    indices_sensors = {s : np.where(sensors == s)[0] for s in unique_sensors}
    return indices_sensors, unique_sensors