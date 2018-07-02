import pandas as pd


def get_number_of_samples(dataset):
    """Get the number of samples held in a dataset."""
    keys = list(dataset.inputs.keys())
    if not keys:
        raise AssertionError('Dataset has no inputs!')

    first_set = dataset.inputs[keys[0]]
    if hasattr(first_set, 'shape'):
        return first_set.shape[0]
    return len(first_set)


def get_number_of_outputs(output_data):
    """Get the number of output variables for a given output array."""
    if not hasattr(output_data, 'shape'):
        raise AttributeError(
            'Output data types must have attribute "shape".'
        )

    if len(output_data.shape) == 1:
        return 1

    return output_data.shape[1]


def split_data(data, indices):
    """Get the data only at the given indices.

    Args:
        data: array-like
        indices: list-like of int

    Returns:
        array-like
    """
    if isinstance(data, pd.DataFrame):
        return data.iloc[indices, :]
    elif isinstance(data, list):
        return [data[i] for i in indices]
    return data[indices]


def split_data_dict(data_dict, indices):
    """Get the data only at the give indices for a dict of arrays.

    Args:
        data_dict: a dict with array-like values
        indices: the indices to grab

    Returns:
        dict
    """
    if data_dict is None:
        return None
    return {
        key: split_data(d, indices)
        for key, d in data_dict.items()
    }
