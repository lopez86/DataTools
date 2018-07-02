import numpy as np


def simple_feed_builder(
    data,
    is_train,
    istrain_str='is_train',
    sparse=None
):
    """Build a tensorflow feed-dict.

    Args:
        data: Dataset instance
        is_train: bool, whether this is a training batch or not
        istrain_str: str, name of the tensor identifying a training batch
        sparse: Maybe(dict), should have the same keys as the inputs
                When true, treat as a sparse input

    Returns:
        dict, a feed dict to use as input to a Tensorflow session
    """
    if data is None:
        return None

    if sparse is None:
        sparse = {key: False for key in data.inputs.keys()}

    feed_dict = {}
    for key, input_data in data.inputs.items():
        input_dict = _make_input_dict(key, input_data, sparse=sparse[key])
        feed_dict.update(input_dict)

    if data.outputs is not None:
        for key, output_data in data.outputs.items():
            output_dict = _make_input_dict(key, output_data, sparse=False)
            feed_dict.update(output_dict)

    feed_dict[istrain_str + ':0'] = is_train

    return feed_dict


def _make_input_dict(name, input_data, sparse=False):
    """Create a Tensorflow feed-dict for a particular data array.

    Args:
        name: str, the name of the corresponding tensorflow variable
        input_data: Array-like, typically a numpy array
        sparse: bool, if True, create a dict for a Tensorflow sparse
                tensor.

    Returns:
        dict
    """
    if not sparse:
        feed_dict = {name + ':0': input_data}
    else:
        input_coo = input_data.tocoo()
        indices = np.mat([input_coo.row, input_coo.col]).transpose()
        feed_dict = {
            name + '/indices:0': indices,
            name + '/values:0': input_coo.data,
            name + '/shape:0': input_coo.shape
        }

    return feed_dict
