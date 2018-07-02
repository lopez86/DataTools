import numpy as np

from .data_types import BatchData
from .util import split_data_dict


def generate_batches(data,
                     n_epochs=5,
                     batch_size=64,
                     shuffle=True,
                     random_state=75894,
                     verbose=1):
    """Generate a series of batches.

    Args:
        data: Dataset
        n_epochs: int, number of epochs to generate
        batch_size: int, size of each batch. The last batch may be truncated
        shuffle: bool, if True, shuffle the data before each epoch
        random_state: int, seed for random number generator
        verbose: int, higher values will print more

    Yields:
        BatchData instance
    """
    input_keys = list(data.inputs.keys())
    if not input_keys:
        return
    size = data.inputs[input_keys[0]].shape[0]
    epoch_done = False
    batches_per_epoch = size // batch_size
    if size % batch_size > 0:
        batches_per_epoch += 1
    if shuffle:
        random = np.random.RandomState(seed=random_state)
    else:
        random = None

    for epoch in range(n_epochs):
        if verbose > 0:
            print('On epoch {} of {}'.format(epoch, n_epochs))
        if shuffle:
            perm = random.permutation(np.arange(size))
        else:
            perm = np.arange(size)
        for b in range(batches_per_epoch):
            indices = perm[b * batch_size: (b + 1) * batch_size]
            batch_inputs = split_data_dict(data.inputs, indices)
            batch_outputs = split_data_dict(data.outputs, indices)

            if b == batches_per_epoch - 1:
                epoch_done = True

            batch = BatchData(
                batch_inputs,
                outputs=batch_outputs,
                epoch=epoch,
                batch=b,
                epoch_done=epoch_done,
                metadata={'description': 'Generated batch'}
            )
            yield batch
            epoch_done = False
