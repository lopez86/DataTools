import funcy
import numpy as np

from .data_types import BatchData
from .util import split_data_dict


def make_batch_generator(
        n_epochs=5,
        batch_size=64,
        shuffle=True,
        random_state=75894,
        batches_per_group=-1,
        verbose=1
):
    generator = funcy.partial(
        generate_batches,
        n_epochs=n_epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        random_state=random_state,
        batches_per_group=batches_per_group,
        verbose=verbose
    )
    return generator


def generate_batches(data,
                     n_epochs=5,
                     batch_size=64,
                     shuffle=True,
                     random_state=75894,
                     batches_per_group=-1,
                     verbose=1):
    """Generate a series of batches.

    Args:
        data: Dataset
        n_epochs: int, number of epochs to generate
        batch_size: int, size of each batch. The last batch may be truncated
        shuffle: bool, if True, shuffle the data before each epoch
        random_state: int, seed for random number generator
        batches_per_group: int, if positive number of batches before setting
                           epoch_done to True
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

    batch_counter = 0
    for epoch in range(n_epochs):
        if verbose > 0:
            print('On epoch {} of {}'.format(epoch, n_epochs))
        if shuffle:
            perm = random.permutation(np.arange(size))
        else:
            perm = np.arange(size)
        for b in range(batches_per_epoch):
            batch_counter += 1
            if verbose > 1:
                if (int(b / batches_per_epoch * 100) >
                        int((b - 1) / batches_per_epoch * 100)):
                    print('{}% done'.format(int(b / batches_per_epoch * 100)), end='\r')
            indices = perm[b * batch_size: (b + 1) * batch_size]
            batch_inputs = split_data_dict(data.inputs, indices)
            batch_outputs = split_data_dict(data.outputs, indices)

            if ((batches_per_group <= 0 and b == batches_per_epoch - 1)
                    or (batches_per_group > 0 and batch_counter % batches_per_group == 0)):
                epoch_done = True
            elif b == batches_per_epoch - 1 and epoch == n_epochs - 1:
                epoch_done = True

            batch = BatchData(
                batch_inputs,
                outputs=batch_outputs,
                index=indices,
                epoch=epoch,
                batch=b,
                epoch_done=epoch_done,
                metadata={'description': 'Generated batch'}
            )
            yield batch
            epoch_done = False
