"""Provide all the data processing methods"""
import collections
import random
import numpy as np

data_index = 0


def rewind():
    global data_index
    data_index = 0


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def generate_batch(data, bound_index, batch_size, num_skips, skip_window):
    """

    :param data:
    :param bound_index: the index for AlbumInfoParser.BOUND, this is a mark for which context words should not cross.
    :param batch_size:
    :param num_skips:
    :param skip_window:
    :return:
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin

    # skip a bound
    if data[data_index + skip_window] == bound_index:
        data_index += 1

    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    i = 0
    while i < batch_size:
        context_words = [w for w in range(span) if w != skip_window and buffer[w] != bound_index]
        if len(context_words) > 0:
            words_to_use = random.sample(context_words, len(context_words))
            for j, context_word in enumerate(words_to_use):
                batch[i] = buffer[skip_window]
                labels[i, 0] = buffer[context_word]
                i += 1
                if i >= batch_size:
                    break
            if i >= batch_size:
                break
        else:
            # print("word %s stands alone" % buffer[skip_window])
            pass
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
            while buffer[skip_window] == bound_index:
                if data_index == len(data):
                    buffer.extend(data[0:span])
                    data_index = span
                else:
                    buffer.append(data[data_index])
                    data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels
