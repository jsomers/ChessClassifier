import os
import numpy as np
import h5py
import random
import chess
from PIL import Image
import tensorflow as tf
from meta import Meta

tf.app.flags.DEFINE_string('data_dir', './data',
                           'Directory to chessboard folders and write the converted files')
FLAGS = tf.app.flags.FLAGS

class ExampleReader(object):
    # WHITE and black
    def __init__(self, path_to_image_files):
        self._path_to_image_files = path_to_image_files
        self._num_examples = len(self._path_to_image_files)
        self._example_pointer = 0

    @staticmethod
    def _piece_labels(raw_fen):
        labels = { 'k': 1, 'q': 2, 'r': 3, 'b': 4, 'n': 5, 'p': 6, 'K': 7, 'Q': 8, 'R': 9, 'B': 10, 'N': 11, 'P': 12 }
        board = chess.Board(raw_fen.replace("-", "/") + " w - - 0 1")
        arr = []
        for i in range(64):
            pc = board.piece_at(i)
            arr.append(labels[str(pc)] if str(pc) in labels else 0)

        return reversed(arr)

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_and_convert(self):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        path_to_image_file = self._path_to_image_files[self._example_pointer]
        self._example_pointer += 1

        raw_fen = path_to_image_file.split("/")[-1].split(".png")[0]

        pieces = ExampleReader._piece_labels(raw_fen) # [0, 0, 12, 9, 8, 8, 0, ..., 0, 5, 13] # 64-length array with all pieces and 0s for the null piece
        image = np.array(Image.open(path_to_image_file)).tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': ExampleReader._bytes_feature(image),
            'pieces': tf.train.Feature(int64_list=tf.train.Int64List(value=pieces))
        }))
        return example


def convert_to_tfrecords(path_to_dataset_dir_and_digit_struct_mat_file_tuples,
                         path_to_tfrecords_files, choose_writer_callback):
    num_examples = []
    writers = []

    for path_to_tfrecords_file in path_to_tfrecords_files:
        num_examples.append(0)
        writers.append(tf.python_io.TFRecordWriter(path_to_tfrecords_file))

    for path_to_dataset_dir in path_to_dataset_dir_and_digit_struct_mat_file_tuples:
        path_to_image_files = tf.gfile.Glob(os.path.join(path_to_dataset_dir, '*.png'))
        total_files = len(path_to_image_files)
        print '%d files found in %s' % (total_files, path_to_dataset_dir)

        example_reader = ExampleReader(path_to_image_files)
        for index, path_to_image_file in enumerate(path_to_image_files):
            print '(%d/%d) processing %s' % (index + 1, total_files, path_to_image_file)

            example = example_reader.read_and_convert()
            if example is None:
                break

            idx = choose_writer_callback(path_to_tfrecords_files)
            writers[idx].write(example.SerializeToString())
            num_examples[idx] += 1

    for writer in writers:
        writer.close()

    return num_examples


def create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file):
    print 'Saving meta file to %s...' % path_to_tfrecords_meta_file
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(path_to_tfrecords_meta_file)


def main(_):
    path_to_train_dir = os.path.join(FLAGS.data_dir, 'train')
    path_to_test_dir = os.path.join(FLAGS.data_dir, 'test')
    path_to_extra_dir = os.path.join(FLAGS.data_dir, 'extra')

    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_to_test_tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')

    for path_to_file in [path_to_train_tfrecords_file, path_to_val_tfrecords_file, path_to_test_tfrecords_file]:
        assert not os.path.exists(path_to_file), 'The file %s already exists' % path_to_file

    print 'Processing training and validation data...'
    [num_train_examples, num_val_examples] = convert_to_tfrecords([(path_to_train_dir),
                                                                   (path_to_extra_dir)],
                                                                  [path_to_train_tfrecords_file, path_to_val_tfrecords_file],
                                                                  lambda paths: 0 if random.random() > 0.1 else 1)
    print 'Processing test data...'
    [num_test_examples] = convert_to_tfrecords([(path_to_test_dir)],
                                               [path_to_test_tfrecords_file],
                                               lambda paths: 0)

    create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file)

    print 'Done'


if __name__ == '__main__':
    tf.app.run(main=main)
