import tensorflow as tf
import numpy as np

tfrecords_filename = './data/train.tfrecords'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

i = 0
for string_record in record_iterator:
    if i == 10: break
    # Parse the next example
    example = tf.train.Example()
    example.ParseFromString(string_record)

    # Get the features you stored (change to match your tfrecord writing code)
    img_string = (example.features.feature['image']
                                  .bytes_list
                                  .value[0])

    pieces = (example.features.feature['pieces']
                                  .int64_list
                                  .value)

    #print(pieces)
    # Convert to a numpy array (change dtype to the datatype you stored)
    img_1d = np.fromstring(img_string, dtype=np.float32)
    # Print the image shape; does it match your expectations?
    print(np.ndarray.tolist(img_1d))
    i += 1
