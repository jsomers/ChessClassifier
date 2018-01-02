import tensorflow as tf
from meta import Meta
from matplotlib.patches import Rectangle
import matplotlib.image as img
#import matplotlib.pyplot as plt

filename = 'data/test.tfrecords'
filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
  serialized_example,
  features={
      'image': tf.FixedLenFeature([], tf.string),
      'pieces': tf.FixedLenFeature([64], tf.int64)
  })

image = tf.decode_raw(features['image'], tf.uint8)
# FIXME: or maybe i can just add some preprocessing steps here and it'll be fine
image = tf.reshape(image, [100, 100, 4]) # FIXME: interesting! wants a 4th channel. should only have 3. Maybe we need to kill the alpha channel on the images?
pieces = features['pieces']

sess = tf.InteractiveSession()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

(image_val, pieces_val) = sess.run([image, pieces])

print 'pieces: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d' % (
    pieces_val[0], pieces_val[1], pieces_val[2], pieces_val[3], pieces_val[4], pieces_val[5], pieces_val[6], pieces_val[7], pieces_val[8], pieces_val[9], pieces_val[10], pieces_val[11], pieces_val[12], pieces_val[13], pieces_val[14], pieces_val[15], pieces_val[16], pieces_val[17], pieces_val[18], pieces_val[19], pieces_val[20], pieces_val[21], pieces_val[22], pieces_val[23], pieces_val[24], pieces_val[25], pieces_val[26], pieces_val[27], pieces_val[28], pieces_val[29], pieces_val[30], pieces_val[31], pieces_val[32], pieces_val[33], pieces_val[34], pieces_val[35], pieces_val[36], pieces_val[37], pieces_val[38], pieces_val[39], pieces_val[40], pieces_val[41], pieces_val[42], pieces_val[43], pieces_val[44], pieces_val[45], pieces_val[46], pieces_val[47], pieces_val[48], pieces_val[49], pieces_val[50], pieces_val[51], pieces_val[52], pieces_val[53], pieces_val[54], pieces_val[55], pieces_val[56], pieces_val[57], pieces_val[58], pieces_val[59], pieces_val[60], pieces_val[61], pieces_val[62], pieces_val[63])

img.imsave("test.png", image_val)
