import tensorflow as tf
from model import Model

tf.app.flags.DEFINE_string('image', None, 'Path to image file')
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           'Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100')
FLAGS = tf.app.flags.FLAGS

def main(_):
    path_to_image_file = FLAGS.image
    path_to_restore_checkpoint_file = FLAGS.restore_checkpoint

    image = tf.image.decode_png(tf.read_file(path_to_image_file), channels=1)
    image = tf.reshape(image, [100, 100, 1])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.multiply(tf.subtract(image, 0.5), 2)
    images = tf.reshape(image, [1, 100, 100, 1])

    pieces_logits = Model.inference(images, drop_rate=0.0)
    pieces_predictions = tf.argmax(pieces_logits, axis=1)
    pieces_predictions_string = tf.reduce_join(tf.as_string(pieces_predictions), axis=1)

    with tf.Session() as sess:
        restorer = tf.train.Saver()
        restorer.restore(sess, path_to_restore_checkpoint_file)

        pieces_predictions_string_val = sess.run([pieces_predictions_string])
        print 'pieces: %s' % pieces_prediction_string_val

if __name__ == '__main__':
    tf.app.run(main=main)
