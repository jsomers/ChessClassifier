import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from model import Model

def process(sess, filename, image_batch_, pieces_logits, pieces_batch_, tfloss):
    img = Image.open(filename)
    npimg = np.array(img.convert("L")).reshape(1, 100, 100, 1).astype(np.float32)
    img.close()
    npimg = -1 + 2 * (npimg - npimg.min()) / (npimg.max() - npimg.min())
    loss, pieces = sess.run([tfloss, pieces_logits], feed_dict={image_batch_: npimg, pieces_batch_: np.zeros((1, 64))})
    print(loss)

def main(argv):
    with tf.variable_scope("input"):
        image_batch_ = tf.placeholder(name="images", shape=(None, 100, 100, 1), dtype=tf.float32)

    with tf.variable_scope("targets"):
        pieces_batch_ = tf.placeholder(name="pieces", shape=(None, 64), dtype=tf.int32)
    pieces_logits = Model.inference(image_batch_, drop_rate=0.2)
    loss = Model.loss(pieces_logits, pieces_batch_)
    with tf.Session() as sess:
        # Tensor("shuffle_batch:0", shape=(32, 100, 100, 1), dtype=float32)       
        writer = tf.summary.FileWriter("test", sess.graph)
        saver = tf.train.Saver()
        saver.restore(sess, "logs/train/latest.ckpt")
        saver.save(sess, "test/test.ckpt")
        for filename in argv:
            process(sess, filename, image_batch_, pieces_logits, pieces_batch_, loss)
        writer.close()

if __name__ == "__main__":
    main(sys.argv[1:])
