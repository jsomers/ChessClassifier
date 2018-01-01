import tensorflow as tf


class Model(object):

    @staticmethod
    def inference(x, drop_rate):
        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(x, filters=48, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden1 = dropout

        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden2 = dropout

        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden3 = dropout

        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=160, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden4 = dropout

        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden5 = dropout

        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden6 = dropout

        with tf.variable_scope('hidden7'):
            conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden7 = dropout

        with tf.variable_scope('hidden8'):
            conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden8 = dropout

        flatten = tf.reshape(hidden8, [-1, 7 * 7 * 192])

        with tf.variable_scope('hidden9'):
            dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
            hidden9 = dense

        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(hidden9, units=3072, activation=tf.nn.relu)
            hidden10 = dense

        with tf.variable_scope('piece1'):
            dense = tf.layers.dense(hidden10, units=13) # FIXME: should this be 11 (unchanged from original), 13 (for 13 choices), or 14 (for n choices + 1)?
            piece1 = dense

        with tf.variable_scope('piece2'):
            dense = tf.layers.dense(hidden10, units=13)
            piece2 = dense
              
        with tf.variable_scope('piece3'):
            dense = tf.layers.dense(hidden10, units=13)
            piece3 = dense
            
        with tf.variable_scope('piece4'):
            dense = tf.layers.dense(hidden10, units=13)
            piece4 = dense
            
        with tf.variable_scope('piece5'):
            dense = tf.layers.dense(hidden10, units=13)
            piece5 = dense
            
        with tf.variable_scope('piece6'):
            dense = tf.layers.dense(hidden10, units=13)
            piece6 = dense
            
        with tf.variable_scope('piece7'):
            dense = tf.layers.dense(hidden10, units=13)
            piece7 = dense
            
        with tf.variable_scope('piece8'):
            dense = tf.layers.dense(hidden10, units=13)
            piece8 = dense
            
        with tf.variable_scope('piece9'):
            dense = tf.layers.dense(hidden10, units=13)
            piece9 = dense
            
        with tf.variable_scope('piece10'):
            dense = tf.layers.dense(hidden10, units=13)
            piece10 = dense
            
        with tf.variable_scope('piece11'):
            dense = tf.layers.dense(hidden10, units=13)
            piece11 = dense
            
        with tf.variable_scope('piece12'):
            dense = tf.layers.dense(hidden10, units=13)
            piece12 = dense
            
        with tf.variable_scope('piece13'):
            dense = tf.layers.dense(hidden10, units=13)
            piece13 = dense
            
        with tf.variable_scope('piece14'):
            dense = tf.layers.dense(hidden10, units=13)
            piece14 = dense
            
        with tf.variable_scope('piece15'):
            dense = tf.layers.dense(hidden10, units=13)
            piece15 = dense
            
        with tf.variable_scope('piece16'):
            dense = tf.layers.dense(hidden10, units=13)
            piece16 = dense
            
        with tf.variable_scope('piece17'):
            dense = tf.layers.dense(hidden10, units=13)
            piece17 = dense
            
        with tf.variable_scope('piece18'):
            dense = tf.layers.dense(hidden10, units=13)
            piece18 = dense
            
        with tf.variable_scope('piece19'):
            dense = tf.layers.dense(hidden10, units=13)
            piece19 = dense
            
        with tf.variable_scope('piece20'):
            dense = tf.layers.dense(hidden10, units=13)
            piece20 = dense
            
        with tf.variable_scope('piece21'):
            dense = tf.layers.dense(hidden10, units=13)
            piece21 = dense
            
        with tf.variable_scope('piece22'):
            dense = tf.layers.dense(hidden10, units=13)
            piece22 = dense
            
        with tf.variable_scope('piece23'):
            dense = tf.layers.dense(hidden10, units=13)
            piece23 = dense
            
        with tf.variable_scope('piece24'):
            dense = tf.layers.dense(hidden10, units=13)
            piece24 = dense
            
        with tf.variable_scope('piece25'):
            dense = tf.layers.dense(hidden10, units=13)
            piece25 = dense
            
        with tf.variable_scope('piece26'):
            dense = tf.layers.dense(hidden10, units=13)
            piece26 = dense
            
        with tf.variable_scope('piece27'):
            dense = tf.layers.dense(hidden10, units=13)
            piece27 = dense
            
        with tf.variable_scope('piece28'):
            dense = tf.layers.dense(hidden10, units=13)
            piece28 = dense
            
        with tf.variable_scope('piece29'):
            dense = tf.layers.dense(hidden10, units=13)
            piece29 = dense
            
        with tf.variable_scope('piece30'):
            dense = tf.layers.dense(hidden10, units=13)
            piece30 = dense
            
        with tf.variable_scope('piece31'):
            dense = tf.layers.dense(hidden10, units=13)
            piece31 = dense
            
        with tf.variable_scope('piece32'):
            dense = tf.layers.dense(hidden10, units=13)
            piece32 = dense
            
        with tf.variable_scope('piece33'):
            dense = tf.layers.dense(hidden10, units=13)
            piece33 = dense
            
        with tf.variable_scope('piece34'):
            dense = tf.layers.dense(hidden10, units=13)
            piece34 = dense
            
        with tf.variable_scope('piece35'):
            dense = tf.layers.dense(hidden10, units=13)
            piece35 = dense
            
        with tf.variable_scope('piece36'):
            dense = tf.layers.dense(hidden10, units=13)
            piece36 = dense
            
        with tf.variable_scope('piece37'):
            dense = tf.layers.dense(hidden10, units=13)
            piece37 = dense
            
        with tf.variable_scope('piece38'):
            dense = tf.layers.dense(hidden10, units=13)
            piece38 = dense
            
        with tf.variable_scope('piece39'):
            dense = tf.layers.dense(hidden10, units=13)
            piece39 = dense
            
        with tf.variable_scope('piece40'):
            dense = tf.layers.dense(hidden10, units=13)
            piece40 = dense
            
        with tf.variable_scope('piece41'):
            dense = tf.layers.dense(hidden10, units=13)
            piece41 = dense
            
        with tf.variable_scope('piece42'):
            dense = tf.layers.dense(hidden10, units=13)
            piece42 = dense
            
        with tf.variable_scope('piece43'):
            dense = tf.layers.dense(hidden10, units=13)
            piece43 = dense
            
        with tf.variable_scope('piece44'):
            dense = tf.layers.dense(hidden10, units=13)
            piece44 = dense
            
        with tf.variable_scope('piece45'):
            dense = tf.layers.dense(hidden10, units=13)
            piece45 = dense
            
        with tf.variable_scope('piece46'):
            dense = tf.layers.dense(hidden10, units=13)
            piece46 = dense
            
        with tf.variable_scope('piece47'):
            dense = tf.layers.dense(hidden10, units=13)
            piece47 = dense
            
        with tf.variable_scope('piece48'):
            dense = tf.layers.dense(hidden10, units=13)
            piece48 = dense
            
        with tf.variable_scope('piece49'):
            dense = tf.layers.dense(hidden10, units=13)
            piece49 = dense
            
        with tf.variable_scope('piece50'):
            dense = tf.layers.dense(hidden10, units=13)
            piece50 = dense
            
        with tf.variable_scope('piece51'):
            dense = tf.layers.dense(hidden10, units=13)
            piece51 = dense
            
        with tf.variable_scope('piece52'):
            dense = tf.layers.dense(hidden10, units=13)
            piece52 = dense
            
        with tf.variable_scope('piece53'):
            dense = tf.layers.dense(hidden10, units=13)
            piece53 = dense
            
        with tf.variable_scope('piece54'):
            dense = tf.layers.dense(hidden10, units=13)
            piece54 = dense
            
        with tf.variable_scope('piece55'):
            dense = tf.layers.dense(hidden10, units=13)
            piece55 = dense
            
        with tf.variable_scope('piece56'):
            dense = tf.layers.dense(hidden10, units=13)
            piece56 = dense
            
        with tf.variable_scope('piece57'):
            dense = tf.layers.dense(hidden10, units=13)
            piece57 = dense
            
        with tf.variable_scope('piece58'):
            dense = tf.layers.dense(hidden10, units=13)
            piece58 = dense
            
        with tf.variable_scope('piece59'):
            dense = tf.layers.dense(hidden10, units=13)
            piece59 = dense
            
        with tf.variable_scope('piece60'):
            dense = tf.layers.dense(hidden10, units=13)
            piece60 = dense
            
        with tf.variable_scope('piece61'):
            dense = tf.layers.dense(hidden10, units=13)
            piece61 = dense
            
        with tf.variable_scope('piece62'):
            dense = tf.layers.dense(hidden10, units=13)
            piece62 = dense
            
        with tf.variable_scope('piece63'):
            dense = tf.layers.dense(hidden10, units=13)
            piece63 = dense
            
        with tf.variable_scope('piece64'):
            dense = tf.layers.dense(hidden10, units=13)
            piece64 = dense

        pieces_logits = tf.stack([piece1, piece2, piece3, piece4, piece5, piece6, piece7, piece8, piece9, piece10, piece11, piece12, piece13, piece14, piece15, piece16, piece17, piece18, piece19, piece20, piece21, piece22, piece23, piece24, piece25, piece26, piece27, piece28, piece29, piece30, piece31, piece32, piece33, piece34, piece35, piece36, piece37, piece38, piece39, piece40, piece41, piece42, piece43, piece44, piece45, piece46, piece47, piece48, piece49, piece50, piece51, piece52, piece53, piece54, piece55, piece56, piece57, piece58, piece59, piece60, piece61, piece62, piece63, piece64], axis=1) # TODO: 64 of these
        #import pdb; pdb.set_trace()
        return pieces_logits

    @staticmethod
    def loss(pieces_logits, pieces_labels):
        piece1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 0], logits=pieces_logits[:, 0, :]))
        piece2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 1], logits=pieces_logits[:, 1, :]))
        piece3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 2], logits=pieces_logits[:, 2, :]))
        piece4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 3], logits=pieces_logits[:, 3, :]))
        piece5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 4], logits=pieces_logits[:, 4, :]))
        piece6_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 5], logits=pieces_logits[:, 5, :]))
        piece7_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 6], logits=pieces_logits[:, 6, :]))
        piece8_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 7], logits=pieces_logits[:, 7, :]))
        piece9_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 8], logits=pieces_logits[:, 8, :]))
        piece10_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 9], logits=pieces_logits[:, 9, :]))
        piece11_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 10], logits=pieces_logits[:, 10, :]))
        piece12_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 11], logits=pieces_logits[:, 11, :]))
        piece13_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 12], logits=pieces_logits[:, 12, :]))
        piece14_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 13], logits=pieces_logits[:, 13, :]))
        piece15_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 14], logits=pieces_logits[:, 14, :]))
        piece16_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 15], logits=pieces_logits[:, 15, :]))
        piece17_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 16], logits=pieces_logits[:, 16, :]))
        piece18_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 17], logits=pieces_logits[:, 17, :]))
        piece19_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 18], logits=pieces_logits[:, 18, :]))
        piece20_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 19], logits=pieces_logits[:, 19, :]))
        piece21_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 20], logits=pieces_logits[:, 20, :]))
        piece22_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 21], logits=pieces_logits[:, 21, :]))
        piece23_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 22], logits=pieces_logits[:, 22, :]))
        piece24_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 23], logits=pieces_logits[:, 23, :]))
        piece25_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 24], logits=pieces_logits[:, 24, :]))
        piece26_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 25], logits=pieces_logits[:, 25, :]))
        piece27_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 26], logits=pieces_logits[:, 26, :]))
        piece28_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 27], logits=pieces_logits[:, 27, :]))
        piece29_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 28], logits=pieces_logits[:, 28, :]))
        piece30_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 29], logits=pieces_logits[:, 29, :]))
        piece31_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 30], logits=pieces_logits[:, 30, :]))
        piece32_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 31], logits=pieces_logits[:, 31, :]))
        piece33_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 32], logits=pieces_logits[:, 32, :]))
        piece34_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 33], logits=pieces_logits[:, 33, :]))
        piece35_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 34], logits=pieces_logits[:, 34, :]))
        piece36_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 35], logits=pieces_logits[:, 35, :]))
        piece37_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 36], logits=pieces_logits[:, 36, :]))
        piece38_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 37], logits=pieces_logits[:, 37, :]))
        piece39_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 38], logits=pieces_logits[:, 38, :]))
        piece40_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 39], logits=pieces_logits[:, 39, :]))
        piece41_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 40], logits=pieces_logits[:, 40, :]))
        piece42_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 41], logits=pieces_logits[:, 41, :]))
        piece43_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 42], logits=pieces_logits[:, 42, :]))
        piece44_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 43], logits=pieces_logits[:, 43, :]))
        piece45_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 44], logits=pieces_logits[:, 44, :]))
        piece46_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 45], logits=pieces_logits[:, 45, :]))
        piece47_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 46], logits=pieces_logits[:, 46, :]))
        piece48_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 47], logits=pieces_logits[:, 47, :]))
        piece49_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 48], logits=pieces_logits[:, 48, :]))
        piece50_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 49], logits=pieces_logits[:, 49, :]))
        piece51_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 50], logits=pieces_logits[:, 50, :]))
        piece52_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 51], logits=pieces_logits[:, 51, :]))
        piece53_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 52], logits=pieces_logits[:, 52, :]))
        piece54_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 53], logits=pieces_logits[:, 53, :]))
        piece55_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 54], logits=pieces_logits[:, 54, :]))
        piece56_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 55], logits=pieces_logits[:, 55, :]))
        piece57_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 56], logits=pieces_logits[:, 56, :]))
        piece58_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 57], logits=pieces_logits[:, 57, :]))
        piece59_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 58], logits=pieces_logits[:, 58, :]))
        piece60_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 59], logits=pieces_logits[:, 59, :]))
        piece61_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 60], logits=pieces_logits[:, 60, :]))
        piece62_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 61], logits=pieces_logits[:, 61, :]))
        piece63_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 62], logits=pieces_logits[:, 62, :]))
        piece64_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=pieces_labels[:, 63], logits=pieces_logits[:, 63, :]))
        loss = piece1_cross_entropy + piece2_cross_entropy + piece3_cross_entropy + piece4_cross_entropy + piece5_cross_entropy + piece6_cross_entropy + piece7_cross_entropy + piece8_cross_entropy + piece9_cross_entropy + piece10_cross_entropy + piece11_cross_entropy + piece12_cross_entropy + piece13_cross_entropy + piece14_cross_entropy + piece15_cross_entropy + piece16_cross_entropy + piece17_cross_entropy + piece18_cross_entropy + piece19_cross_entropy + piece20_cross_entropy + piece21_cross_entropy + piece22_cross_entropy + piece23_cross_entropy + piece24_cross_entropy + piece25_cross_entropy + piece26_cross_entropy + piece27_cross_entropy + piece28_cross_entropy + piece29_cross_entropy + piece30_cross_entropy + piece31_cross_entropy + piece32_cross_entropy + piece33_cross_entropy + piece34_cross_entropy + piece35_cross_entropy + piece36_cross_entropy + piece37_cross_entropy + piece38_cross_entropy + piece39_cross_entropy + piece40_cross_entropy + piece41_cross_entropy + piece42_cross_entropy + piece43_cross_entropy + piece44_cross_entropy + piece45_cross_entropy + piece46_cross_entropy + piece47_cross_entropy + piece48_cross_entropy + piece49_cross_entropy + piece50_cross_entropy + piece51_cross_entropy + piece52_cross_entropy + piece53_cross_entropy + piece54_cross_entropy + piece55_cross_entropy + piece56_cross_entropy + piece57_cross_entropy + piece58_cross_entropy + piece59_cross_entropy + piece60_cross_entropy + piece61_cross_entropy + piece62_cross_entropy + piece63_cross_entropy + piece64_cross_entropy
        return loss
