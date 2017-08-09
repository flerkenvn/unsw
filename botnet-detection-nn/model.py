import tensorflow as tf
from utils import DataLoader


class Model:
    def __init__(self, args):
        print 'Building graph...'
        self.args = args
        self.loader = DataLoader(args.data_dir, args.batch_size, args.input_size)
        self.num_epochs = args.num_epochs

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.input_size * args.input_size])
        self.labels = tf.placeholder(tf.float32, [args.batch_size, args.num_labels])
        self.valid_data = tf.constant(self.loader.valid_dataset)
        self.test_data = tf.constant(self.loader.test_dataset)

        weights = tf.Variable(tf.truncated_normal([args.input_size * args.input_size, args.num_labels]), name='weights')
        biases = tf.Variable(tf.truncated_normal([args.num_labels]), name='weights')

        self.logits = tf.matmul(self.input_data, weights) + biases
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.labels))
        self.optimizer = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss=self.loss)

        self.train_predictions = tf.nn.softmax(self.logits)
        self.test_predictions = tf.nn.softmax(tf.matmul(self.test_data, weights) + biases)
        self.valid_predictions = tf.nn.softmax(tf.matmul(self.valid_data, weights) + biases)

        self.feature = tf.placeholder(tf.float32, [1, args.input_size * args.input_size])
        self.label = tf.nn.softmax(tf.matmul(self.feature, weights) + biases)