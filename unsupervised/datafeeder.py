import tensorflow as tf
class DataFeeder:
    def __init__(self, filenames, is_training=True, max_test = 12800):
        self.filenames = filenames
        self.is_training = is_training
        self.max_test = max_test
        with open(self.filenames[0], 'r') as f:
            line = f.readline().rstrip()
            tokens = line.rstrip().split('\t')
            id = tokens[0]
            self.dan_dim = tokens[1].count(',')+1
            self.cdssm_dim = tokens[2].count(',')+1
            self.content_dim = tokens[3].count(',')+1
        print ("input dims:", self.dan_dim, self.cdssm_dim, self.content_dim)

    def read_feature(self):
        for filename in self.filenames:
            with open(filename, 'r') as f:
                for i, line in enumerate(f):
                    tokens = line.rstrip().split('\t')
                    key = tokens[0]
                    dan_str = tokens[1]
                    cdssm_str = tokens[2]
                    content_str = tokens[3]
                    dan = [float(t) for t in dan_str.split(',')]
                    cdssm = [float(t) for t in cdssm_str.split(',')]
                    content = [float(t) for t in content_str.split(',')]
                    yield (key, dan, cdssm, content)
                    if not self.is_training and i == self.max_test:
                        break

    def tensor_data_generator(self, batch_size, repeats):
        dataset = tf.data.Dataset.from_generator(self.read_feature, (tf.string, tf.float32, tf.float32, tf.float32), (tf.TensorShape([]), tf.TensorShape([self.dan_dim]), tf.TensorShape([self.cdssm_dim]), tf.TensorShape([self.content_dim])))
        if self.is_training:
            dataset = dataset.shuffle(100)
        dataset = dataset.repeat(repeats)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

if __name__ == '__main__':
    data_feeder = DataFeeder('train_random_video.txt')
    one_query = data_feeder.tensor_data_generator(32, 100)
    with tf.Session() as sess:
        try:
            data = sess.run(one_query)
            print (data)
        except tf.errors.OutOfRangeError:
            print ('end!')
