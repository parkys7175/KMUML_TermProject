import tensorflow as tf

class CNNet :
    def __init__(self, sess, input_dim, output_dim) :
        self.sess = sess
        self.in_dim = input_dim
        self.out_dim = output_dim


        self.l_rate = 0.01

        self.build_net(self.in_dim, self.out_dim)
        self.build_summary() 

        self.sess.run(tf.global_variables_initializer())


    def build_net(self, in_dim, out_dim) :
        L1_ch = 16
        L2_ch = 32
        L3_ch = 64


        self.input_img = tf.placeholder("float", [None, 64, 64, 3])
        self.Y = tf.placeholder("float", [None, out_dim])


        W1 = tf.get_variable("W1", shape = [3, 3, 3, L1_ch],
                initializer = tf.contrib.layers.xavier_initializer())

        W2 = tf.get_variable("W2", shape = [3, 3, L1_ch, L2_ch],
                initializer = tf.contrib.layers.xavier_initializer())

        W3 = tf.get_variable("W3", shape = [3, 3, L2_ch, L3_ch],
                initializer = tf.contrib.layers.xavier_initializer())


        W4 = tf.get_variable("W4", shape = [8 * 8 * L3_ch, 625],
                initializer = tf.contrib.layers.xavier_initializer())
        
        B4 = tf.get_variable("B4", shape = [625],
                initializer = tf.contrib.layers.xavier_initializer())

        W5 = tf.get_variable("W5", shape = [625, out_dim],
                initializer = tf.contrib.layers.xavier_initializer())

        B5 = tf.get_variable("B5", shape = [out_dim],
                initializer = tf.contrib.layers.xavier_initializer())


        
        L1 = tf.nn.conv2d(self.input_img, W1, strides = [1, 1, 1, 1], padding = 'SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


        L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        
        
        L3 = tf.nn.conv2d(L2, W3, strides = [1, 1, 1, 1], padding = 'SAME')
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


        L4_flat = tf.reshape(L3, [-1, 8 * 8 * L3_ch])
        L4 = tf.nn.relu(tf.matmul(L4_flat, W4) + B4)

        self.logit = tf.matmul(L4, W5) + B5 

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits = self.logit, labels = self.Y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.l_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logit, 1), tf.argmax(self.Y, 1))

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def prediction(self, x_img) :
        return self.sess.run(self.logits, feed_dict = {self.input_img : x_img})

    def get_accuracy(self, x_img, y) :
        return self.sess.run(self.accuracy, feed_dict = {self.input_img : x_img, self.Y : y})

    def train(self, x_img, y) :
        return self.sess.run([self.cost, self.accuracy, self.optimizer], feed_dict = {self.input_img : x_img, self.Y : y})

    def build_summary(self) :
        self.accu_summary = tf.summary.scalar("accuracy", self.accuracy)
        self.cost_summary = tf.summary.scalar("cost", self.cost)
        
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs/traffic_light_classification")

        self.writer.add_graph(self.sess.graph)

    def write_summary(self, img, y) :
        summary = self.sess.run(self.merged_summary, feed_dict = {self.input_img : img, self.Y : y})
        self.writer.add_summary(summary) 



