import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import import_data
import math
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

class CapsLayer(object):
    def __init__(self, batch_size, epsilon, iter_routing, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.epsilon = epsilon
        self.iter_routing = iter_routing
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type
        self.batch_size = batch_size

    def __call__(self, input, kernel_size=None, stride=None):
        # convolute
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride
            if not self.with_routing:
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len, self.kernel_size, self.stride, padding="VALID", activation_fn=tf.nn.relu)
                capsules = tf.reshape(capsules, (-1, capsules.shape[1].value*capsules.shape[2].value*self.num_outputs, self.vec_len, 1))
                capsules = self.squash(capsules)
                return (capsules)
        # transform primary capsule into senior capsule 
        if self.layer_type == 'FC':
            if self.with_routing:
                self.input = tf.reshape(input, shape=(-1, input.shape[1].value, 1, input.shape[-2].value, 1))
                with tf.variable_scope('routing'):
                    b_IJ = tf.constant(np.zeros([self.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = self.routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)
            return(capsules)
    
    # dynamic routing
    def routing(self, input, b_IJ):
        input_shape = input.get_shape()
        W = tf.get_variable('Weight', shape=(1, input_shape[1], self.num_outputs*self.vec_len, input_shape[3],input_shape[4]), dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
        input = tf.tile(input, [1, 1, self.num_outputs * self.vec_len, 1, 1])
        u_hat = tf.reduce_sum(W * input, axis=3, keepdims=True)
        u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], self.num_outputs, self.vec_len, 1])
        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
        for r_iter in range(self.iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                c_IJ = tf.nn.softmax(b_IJ, axis=2)
                if r_iter == self.iter_routing-1:
                    s_J = tf.multiply(c_IJ, u_hat)
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                    v_J = self.squash(s_J)
                elif r_iter < self.iter_routing-1:
                    s_J = tf.multiply(c_IJ, u_hat_stopped)
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                    v_J = self.squash(s_J)
                    v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                    u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                    b_IJ += u_produce_v
        return(v_J)
    
    # activation function squash
    def squash(self, vector):
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + self.epsilon)
        vec_squashed = scalar_factor * vector
        return(vec_squashed)

# attention mechanism
class SpatialAttention():
    def __init__(self, input_shape):
        self.w_tanh = self.weight_variable((1, input_shape[1].value, input_shape[2].value, input_shape[3].value))
        self.b_tanh = self.bias_variable((1, input_shape[1].value, input_shape[2].value, 1))

    def __call__(self, input):
        # fuse channel information
        fc_first = tf.nn.tanh(tf.reduce_sum(tf.multiply(input, self.w_tanh), axis=3, keep_dims=True) + self.b_tanh)
        # using overlapping pooling to get maximum and mean value of feature map (Dimension is M*M*1)
        spatial_attention_max = tf.nn.max_pool(fc_first, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        spatial_attention_mean = tf.nn.avg_pool(fc_first, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        # concat maximum and mean value as [max,mean] (Dimension is M*M*2)
        spatial_attention_concat = tf.concat([spatial_attention_max, spatial_attention_mean], axis=3)
        # squash the third dimensional as attention weight (Dimension is M*M*1)
        self.attention = tf.contrib.layers.conv2d(spatial_attention_concat, num_outputs=1, kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.sigmoid)
        output = tf.multiply(input, self.attention)
        return output

    # define weight
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape=shape, stddev=0.01)  
        return tf.Variable(initial)
    # define bias
    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


class Capsnet():
    def __init__(self, image_size, channal, num_classes, lambda_val, m_plus, m_minus, epsilon, iter_routing, num_outputs_decode, num_dims_decode, batch_size):
        self.image_size = image_size
        self.lambda_val = lambda_val
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.epsilon = epsilon
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.iter_routing = iter_routing
        self.batch_size = batch_size
        self.num_outputs_layer_conv2d1 = 128
        self.num_outputs_layer_PrimaryCaps = 32
        self.num_dims_layer_PrimaryCaps = 8
        self.num_outputs_decode = num_outputs_decode
        self.num_dims_decode = num_dims_decode
        self.image = tf.placeholder(tf.float32, [self.batch_size, image_size, image_size, channal])
        self.label = tf.placeholder(tf.int64, [self.batch_size, 1])
        label_onehot = tf.one_hot(self.label, depth=num_classes, axis=1, dtype=tf.float32)
        with tf.variable_scope('Conv1_layer'):
            self.conv1 = tf.contrib.layers.conv2d(self.image, num_outputs=self.num_outputs_layer_conv2d1, kernel_size=5, stride=1, padding='VALID')
        attention_shape = self.conv1.get_shape()
        with tf.variable_scope('Soft_attention'):
            self.spatialAtt = SpatialAttention(attention_shape)
            self.attention1 = self.spatialAtt(self.conv1)
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(self.batch_size, self.epsilon, self.iter_routing, num_outputs=self.num_outputs_layer_PrimaryCaps, vec_len=self.num_dims_layer_PrimaryCaps, with_routing=False, layer_type='CONV')
            self.caps1 = primaryCaps(self.attention1, kernel_size=5, stride=1)
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(self.batch_size, self.epsilon, self.iter_routing, num_outputs=self.num_outputs_decode, vec_len=self.num_dims_decode, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(self.caps1)
        with tf.variable_scope('Masking'):
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + self.epsilon)

        # loss function
        max_l = tf.square(tf.maximum(0., self.m_plus - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - self.m_minus))
        max_l_out = tf.reshape(max_l, shape=(-1, self.num_outputs_decode))
        max_r_out = tf.reshape(max_r, shape=(-1, self.num_outputs_decode))
        label_onehot_out = tf.squeeze(label_onehot, axis=2)
        L_c = label_onehot_out * max_l_out + self.lambda_val * (1 - label_onehot_out) * max_r_out
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        self.total_loss = self.margin_loss

        # Adam optimization
        self.train_op = tf.train.AdamOptimizer().minimize(self.total_loss, global_step=self.global_step)


class RunMain():
    def __init__(self,data_class):
        self.data_class = data_class
        sign_handle = import_data.DealSign()
        trainData, self.trainFlag, testData, self.testFlag = sign_handle.readFile(self.data_class)
        self.image_size = 15
        self.channal = 16
        self.num_classes = 5
        self.batch_size = 16 if self.data_class == "person" else 32
        self.lambda_val = 0.5
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.epsilon = 1e-9
        self.iter_routing = 3
        self.num_outputs_decode = 5
        self.num_dims_decode = 16
        self.trainData = self.two_dimension_graph(trainData)
        self.testData = self.two_dimension_graph(testData)
        self.capsnet_model = Capsnet(self.image_size, self.channal, self.num_classes, self.lambda_val, self.m_plus,
                                     self.m_minus, self.epsilon, self.iter_routing, self.num_outputs_decode, self.num_dims_decode, self.batch_size)
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())

    # Two-dimensional discrete features(using Cartesian Product)
    def two_dimension_graph(self, feature):
        feature = feature.reshape((-1, self.image_size-1, self.channal))
        feature = feature.transpose(0, 2, 1)
        feature_graph = np.zeros((feature.shape[0], self.image_size, self.image_size, self.channal))
        for i in range(feature.shape[0]):
            for j in range(feature.shape[1]):
                # Cartesian Product
                single_use = np.hstack((feature[i, j, :], 1)).reshape(-1, self.image_size)
                single_graph = self.sigmoid(0.5*np.sqrt(single_use.T*single_use))
                feature_graph[i, :, :, j] = single_graph
        feature_graph = feature_graph.reshape((feature.shape[0], -1))
        return feature_graph

    def sigmoid(self, single_feature):
        output_s = 1/(1+np.exp(-single_feature))
        return output_s

    def train(self, iteration):
        seed = 3
        m = self.trainData.shape[0]
        i = 0
        num_minibatches = int(m / self.batch_size)
        self.record_epoch_loss = np.zeros(iteration)
        self.record_accuracy = np.zeros(iteration)
        for step in tqdm(range(iteration), total=iteration, ncols=70, leave=False, unit='b'):
            epoch_cost = 0
            seed += 1
            minibatches = self.random_mini_batches(self.batch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                minibatch_X = minibatch_X.reshape((-1, self.image_size, self.image_size, self.channal))
                minibatch_Y = minibatch_Y.reshape((-1, 1))
                output_feed = [self.capsnet_model.train_op,self.capsnet_model.total_loss]
                _, loss = self.sess.run(output_feed, feed_dict={self.capsnet_model.image: minibatch_X, self.capsnet_model.label: minibatch_Y})
                epoch_cost = epoch_cost + loss / num_minibatches
            if step % 1 == 0 or step == iteration-1:
                print('step {}:loss = {:3.10f}'.format(step, epoch_cost))
            self.record_epoch_loss[i] = epoch_cost
            accuracy = self.predict()
            self.record_accuracy[i] = accuracy
            i = i + 1

    def save_best(self, accuracy):
        # save best
        saver = tf.train.Saver(max_to_keep=5)
        saver.save(self.sess,'saver_caps_best/saver' +'_caps_'+str(accuracy)+'/muscle.ckpt')
        # saver correct
        m = self.correct_action.shape[0]
        with open('saver_caps_best/saver'+'_caps_'+str(accuracy)+'/correct.txt', 'w') as f:
            for i in range(m):
                f.write(str(self.correct_action[i]))
                f.write('\n')
            f.write(str(self.correct))
        # saver epoch_loss
        m = self.record_epoch_loss.shape[0]
        with open('saver_caps_best/saver'+'_caps_'+str(accuracy)+'/epoch_loss.txt', 'w') as f:
            for i in range(m):
                f.write(str(self.record_epoch_loss[i]))
                f.write('\n')

        # saver accuracy
        m = self.record_accuracy.shape[0]
        with open('saver_caps_best/saver'+'_caps_'+str(accuracy)+'/accuracy.txt', 'w') as f:
            for i in range(m):
                f.write(str(self.record_accuracy[i]))
                f.write('\n')

    def predict(self):
        m = self.testData.shape[0]
        num = int(m/self.batch_size)
        self.correct = 0
        self.correct_action = np.zeros(self.num_classes)
        for i in range(num):
            datafortest = self.testData[i*self.batch_size:(i+1)*self.batch_size, :].reshape((-1, self.image_size, self.image_size, self.channal))
            answer = self.sess.run(self.capsnet_model.v_length, feed_dict={self.capsnet_model.image: datafortest})
            prediction = np.argmax(np.squeeze(answer), axis=1)
            predict_transform = (prediction.reshape((-1, 1)) == self.testFlag[i*self.batch_size:(i+1)*self.batch_size, :])+0
            self.correct += np.sum(predict_transform)
            for j in range(predict_transform.shape[0]):
                if predict_transform[j]==1:
                    self.correct_action[prediction[j]] += 1
        accuracy = self.correct/(self.batch_size*num)
        print('test accuracy = {:3.6f}'.format(accuracy))
        return accuracy

    # scramble training sample order
    def random_mini_batches(self, mini_batch_size=64, seed=0):
        X = self.trainData
        Y = self.trainFlag
        m = X.shape[0]    # number of training examples
        mini_batches = []
        np.random.seed(seed)
        permutation = list(np.random.permutation(m))  # sorted 0,1,2,..,m-1
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
        mini_batch_X = shuffled_X[0 * mini_batch_size: 0 * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[0 * mini_batch_size: 0 * mini_batch_size + mini_batch_size, :]
        num_complete_minibatches = math.floor(m/mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches

if __name__ == "__main__":
    data_class = "person"
    # data_class = "people"
    iteration = 40 if data_class == "person" else 30
    TDACAPS = RunMain(data_class)
    TDACAPS.train(iteration) 
    accuracy = TDACAPS.predict()
    TDACAPS.save_best(accuracy)
