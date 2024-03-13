import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam

class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size

    def build(self, input_shape):
        # Trend part
        self.wb = self.add_weight(name='wb', shape=(input_shape[1], 1), initializer='uniform', trainable=True)
        self.bb = self.add_weight(name='bb', shape=(input_shape[1], 1), initializer='uniform', trainable=True)
        # Periodic part
        self.wa = self.add_weight(name='wa', shape=(input_shape[2], self.k), initializer='uniform', trainable=True)
        self.ba = self.add_weight(name='ba', shape=(input_shape[1], self.k), initializer='uniform', trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, x):
        bias = self.wb * x + self.bb
        dp = tf.matmul(x, self.wa) + self.ba
        wgts = tf.math.sin(dp)
        return tf.concat([bias, wgts], -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] + self.k)

class SingleAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.key = Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.value = Dense(self.d_v, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])
        v = self.value(inputs[2])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x / tf.math.sqrt(tf.cast(tf.shape(x)[0], tf.float32)), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        output = tf.matmul(attn_weights, v)
        return output

class MultiAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))
        self.linear = Dense(input_shape[0][-1], input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.3, activation='relu'):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.activation = activation
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Dense(self.ff_dim, activation=self.activation, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.ff_conv1D_2 = Dense(input_shape[0][-1], kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(attn_layer + ff_layer)
        return ff_layer

class Transformer(tf.keras.Model):
    def __init__(self, time_embedding=True, n_layers=3, d_k=64, d_v=64, n_heads=4, ff_dim=64, feature_size=128, seq_len=50, out_dim=1, dropout=0.3):
        super(Transformer, self).__init__()
        self.time_embedding = time_embedding
        self.n_layers = n_layers
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.feature_size = feature_size
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.dropout_rate = dropout

        if self.time_embedding:
            self.time_embedding_layer = Time2Vec(kernel_size=1)

        self.encoders = [TransformerEncoder(self.d_k, self.d_v, self.n_heads, self.ff_dim, self.dropout_rate) for _ in range(self.n_layers)]
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dropout = Dropout(self.dropout_rate)
        self.dense_out = Dense(self.out_dim, activation='linear')

    def call(self, inputs):
        if self.time_embedding:
            x = self.time_embedding_layer(inputs)
        else:
            x = inputs

        for i in range(self.n_layers):
            x = self.encoders[i]((x, x, x))

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense_out(x)
        return x

    def model(self):
        return tf.keras.Model(inputs=self.inputs, outputs=self.call(self.inputs))

