import tensorflow as tf


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """Perform weighted attention"""

    def __init__(self):
        """Initialize weighted attention"""
        super(ScaledDotProductAttention, self).__init__()

    def call(self, query, key, value, mask=None):
        """Calculate weighted attention
        Args:
            query: tf.tensor, A tensor of shape [batch_size, num_heads, seq_len_q, depth]
            key: tf.tensor, A tensor of shape [batch_size, num_heads, seq_len_k, depth]
            value: tf.tensor, A tensor of shape [batch_size, num_heads, seq_len_v, depth]
            mask: tf.tensor, A tensor with shape broadcastable to [batch_size, seq_len_q, seq_len_k]
        Returns:
            A tensor of shape [batch_size, num_heads, seq_len_q, depth]
        """
        depth_k = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(depth_k)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        return tf.matmul(attention_weights, value)


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-headed attention layer"""

    def __init__(self, d_model, num_heads):
        """Initialize Multi-headed attention.
        Args:
          d_model: int, output dim of hidden layer.
          num_heads: int, number of heads to repeat the same attention structure.
        """

        assert d_model % num_heads == 0
        super(MultiHeadAttention, self).__init__()

        self.d_model, self.h = d_model, num_heads
        self.depth = self.d_model // self.h

        self.query_dense_layer = tf.keras.layers.Dense(d_model,
                                                       use_bias=False,
                                                       name='query')
        self.key_dense_layer = tf.keras.layers.Dense(d_model,
                                                     use_bias=False,
                                                     name='key')
        self.value_dense_layer = tf.keras.layers.Dense(d_model,
                                                       use_bias=False,
                                                       name='value')
        self.scaled_attn_layer = ScaledDotProductAttention()

        self.output_dense = tf.keras.layers.Dense(d_model,
                                                  use_bias=False,
                                                  name='value')

    def split_heads(self, x, batch_size):
        """Split last dimension of input tensor into (num_heads, depth)
        Args:
            x: tf.tensor, input tensor of shape [batch_size, seq_len, d_model]
            batch_size: int, Batch size
        Returns:
            A tensor of shape [batch_size, num_heads, seq_len, depth]
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, memory=None, attention_mask=None):
        """Apply attention to query and memory inputs
        Args:
            query: tf.tensor, tensor of shape [batch_size, seq_len_q, d_model] (Self-attention)
            memory: tf.tensor, tensor of shape [batch_size, seq_len_q, d_model] (Encoder-attention)
            attention_mask: tf.tensor, tensor of shape [batch_size, seq_len_q, d_model]
        Returns:
            A tensor of shape [batch_size, seq_len_q, d_model]
        """
        batch_size = tf.shape(query)[0]

        if memory is None:
            memory = query

        q = self.query_dense_layer(query)
        k = self.key_dense_layer(memory)
        v = self.value_dense_layer(memory)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention = self.scaled_attn_layer(q, k, v, attention_mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        scaled_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        return self.output_dense(scaled_attention)
