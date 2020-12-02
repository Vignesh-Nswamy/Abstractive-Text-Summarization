import tensorflow as tf


class PositionEncoder(tf.keras.layers.Layer):
    """Adds positional information to input sequence"""

    def __init__(self, max_seq_len=512, d_model=768):
        """Initialize PositionEncoder
        Args:
            max_seq_len: int, Maximum expected length of each training sequence
            d_model: int, Number of hidden units
        """
        super(PositionEncoder, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pos_encoding = self.positional_encoding()

    @staticmethod
    def get_angles(position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self):
        angle_rads = self.get_angles(position=tf.range(self.max_seq_len, dtype=tf.float32)[:, tf.newaxis],
                                     i=tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :],
                                     d_model=self.d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
