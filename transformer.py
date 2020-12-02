from abc import ABC

import tensorflow as tf
from encoder import TransformerEncoder
from decoder import TransformerDecoder

from scheduler import CustomSchedule


class Transformer(tf.keras.Model, ABC):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        lr_scheduler = CustomSchedule(self.config['decoder']['d_model'])
        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.encoder = TransformerEncoder(self.config['encoder'])
        self.decoder = TransformerDecoder(self.config['decoder'])

        self.final_layer = tf.keras.layers.Dense(self.config['decoder']['vocab_size'])

    def call(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask, training=False):
        enc_output = self.encoder(inp, enc_padding_mask, training=training)

        dec_output = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask, training=training)

        final_output = self.final_layer(dec_output)

        return final_output

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    @staticmethod
    def accuracy_function(real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
