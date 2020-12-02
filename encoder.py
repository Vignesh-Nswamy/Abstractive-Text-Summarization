import tensorflow as tf
import tensorflow_hub as hub
from attention import MultiHeadAttention
from positional_encoding import PositionEncoder


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()

        d_model = config['d_model']
        num_heads = config['num_attention_heads']
        num_fc_units = config['intermediate_size']
        self_attn_dropout_rate = config['attention_dropout_rate']
        fc_dropout_rate = config['dropout_rate']

        self.self_attention_layer = MultiHeadAttention(d_model, num_heads)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(num_fc_units, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.self_attn_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.fc_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.self_attn_dropout = tf.keras.layers.Dropout(self_attn_dropout_rate)
        self.fc_dropout = tf.keras.layers.Dropout(fc_dropout_rate)

    def call(self, inp, mask=None, training=False):
        self_attention = self.self_attention_layer(inp, attention_mask=mask)
        self_attention = self.self_attn_dropout(self_attention, training=training)
        self_attention_out = self.self_attn_layernorm(inp + self_attention)

        fc_output = self.ffn(self_attention_out)
        fc_output = self.fc_dropout(fc_output, training=training)
        enc_layer_out = self.fc_layernorm(self_attention_out + fc_output)

        return enc_layer_out


class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer Encoder"""

    def __init__(self, config):
        """Encoder initialization with given config
        Args:
            config: Python dict containing specifications of encoder
        Returns:
            An encoder layer
        """
        super(TransformerEncoder, self).__init__()

        self.config = config

        if self.config['is_bert']:
            print(f'Downloading bert from {self.config["bert_hub_url"]}')
            self.bert_encoder = hub.KerasLayer(self.config['bert_hub_url'],
                                               trainable=self.config['finetune_bert'])
            print('Downloaded and initialized bert')
        else:
            self.embedder = tf.keras.layers.Embedding(input_dim=self.config['vocab_size'],
                                                      output_dim=self.config['d_model'])
            self.position_embedder = PositionEncoder(max_seq_len=self.config['max_sequence_length'],
                                                     d_model=self.config['d_model'])

            self.encoder_stack = [
                EncoderBlock(config) for _ in range(self.config['num_layers'])
            ]

            self.dropout = tf.keras.layers.Dropout(self.config['dropout_rate'])

    def call(self, inp, mask=None, training=False):
        if self.config['is_bert']:
            training = training and self.config['finetune_bert']
            return self.bert_encoder(dict(input_word_ids=tf.cast(inp, tf.int32),
                                          input_mask=tf.cast(tf.math.logical_not(tf.math.equal(inp, 0)), tf.int32),
                                          input_type_ids=tf.cast(tf.math.equal(inp, 0), tf.int32)), training=training)['sequence_output']
        else:
            inp = self.embedder(inp)
            inp *= tf.math.sqrt(tf.cast(self.config['d_model'], tf.float32))
            inp = self.position_embedder(inp)
            inp = self.dropout(inp, training=training)

            for encoder_block in self.encoder_stack:
                inp = encoder_block(inp, mask, training)

            return inp