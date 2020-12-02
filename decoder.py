import tensorflow as tf
from attention import MultiHeadAttention
from positional_encoding import PositionEncoder


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()

        d_model = config['d_model']
        num_heads = config['num_attention_heads']
        num_fc_units = config['intermediate_size']
        self_attn_dropout_rate = config['attention_dropout_rate']
        encoder_attn_dropout_rate = config['attention_dropout_rate']
        fc_dropout_rate = config['dropout_rate']

        self.self_attention_layer = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention_layer = MultiHeadAttention(d_model, num_heads)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(num_fc_units, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.self_attn_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.enc_attn_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.fc_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.self_attn_dropout = tf.keras.layers.Dropout(self_attn_dropout_rate)
        self.enc_attn_dropout = tf.keras.layers.Dropout(encoder_attn_dropout_rate)
        self.fc_dropout = tf.keras.layers.Dropout(fc_dropout_rate)

    def call(self, inp, enc_output, look_ahead_mask, padding_mask, training):
        self_attention = self.self_attention_layer(inp, attention_mask=look_ahead_mask)
        self_attention = self.self_attn_dropout(self_attention, training=training)
        self_attention_out = self.self_attn_layernorm(self_attention + inp)

        encoder_attention = self.encoder_attention_layer(self_attention_out, memory=enc_output, attention_mask=padding_mask)
        encoder_attention = self.enc_attn_dropout(encoder_attention, training=training)
        encoder_attention_out = self.enc_attn_layernorm(encoder_attention + self_attention_out)

        fc_output = self.ffn(encoder_attention_out)
        fc_output = self.fc_dropout(fc_output, training=training)
        dec_layer_out = self.fc_layernorm(fc_output + encoder_attention_out)

        return dec_layer_out


class TransformerDecoder(tf.keras.layers.Layer):
    """Transformer Decoder"""

    def __init__(self, config):
        """Decoder initialization with given config
        Args:
            config: Python dict containing specifications of decoder
        Returns:
            A decoder layer
        """
        super(TransformerDecoder, self).__init__()

        self.config = config

        self.embedder = tf.keras.layers.Embedding(input_dim=self.config['vocab_size'],
                                                  output_dim=self.config['d_model'])
        self.position_embedder = PositionEncoder(max_seq_len=self.config['max_sequence_length'],
                                                 d_model=self.config['d_model'])

        self.decoder_stack = [
            DecoderBlock(config) for _ in range(self.config['num_layers'])
        ]

        self.dropout = tf.keras.layers.Dropout(self.config['dropout_rate'])

    def call(self, inp, enc_output, look_ahead_mask, padding_mask, training=False):
        inp = self.embedder(inp)
        inp *= tf.math.sqrt(tf.cast(self.config['d_model'], tf.float32))
        inp = self.position_embedder(inp)
        inp = self.dropout(inp, training=training)

        for decoder_block in self.decoder_stack:
            inp = decoder_block(inp, enc_output, look_ahead_mask, padding_mask, training)

        return inp

