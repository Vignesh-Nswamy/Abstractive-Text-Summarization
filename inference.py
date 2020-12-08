import tensorflow as tf
import yaml

from data_processor import get_bert_tokenizer
from masks import create_masks
from transformer import Transformer

tf.compat.v1.flags.DEFINE_string('config_path', 'configs/default_config.yml',
                                 'Path to a YAML configuration file')
tf.compat.v1.flags.DEFINE_string('article_path', 'test_article.txt', 'Path to article in txt format')
tf.compat.v1.flags.DEFINE_string('out_path', 'test_article_predic_summary.txt', 'Path to article in txt format')
tf.compat.v1.flags.DEFINE_string('ckpt_dir', 'ckpts/', 'Directory where checkpoints are stored.')
FLAGS = tf.compat.v1.flags.FLAGS

global_config = yaml.load(open(FLAGS.config_path))

tokenizer = get_bert_tokenizer()
global_config['transformer']['encoder']['vocab_size'] = len(tokenizer.vocab)
global_config['transformer']['decoder']['vocab_size'] = len(tokenizer.vocab)


def main(_):
    transformer = Transformer(config=global_config['transformer'])
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=transformer.optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_dir, max_to_keep=10)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    def predict(inp):
        encoder_input = tf.expand_dims(inp, 0)

        decoder_input = tokenizer.convert_tokens_to_ids(['[CLS]'])
        output = tf.expand_dims(decoder_input, 0)
        end_token_id = tokenizer.convert_tokens_to_ids(['[SEP]'])
        for i in range(512):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions = transformer(encoder_input,
                                      output,
                                      enc_padding_mask,
                                      combined_mask,
                                      dec_padding_mask,
                                      training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == end_token_id:
                return tf.squeeze(output, axis=0)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)

    def clean_pred(preds):
        final = list()
        for word_piece in preds:
            if not (word_piece == '[CLS]' or word_piece == '[SEP]'):
                if word_piece.startswith('##'):
                    final[-1] = final[-1] + word_piece[2:]
                elif word_piece.startswith('\'') or word_piece.startswith('.') or word_piece.startswith(','):
                    final[-1] = final[-1] + word_piece
                else:
                    final.append(word_piece)
        return ' '.join(final)

    article = open(FLAGS.article_path).read().replace('\n', ' ')
    model_inp = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(article) + ['[SEP]'])
    if len(model_inp) > 512:
        raise Exception(f'Input file must not have more than 512 token ids. Found {len(model_inp)}')
    predicted_summary_ids = predict(model_inp)
    predicted_summary = clean_pred(tokenizer.convert_ids_to_tokens(predicted_summary_ids.numpy()))

    open(FLAGS.out_path, 'w').write(predicted_summary)

    print(f'Summary saved at {FLAGS.out_path}')


if __name__ == '__main__':
    tf.compat.v1.app.run()
