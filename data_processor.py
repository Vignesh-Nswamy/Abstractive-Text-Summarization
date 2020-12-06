import os
import tensorflow as tf
import tokenization


class DataProcessor:
    def __init__(self, config, tokenizer=None):
        self.config = config

        if tokenizer is None:
            gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
            self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
                                                             do_lower_case=True)
        else:
            self.bert_tokenizer = tokenizer

        self.target_vocab_size = len(self.bert_tokenizer.vocab)

    def get_dataset(self, split='train'):
        feature_description = {
            'article': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'summary': tf.io.FixedLenFeature([], tf.string, default_value='')
        }
        
        records_path = tf.data.Dataset.list_files(self.config['train_path']) if split == 'train' \
            else self.config['val_path'] if split == 'val' \
            else self.config['test_path']
        max_len = self.config.get('max_len', 512)
        
        def _parse_function(example_proto):
            # Parse the input `tf.train.Example` proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, feature_description)

        def encode(article, summary):
            article_tokens = self.bert_tokenizer.convert_tokens_to_ids(
                ['[CLS]'] + self.bert_tokenizer.tokenize(article.numpy()) + ['[SEP]'])
            summary_tokens = self.bert_tokenizer.convert_tokens_to_ids(
                ['[CLS]'] + self.bert_tokenizer.tokenize(summary.numpy()) + ['[SEP]'])

            return article_tokens, summary_tokens

        def tf_encode(example):
            result_ar, result_sum = tf.py_function(encode, [example['article'], example['summary']],
                                                   [tf.int64, tf.int64])
            result_ar.set_shape([None])
            result_sum.set_shape([None])

            return result_ar, result_sum

        def filter_max_length(x, y):
            return tf.logical_and(tf.size(x) <= max_len,
                                  tf.size(y) <= max_len)

        dataset = tf.data.TFRecordDataset(records_path).map(_parse_function)
        dataset = dataset.map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.filter(filter_max_length)
        dataset = dataset.cache()
        dataset = dataset.shuffle(self.config['shuffle_buffer_size']) if split == 'train' else dataset
        dataset = dataset.padded_batch(self.config['batch_size'])
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
