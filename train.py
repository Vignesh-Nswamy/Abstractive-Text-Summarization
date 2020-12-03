import tensorflow as tf
from transformer import Transformer
from data_processor import DataProcessor
from masks import create_masks


print(f'TensorFlow version: {tf.__version__}')

tf.compat.v1.flags.DEFINE_integer('batch_size', 16, 'Batch size')
tf.compat.v1.flags.DEFINE_integer('buffer_size', 10000, 'Shuffle buffer size')
tf.compat.v1.flags.DEFINE_integer('num_epochs', 6, 'Number of training epochs')
tf.compat.v1.flags.DEFINE_integer('max_len', 256, 'Maximum length of input sentences')
FLAGS = tf.compat.v1.flags.FLAGS

batch_size = FLAGS.batch_size
num_epochs = FLAGS.num_epochs
max_len = FLAGS.max_len
buffer_size = FLAGS.buffer_size

data_proc = DataProcessor(batch_size=batch_size,
                          buffer_size=buffer_size,
                          max_len=max_len,
                          tokenizer=None)
target_vocab_size = data_proc.target_vocab_size
train_dataset, val_dataset = data_proc.train_dataset, data_proc.val_dataset

config = {
    'encoder': {
        'is_bert': True,
        'bert_hub_url': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
        'finetune_bert': False,
        'd_model': 768,
        # 'vocab_size': input_vocab_size,
        'num_layers': 4,
        'max_sequence_length': 1024,
        'dropout_rate': 0.1,
        'intermediate_size': 512,
        'num_attention_heads': 8,
        'attention_dropout_rate': 0.1
    },
    'decoder': {
        'd_model': 768,
        'vocab_size': target_vocab_size,
        'num_layers': 4,
        'max_sequence_length': 512,
        'dropout_rate': 0.4,
        'intermediate_size': 512,
        'num_attention_heads': 8,
        'attention_dropout_rate': 0.3
    }
}


def main(_):
    transformer = Transformer(config)

    # tokenizer = None
    # if config['encoder']['is_bert']:
    #     vocab_file = transformer.encoder.bert_encoder.resolved_object.vocab_file.asset_path.numpy()
    #     do_lower_case = transformer.encoder.bert_encoder.resolved_object.do_lower_case.numpy()
    #     tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    checkpoint_path = './ckpts'
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=transformer.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')


    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = transformer(inp,
                                      tar_inp,
                                      enc_padding_mask,
                                      combined_mask,
                                      dec_padding_mask,
                                      training=True)
            loss = transformer.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        transformer.optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(transformer.accuracy_function(tar_real, predictions))

    @tf.function(input_signature=train_step_signature)
    def val_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        predictions = transformer(inp,
                                  tar_inp,
                                  enc_padding_mask,
                                  combined_mask,
                                  dec_padding_mask,
                                  training=False)
        loss = transformer.loss_function(tar_real, predictions)

        val_loss(loss)
        val_accuracy(transformer.accuracy_function(tar_real, predictions))

    metrics_names = ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
    total_train_examples = None
    for epoch in range(num_epochs):
        print("\nepoch {}/{}".format(epoch + 1, num_epochs))

        progBar = tf.keras.utils.Progbar(total_train_examples, stateful_metrics=metrics_names)

        train_loss.reset_states()
        train_accuracy.reset_states()

        val_loss.reset_states()
        val_accuracy.reset_states()

        for (t_batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            values = [('train_loss', train_loss.result()), ('train_accuracy', train_accuracy.result())]
            progBar.update(t_batch * batch_size, values=values)

        for (batch, (inp, tar)) in enumerate(val_dataset):
            val_step(inp, tar)

        values = [('train_loss', train_loss.result()), ('train_accuracy', train_accuracy.result()),
                  ('val_loss', val_loss.result()), ('val_accuracy', val_accuracy.result())]

        total_train_examples = t_batch * batch_size
        progBar.update(total_train_examples, values=values)

        if (epoch + 1) % 3 == 0:
            ckpt_manager.save()
            print(f'\nCheckpoint saved - Epoch {epoch + 1}')


if __name__ == '__main__':
    tf.compat.v1.app.run()
