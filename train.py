import yaml
import json
import tensorflow as tf
from transformer import Transformer
from data_processor import DataProcessor
from masks import create_masks


tf.compat.v1.flags.DEFINE_integer('num_epochs', 6, 'Number of training epochs')
tf.compat.v1.flags.DEFINE_string('config_path', 'configs/default_config.yml',
                                  'Path to a YAML configuration file')
tf.compat.v1.flags.DEFINE_string('logdir', 'ckpts/', 'Directory to write event logs and store checkpoints.')
FLAGS = tf.compat.v1.flags.FLAGS

global_config = yaml.load(open(FLAGS.config_path))

data_proc = DataProcessor(global_config['data'])
target_vocab_size = data_proc.target_vocab_size
input_vocab_size = data_proc.target_vocab_size
train_dataset, val_dataset = data_proc.get_dataset('train'), data_proc.get_dataset('val')

global_config['transformer']['encoder']['vocab_size'] = input_vocab_size
global_config['transformer']['decoder']['vocab_size'] = target_vocab_size

# print(json.dumps(global_config, indent=4, sort_keys=True))

num_epochs = FLAGS.num_epochs
batch_size = global_config['data']['batch_size']


def main(_):
    transformer = Transformer(global_config['transformer'])

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=transformer.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.logdir, max_to_keep=10)

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

        # if (epoch + 1) % 3 == 0:
        ckpt_manager.save()
        print(f'\nCheckpoint saved - Epoch {epoch + 1}')


if __name__ == '__main__':
    tf.compat.v1.app.run()
