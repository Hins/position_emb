import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('word_embedding_size', 256, 'word embedding size')
flags.DEFINE_integer('position_embedding_size', 100, 'position embedding size')
flags.DEFINE_integer('embedding_window', 4, 'embedding window')
flags.DEFINE_integer('hidden_size', 128, 'word2vec weight size')
flags.DEFINE_float('train_set_ratio', 0.7, 'train set ratio')
flags.DEFINE_integer('batch_size', 256, 'train batch size')
flags.DEFINE_integer('top_k_sim', -5, 'top k similarity items')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_integer('negative_sample_size', 5, 'negative sample size')
flags.DEFINE_integer('MAX_GRAD_NORM', 5, 'maximum gradient norm')
flags.DEFINE_integer('epoch_size', 5, 'epoch size')

flags.DEFINE_string('summaries_dir', '../tb/multi-classification-position', 'Summaries directory')
flags.DEFINE_string('train_summary_writer_path', '/train', 'train summary writer path')
flags.DEFINE_string('test_summary_writer_path', '/test', 'test summary writer path')

cfg = tf.app.flags.FLAGS