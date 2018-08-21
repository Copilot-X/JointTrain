from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def joint(is_training):
    if is_training:
        framework = Framework(is_training=True)
    else:
        framework = Framework(is_training=False)

    word_embedding = framework.embedding.word_embedding()
    pos_embedding = framework.embedding.pos_embedding()
    embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
    x = framework.encoder.cnn(embedding, FLAGS.hidden_size, framework.mask, activation=tf.nn.relu)
    rela_embed = framework.gcn.gcn(framework.features, framework.supports, framework.num_features_nonzero)
    logit, repre = framework.selector.attention(x, framework.scope, framework.label_for_select)

    if is_training:
        loss = framework.classifier.softmax_cross_entropy(logit)
        output = framework.classifier.output(logit)
        framework.init_train_model(loss, output, optimizer=tf.train.AdamOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        framework.init_test_model(logit)
        framework.load_test_data()
        framework.test()