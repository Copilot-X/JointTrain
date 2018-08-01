from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def pcnn_att(is_training):
    if is_training:
        framework = Framework(is_training=True)
    else:
        framework = Framework(is_training=False)
    # embedding
    word_embedding = framework.embedding.word_embedding()
    pos_embedding = framework.embedding.pos_embedding()
    embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
    # pcnn layer
    x = framework.encoder.pcnn(embedding, FLAGS.hidden_size, framework.mask, activation=tf.nn.relu)
    # gcn layer
    rela_embed = framework.gcn.gcn(framework.features, framework.supports, framework.num_features_nonzero)
    # attention
    logit, repre = framework.selector.attention(x, framework.scope, framework.label_for_select, gcn_rela=rela_embed)

    if is_training:
        loss = framework.classifier.softmax_cross_entropy(logit)
        output = framework.classifier.output(logit)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        framework.init_test_model(tf.nn.softmax(logit))
        framework.load_test_data()
        framework.test()

