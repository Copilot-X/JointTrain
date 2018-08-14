from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def sep(is_training):
    if is_training:
        framework = Framework(is_training=True)
    else:
        framework = Framework(is_training=False)
    # cnn
    word_embedding = framework.embedding.word_embedding()
    pos_embedding = framework.embedding.pos_embedding()
    embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
    x = framework.encoder.cnn(embedding, FLAGS.hidden_size, framework.mask, activation=tf.nn.relu)
    logit, repre = framework.selector.attention(x, framework.scope, framework.label_for_select)
    cnn_loss = framework.classifier.softmax_cross_entropy(logit)
    output = framework.classifier.output(logit)
    # gcn
    rela_embed = framework.gcn.gcn(framework.features, framework.supports, framework.num_features_nonzero)
    gcn_loss = framework.gcn.loss(rela_embed, framework.gcn_label)

    framework.init_train_model(cnn_loss,
                               output,
                               optimizer=tf.train.AdamOptimizer,
                               gcn_loss=gcn_loss,
                               gcn_optimizer=tf.train.AdamOptimizer)
    framework.load_train_data()
    # train cnn
    framework.train()
    # train gcn
    framework.train_gcn()

