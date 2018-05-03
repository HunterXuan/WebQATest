import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from insqa_cnn import InsQACNN
import operator
import json
import jieba

def print_random_qa():
    # 一次打印10个qa
    data_helpers.print_random_qa()

def get_answers(i, batch=100):
    train_data = data_helpers.get_train_data()
    vocab = data_helpers.get_vocab()
    answers = []
    for index in range(i*batch, i*batch+batch):
        answers.append(data_helpers.encode_sent(vocab, train_data[index]['a'], 200))
    return answers

def get_questions(question,batch=100):
    vocab = data_helpers.get_vocab()
    q = data_helpers.encode_sent(vocab, question, 200)
    #print(q)
    questions = []
    for i in range(0, 100):
        questions.append(q)
    return questions

def get_answer(question):
    vocab = data_helpers.get_vocab()
    q = data_helpers.encode_sent(vocab, question, 200)

def get_top_answers(question):
    aids = get_best_answer_id(question)
    train_data = data_helpers.get_train_data()
    print('可能的答案：')
    for i in range(0, len(aids)):
        print(train_data[aids[i]]['a']) 
        print('-----')

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
#tf.flags.DEFINE_integer("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5000000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 3000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 3000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def get_best_answer_id(question):
    FLAGS = tf.flags.FLAGS    

    vocab = data_helpers.get_vocab()
    test_data = data_helpers.get_test_data()
    train_data = data_helpers.get_train_data()

    with tf.Graph().as_default():
        with tf.device("/gpu:1"):
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():        
                #saver = tf.train.import_meta_graph('./runs/1525274605/checkpoints/model-' + str(checkpoint) + '.meta', clear_devices=True)    
                #saver.restore(sess, tf.train.latest_checkpoint('./runs/1525274605/checkpoints'))
                 
                cnn = InsQACNN(
                    sequence_length=200,
                    batch_size=FLAGS.batch_size,
                    vocab_size=len(vocab),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint('./runs/1525274605/checkpoints'))
                
                max_q = len(train_data)
                print(max_q)
                scores = []
                for i in range(0, round(max_q/100)):
                    questions = get_questions(question)
                    answers = get_answers(i)
                    feed_dict = {
                        cnn.input_x_1: questions,
                        cnn.input_x_2: answers,
                        cnn.input_x_3: answers,
                        cnn.dropout_keep_prob: 1.0
                    }
                    batch_scores = sess.run([cnn.cos_12], feed_dict)
                    batch_scores = batch_scores[0]
                    batch_scores = batch_scores.tolist()
                    scores.append((max(batch_scores), i*100 + batch_scores.index(max(batch_scores))))
                    #print(scores)
                    #print(max(batch_scores))
                    #scores.extend(batch_scores)
                scores.sort(key=operator.itemgetter(0), reverse=True)
                #print(scores)
                result = []
                for i in range(0,5):
                    score,qid = scores[i]
                    #print(scores[i])
                    result.append(qid)
                #print(result)            
                return result

#get_top_answers('杨利伟是谁')
