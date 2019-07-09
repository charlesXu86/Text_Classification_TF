# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   predict.py
 
@Time    :   2019-07-08 17:13
 
@Desc    :
 
'''

import tensorflow as tf

from TC_model.fasttext.parameters import parameters
from TC_model.fasttext.data_helper import load_json, padding


class Predict():
    def __init__(self, config=parameters, model='./runs/1562578812', word_to_index='./vocabs/word_to_index.json',
                 index_to_label='./vocabs/index_to_label.json'):
        self.word_to_index = load_json(word_to_index)
        self.index_to_label = load_json(index_to_label)

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=config['allow_soft_placement'],
                log_device_placement=config['log_device_placement'])
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # 加载训练好的pb模型
                tf.saved_model.loader.load(self.sess, ['tag_string'],model)

                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]

                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    def fc_predicts(self, msg):
        input_x = padding(msg, None, parameters, self.word_to_index, None)
        feed_dict = {
            self.input_x: input_x,
            self.dropout_keep_prob: 1    # 设置为1就是保留全部结果，所以这个只有在训练的时候用。
        }
        predictions = self.sess.run(self.predictions, feed_dict=feed_dict)
        return [self.index_to_label[str(idx)] for idx in predictions]


if __name__ == '__main__':
    prediction = Predict(parameters)
    result = prediction.fc_predicts(["""我打算明年买房"""])
    print(result)