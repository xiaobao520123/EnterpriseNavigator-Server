import numpy as np
import tensorflow as tf

from .parse_dict import get_dict

class TextRecognition(object):
    def __init__(self, pb_file, dict_file, seq_len):
        self.pb_file = pb_file
        self.dict_file = dict_file
        self.seq_len = seq_len
        self.init_model()
        self.init_dict()
    
    # 加载模型文件
    def init_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.io.gfile.GFile(self.pb_file, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')
        
        # 配置Tensorflow GPU设置
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True # 允许动态增长显存使用
        # 创建Tensorflow会话
        self.sess = tf.Session(graph=self.graph, config=config)
        
        # 获得模型参量对象
        self.img_ph = self.sess.graph.get_tensor_by_name('image:0')
        self.label_ph = self.sess.graph.get_tensor_by_name('label:0')
        self.is_training = self.sess.graph.get_tensor_by_name('is_training:0')
        self.dropout = self.sess.graph.get_tensor_by_name('dropout_keep_prob:0')
        self.preds = self.sess.graph.get_tensor_by_name('sequence_preds:0')
        self.probs = self.sess.graph.get_tensor_by_name('sequence_probs:0')

        # 这部分的参量来自AttentionOCR作者的Docker里的Demo模型(text_recognition.pb)
        ''' self.img_ph = self.sess.graph.get_tensor_by_name('image:0')
        self.is_training = self.sess.graph.get_tensor_by_name('is_training:0')
        self.dropout = self.sess.graph.get_tensor_by_name('dropout:0')
        self.preds = self.sess.graph.get_tensor_by_name('sequence_preds:0')
        self.probs = self.sess.graph.get_tensor_by_name('sequence_probs:0')'''
    
    # 加载标签字典文件
    def init_dict(self):
        self.lable_dict = get_dict(self.dict_file)

    # 执行OCR预测过程
    def predict(self, image_padded, EOS='EOS'):
        results = []
        probabilities = []
        
        pred_sentences, pred_probs = self.sess.run([self.preds, self.probs], \
                    #feed_dict={self.is_training: False, self.dropout: 1.0, self.img_ph: image})
                    feed_dict={self.is_training: False, self.dropout: 1.0, self.img_ph: image_padded, self.label_ph: np.ones((1, self.seq_len), np.int32)})
        
        for pred_sentence in pred_sentences:
            char_list = []
            for char in pred_sentence:
                if self.lable_dict[char] == EOS: # 当识别到句子末尾结束标志时，结束循环
                    break
                char_list.append(self.lable_dict[char])
            results.append(char_list)
        
        for pred_prob in pred_probs:
            char_probs = pred_prob[:min(len(results)+1, self.seq_len)]
            probabilities.append(char_probs)

        return results, probabilities