from __future__ import print_function

import pandas as pd
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
import numpy as np
import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def main():
    np.random.seed(42)
    data_dir_path = './data'
    model_dir_path = './models'

    print('loading csv file ...')
    with open(data_dir_path + '/validation.en') as f:
        X = f.read().split('\n');

    with open(data_dir_path + '/validation.de') as f:
        Y = f.read().split('\n');

    config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path),allow_pickle=True).item()

    summarizer = Seq2SeqSummarizer(config)
    summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    print('start predicting ...')
    file1 = open("output_validation.txt","w") 
    for i in np.random.permutation(np.arange(len(X))):
        x = X[i]
        actual_headline = Y[i]
        headline = summarizer.summarize(x)
        # print('Article: ', x)
        file1.writelines('Input sentence: \n' + x + '\n')
        file1.writelines('Decoded sentence: \n'+headline+ '\n')
        file1.writelines('Original Decoded sentence: \n'+actual_headline+ '\n\n')
    file1.close()

if __name__ == '__main__':
    main()
