from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from keras_text_summarization.library.utility.plot_utils import plot_and_save_history
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
from keras_text_summarization.library.applications.fake_news_loader import fit_text
import numpy as np
import os
import tensorflow as tf

LOAD_EXISTING_WEIGHTS = False
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


def main():
    np.random.seed(42)
    data_dir_path = './data'
    report_dir_path = './reports'
    model_dir_path = './models'

    print('loading csv file ...')
    #df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    print('extract configuration from input texts ...')
    with open(data_dir_path + '/train_preprocessed.en') as f:
        X = f.read().split('\n');

    with open(data_dir_path + '/train_preprocessed.de') as f:
        Y = f.read().split('\n');
    config = fit_text(X, Y)

    summarizer = Seq2SeqSummarizer(config)

    if LOAD_EXISTING_WEIGHTS:
        summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('demo size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = summarizer.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=100)

    history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history.png'
    if LOAD_EXISTING_WEIGHTS:
        history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history-v' + str(summarizer.version) + '.png'
    plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={'loss', 'acc'})


if __name__ == '__main__':
    main()
