import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


def parse_log(log_path):
    train_logs = []
    test_logs = []
    last_epoch = 0
    last_epoch_log = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.startswith('Epoch:'):
                this_epoch = int(re.findall('\d+', line)[0])
                log_info = line.split(' ')
                # loss, prec1, prec2
                this_epoch_log = [this_epoch, log_info[8], log_info[10]]

                if this_epoch == last_epoch + 1:
                    train_logs.append(map(float, last_epoch_log))
                last_epoch_log = this_epoch_log
                last_epoch = this_epoch
            elif line.startswith('Testing Results:'):
                log_info = line.split(' ')
                test_log = [last_epoch, log_info[7], log_info[3]]
                test_logs.append(map(float, test_log))
    train_logs_df = pd.DataFrame(train_logs)
    train_logs_df.columns = ['epoch', 'loss', 'prec1']
    test_logs_df = pd.DataFrame(test_logs)
    test_logs_df.columns = ['epoch', 'loss', 'prec1']

    return train_logs_df, test_logs_df


def vis_logs(train_logs, test_logs):
    train_logs['subject'] = 0
    test_logs['subject'] = 1
    logs = pd.concat([train_logs, test_logs])

    fig, ax = plt.subplots(1, 2)
    infos = ['loss', 'prec1']
    for i in range(2):
        # show_figure(data=logs, time='epoch', value=infos[i])
        sns.tsplot(data=logs, time='epoch', value=infos[i], unit='subject', condition='subject', ax=ax[i])
    plt.show()


def show_figure(data, time, value):
    sns.tsplot(data=data, time=time, value=value)
    plt.show()


if __name__ == '__main__':
    log_file_path = 'E:\VIRAT\\action_recognition\\res\TRN\\vehicle_person\RGB\\reverse_270epoch\log\TRN_vehicle_person_RGB_BNInception_TRNmultiscale_segment8.csv'
    # \vehicle_person\RGB\log\
    train_logs, test_logs = parse_log(log_file_path)
    vis_logs(train_logs, test_logs)

