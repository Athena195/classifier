import os
from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim


def get_data(folder_path, train_data):
    re_file = open(train_data, 'w', encoding='utf8')
    dirs = os.listdir(folder_path)
    topic = ['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh', 'Phap luat', 'Suc khoe', 'The gioi', 'The thao',
             'Van hoa', 'Vi tinh']
    count = 0;
    for path in dirs:
        print(path)
        file_paths = os.listdir(os.path.join(folder_path, path))
        label = '__label__'
        for i in range(topic.__len__()):
            if (path == topic[i]):
                label = label + str(i)
        for file_path in file_paths:
            text = label + ' , '
            with open(os.path.join(folder_path, path, file_path), 'r', encoding='utf-16') as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)
                text = text + lines
            re_file.write(text + '\n')

    re_file.close()


if __name__ == '__main__':
    train_path = 'D:\\My Document\\NLP\\Task3\\VNTC-master\\Data\\10Topics\\Ver1.1\\Train_Full1'
    train_data = 'D:\\My Document\\NLP\\Text-Classification-Models-Pytorch\\data\\topic\\train_data.txt'
    get_data(train_path, train_data)

    test_path = 'D:\\My Document\\NLP\\Task3\\VNTC-master\\Data\\10Topics\\Ver1.1\\Test_Full1'
    test_data =  'D:\\My Document\\NLP\\Text-Classification-Models-Pytorch\\data\\topic\\test_data.txt'
    get_data(test_path, test_data)

