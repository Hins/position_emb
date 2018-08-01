# -*- coding: utf-8 -*-
# @Time        : 2018/7/30 14:47
# @Author      : panxiaotong
# @Description : extract training samples from text and dictionary

import random
import sys
from config import cfg

if __name__ == "__main__":
    '''
    <input file>: <word_1>,<word_2> \cdots <word_n>
    <dict file>: <word>\t<index>
    <output file>: <word>\t<pos/neg word>\t<position>
    '''
    if len(sys.argv) < 4:
        print("load_sample <input file> <dict file> <output file>")
        sys.exit()

    word_dict = {}
    reverse_word_dict = {}
    with open(sys.argv[2], 'r') as f:
        for line in f:
            line = line.strip('\r\n').split('\t')
            word_dict[line[0]] = int(line[1])
            reverse_word_dict[int(line[1])] = line[0]
        f.close()
    word_dictionary_size = len(word_dict)

    word1_list = []
    word2_list = []
    position_list = []
    with open(sys.argv[3], 'w') as w_f:
        with open(sys.argv[1], 'r') as f:
            for item in f:
                elements = item.strip('\r\n').split(",")
                word_len = len(elements)
                for index, word in enumerate(elements):
                    if word not in word_dict:
                        continue
                    if index > 1 and elements[index - 2] in word_dict:
                        word1_list.append(word_dict[word])
                        word2_list.append(word_dict[elements[index - 2]])
                        position_list.append(1)
                        accu = 0
                        while accu < cfg.negative_sample_size:
                            negative_index = random.randint(1, word_dictionary_size)
                            negative_word = reverse_word_dict[negative_index]
                            if negative_word not in elements:
                                word1_list.append(word_dict[word])
                                word2_list.append(negative_index)
                                position_list.append(0)
                                accu += 1
                    if index > 0 and elements[index - 1] in word_dict:
                        word1_list.append(word_dict[word])
                        word2_list.append(word_dict[elements[index - 1]])
                        position_list.append(2)
                        accu = 0
                        while accu < cfg.negative_sample_size:
                            negative_index = random.randint(1, word_dictionary_size)
                            negative_word = reverse_word_dict[negative_index]
                            if negative_word not in elements:
                                word1_list.append(word_dict[word])
                                word2_list.append(negative_index)
                                position_list.append(0)
                                accu += 1
                    if index + 1 < word_len:
                        word1_list.append(word_dict[word])
                        word2_list.append(word_dict[elements[index + 1]])
                        position_list.append(3)
                        accu = 0
                        while accu < cfg.negative_sample_size:
                            negative_index = random.randint(1, word_dictionary_size)
                            negative_word = reverse_word_dict[negative_index]
                            if negative_word not in elements:
                                word1_list.append(word_dict[word])
                                word2_list.append(negative_index)
                                position_list.append(0)
                                accu += 1
                    if index + 2 < word_len:
                        word1_list.append(word_dict[word])
                        word2_list.append(word_dict[elements[index + 2]])
                        position_list.append(4)
                        accu = 0
                        while accu < cfg.negative_sample_size:
                            negative_index = random.randint(1, word_dictionary_size)
                            negative_word = reverse_word_dict[negative_index]
                            if negative_word not in elements:
                                word1_list.append(word_dict[word])
                                word2_list.append(negative_index)
                                position_list.append(0)
                                accu += 1
            f.close()
        for index, word in enumerate(word1_list):
            w_f.write(str(word) + "\t" + str(word2_list[index]) + "\t" + str(position_list[index]) + "\n")
        w_f.close()