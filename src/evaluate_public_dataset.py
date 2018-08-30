# -*- coding: utf-8 -*-
# @Time        : 2018/8/30 11:02
# @Author      : panxiaotong
# @Description : evaluate in public dataset

import sys
import numpy as np
from numpy import linalg as la

def cosSimilar(inA,inB):
    inA = np.mat(inA)
    inB = np.mat(inB)
    num = float(inA * inB.T)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5+0.5*(num/denom)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("evaluate_public_dataset <input data file> <word emb file> <dict file> <output metrics file>")
        sys.exit()

    word_dict = {}
    with open(sys.argv[3], 'r') as f:
        for line in f:
            word_dict[len(word_dict)] = line.strip('\r\n')
        f.close()

    word_emb_dict = {}
    with open(sys.argv[2], 'r') as f:
        for index, line in enumerate(f):
            word_emb_dict[word_dict[index]] = [float(item) for item in line.strip('\r\n').split(',')]
        f.close()

    with open(sys.argv[4], 'w') as o_f:
        with open(sys.argv[1], 'r') as f:
            counter = 0
            correct_counter = 0
            for line in f:
                elements = line.strip('\r\n').split('\t')
                word1 = elements[0].split('_')[0]
                word2 = elements[0].split('_')[1]
                word3 = elements[1].split('_')[0]
                word4 = elements[1].split('_')[1]
                if word1 in word_emb_dict and word2 in word_emb_dict and word3 in word_emb_dict and word4 in word_emb_dict:
                    counter += 1
                if cosSimilar(word_emb_dict[word1], word_emb_dict[word2]) > cosSimilar(word_emb_dict[word3], word_emb_dict[word4]):
                    correct_counter += 1
            f.close()
        if counter == 0:
            o_f.write("0")
        else:
            o_f.write(str(float(correct_counter) / float(counter)))
        o_f.close()