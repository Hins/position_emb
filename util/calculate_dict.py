# -*- coding: utf-8 -*-
# @Time        : 2018/7/26 19:03
# @Author      : panxiaotong
# @Description : calculate dictionary

import sys

if __name__ == "__main__":
    '''
    <input file>: <word_1>,<word_2> \cdots <word_n>
    <output file>: <word>\t<index>
    '''
    if len(sys.argv) < 3:
        print("prepare_sample <input file> <output file>")
        sys.exit()

    with open(sys.argv[2], 'w') as out_file:
        word_dict = {}
        with open(sys.argv[1], 'r') as in_file:
            for item in in_file:
                words = item.strip("\r\n").split(",")
                for word in words:
                    if word not in word_dict:
                        word_dict[word] = len(word_dict) + 1
            in_file.close()

        for k,v in word_dict.items():
            out_file.write(k + "\t" + str(v) + "\n")
        out_file.close()