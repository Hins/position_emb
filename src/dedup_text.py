# -*- coding: utf-8 -*-
# @Time        : 2018/8/1 12:19
# @Author      : panxiaotong
# @Description : dedup embedding text

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("dedup_text <input file> <output file>")
        sys.exit()

    dedup_text_dict = {}
    with open(sys.argv[2], 'w') as out_f:
        with open(sys.argv[1], 'r') as in_f:
            for line in in_f:
                sentences = line.strip('\r\n').split('\t')[0]
                sentence_list = sentences.split('\001')
                for sentence in sentence_list:
                    if sentence not in dedup_text_dict:
                        dedup_text_dict[sentence] = 1
            in_f.close()
        for k,v in dedup_text_dict.items():
            out_f.write(k + "\n")
        out_f.close()