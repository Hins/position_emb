# -*- coding: utf-8 -*-
# @Time        : 2018/8/20 17:48
# @Author      : panxiaotong
# @Description : word break

import sys
import jieba

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("word_break <input file> <stopword file> <entity file> <output_file>")
        exit()

    jieba.load_userdict(sys.argv[3])
    stopword_dict = {}
    with open(sys.argv[2], 'r') as f:
        for line in f:
            stopword_dict[line.strip('\r\n')] = 1
        f.close()

    with open(sys.argv[4], 'w') as out_f:
        with open(sys.argv[1], 'r') as in_f:
            for line in in_f:
                seg_list = jieba.cut(line.strip('\r\n'), cut_all=False)
                seg_list = [item for item in seg_list if item.encode('utf-8') not in stopword_dict and item.strip() != ""]
                out_f.write(",".join(seg_list).encode('utf-8') + "\n")
            in_f.close()
        out_f.close()