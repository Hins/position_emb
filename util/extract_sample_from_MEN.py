# -*- coding: utf-8 -*-
# @Time        : 2018/8/29 18:50
# @Author      : panxiaotong
# @Description : extract samples from MEN dataset

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("extract_sample_from_MEN <input_file> <output_file>")
        sys.exit()

    with open(sys.argv[2], 'w') as o_f:
        pair_dict = {}
        with open(sys.argv[1], 'r') as i_f:
            for line in i_f:
                elements = line.strip('\r\n').split(' ')
                pair_dict[elements[0] + "_" + elements[1]] = float(elements[2])
            i_f.close()

        dedup_dict = {}
        for k,v in pair_dict.items():
            for k2,v2 in pair_dict.items():
                if k == k2 or v == v2 or k + "-" + k2 in dedup_dict or k2 + "-" + k in dedup_dict:
                    continue
                dedup_dict[k + "-" + k2] = 1
                dedup_dict[k2 + "-" + k] = 1
                if v > v2:
                    o_f.write(k + "\t" + k2 + "\n")
                else:
                    o_f.write(k2 + "\t" + k + "\n")
        o_f.close()