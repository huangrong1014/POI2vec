# -*- coding: utf-8 -*-
import para,os
import algorithm
if not os.path.isdir(para.node2vec_path):
    os.mkdir(para.node2vec_path)


def main(path):
    words_num,vec=algorithm.readVectors2(para.node2vec_path+"\words_vec_0.25_0.75.txt")
    f = open(path, "w", encoding="utf-8")
    f.write(words_num+'\n')
    for i in range(0, int(words_num)):
        word=vec[i]
        f.write('\n'+word+'\n')
        f.write('place2vec'+'\n')
        words_place2vec=algorithm.getSimWords_of_a_word("D:\word2vec\data\group\place2vec8\wordvec_place2vec.txt",word)
        f.write(words_place2vec)
        f.write('node2vec'+'\n')
        words_node2vec=algorithm.getSimWords_of_a_word(para.node2vec_path+"\words_vec_0.25_0.75.txt", word)
        f.write(words_node2vec)
    f.close()

if __name__ == "__main__":
	main(para.node2vec_path+"\compare_node2vec_0.25_0.75.txt")