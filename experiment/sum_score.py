# -- coding:utf-8 --
# 计算相关度得分
import torch
from sentence_transformers.util import cos_sim  
from sentence_transformers import SentenceTransformer as SBert




model = SBert("D:/aPythonProject/zujuanSystem/paper/paraphrase-multilingual-MiniLM-L12-v2")


def word_score(sents_1, sents_2):
    counter = 0
    for sent in sents_1:
        if sent in sents_2:
            counter += 1
    sents_similarity=counter/(len(sents_1) + len(sents_2) - counter)
    return sents_similarity

def cos_score(entity1,entity2):
    embeddings1 = model.encode(entity1)
    embeddings2 = model.encode(entity2)
    score = cos_sim(embeddings1, embeddings2)
    if score.shape[0] > 1:
        # print(score)
        score = torch.max(score)
    return float(score) 


# 两个三元组相关度得分
def list_score(list1,list2):
    score1 = word_score(''.join(list1),''.join(list2))
    score2 = cos_score(list1,list2)
    res =(score1 + score2) / 2
    # print("得分：", res)
    return res
#定义kp知识点，将kp与item所有的部位比较，取最大值
def kp_score(kp,list):
    res=0
    for l in list:
        s=cos_score(kp,l)
        res=max(s,res)
    # print("得分：", res)
    return res

def kkp_score(kp,list):
    res=cos_score(kp,''.join(list))
    # print("得分：", res)
    return res

#返回知识点列表中每一个知识点分别与词条中的六元组的相似度的总得分
def sort_score(kp_list,kg):
    import tool
    tool.set_gpu()
    # print()
    # print(kp_list)
    # print(kg)
    return word_score(''.join(kp_list),''.join(kg[1][2-7]))



if __name__ == '__main__':
    entity1 = [1, (632, 124, '1069年', '宋神宗、王安石', '富国强兵', '变法', '目的', '富国强兵')]
    entity2 = ['奖励耕种','郡县制' ]
    # score1 = word_score(entity1,entity2)
    # score2 = bert_score(entity1,entity2)
    # score3 = nlp_score(entity1,entity2)
    # print("字符分数:",score1)
    # print("飞蒋分数:",score3)
    # print("bert分数:",score2)
    # get_score(entity1,entity2)
    score =sort_score(entity2,entity1)
    print("分数",score)
