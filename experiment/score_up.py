# -- coding:utf-8 --
import json
import jieba
import time
import jieba
import jieba.analyse
from sentence_transformers.util import cos_sim 
from sentence_transformers import SentenceTransformer as SBert
from score1 import calc_bleu, calc_distinct, calc_f1 

model = SBert("D:\\model\paraphrase-multilingual-MiniLM-L12-v2")


def EAS(reference,candidate):
    embeddings1 = model.encode(reference)
    embeddings2 = model.encode(candidate)
    score = cos_sim(embeddings1, embeddings2)
    return float(score)


def NewEnt(pred,x):
    # allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v','nr']. if the POS of w is not in this list,it will be filtered.
    pred_entity = jieba.analyse.extract_tags(pred,allowPOS=('ns','n','nr','vn'))
    length = len(pred_entity)
    if length != 0:
        score = 1 - 1/length
    else:
        score = 0
    return score



def KeyEnt(pred,x):
    pred_entity = jieba.analyse.extract_tags(pred,allowPOS=('ns','n','nr','vn'))
    x_entity = jieba.analyse.extract_tags(x,allowPOS=('ns','n','nr','vn'))
    for item in x_entity:
        if item not in pred_entity:
            return 0
    return 1
       


# 评分函数
# pred_list 生成的list
# gold_list 参考list
# method 方法参数  F1\BLEU\Distinct\EAS

def score(pred_list,gold_list,method):
    res_list = []
    if method != 'EAS' and method != 'NewEnt':
        for i in range(len(pred_list)):
            res_list.append([list(jieba.cut(pred_list[i])),list(jieba.cut(gold_list[i]))])
    if method == 'F1': 
        score = calc_f1(res_list)
    if method == 'BLEU':
        score = calc_bleu(res_list)
        score = score[0]
    if method == 'Distinct':
        score = calc_distinct(res_list)
        score = score[1]
    if method == 'EAS':
        score = 0
        for i in range(len(pred_list)):
            score += EAS(pred_list[i],gold_list[i])
        score = score /len(pred_list)
    if method == 'NewEnt':
        score = 0
        for i in range(len(pred_list)):
            score += NewEnt(pred_list[i],gold_list[i])
        score = score /len(pred_list)
    if method == 'KeyEnt':
        score = 0
        for i in range(len(pred_list)):
            score += KeyEnt(pred_list[i],gold_list[i])
        score = score /len(pred_list)

        
    return score



def list_score(pred_list,f_out,type):
    if type == 1:
        # 加载数据
        content_pred,x,x_entity,A,B,C,D = [],[],[],[],[],[],[]
        for item in pred_list:
            content_pred.append(item['content'])
            x.append(str(item['x']))
            x_entity.append([item['x']['entity1'],item['x']['entity2']])
            A.append(item['A'])
            B.append(item['B'])
            C.append(item['C'])
            D.append(item['D'])

        # 计算得分
        # # 生成的题目 VS x
        EAS_score = score(content_pred,x,'EAS')
        BLEU_score = score(content_pred,x,'BLEU')
        F1_score = score(content_pred,x,'F1')
        f_out.write("\nPred_content VS x:")
        f_out.write('\nEAS:%.4f' % EAS_score)
        f_out.write('\nBLEU:%.4f' % BLEU_score)
        f_out.write('\nF1:%.4f' % F1_score)

        # # Distinct
        Distinct_score = score(content_pred,x,'Distinct')
        f_out.write('\nDistinct:%.4f' % Distinct_score)

        # entity
        NewEnt_score = score(content_pred,x_entity,'NewEnt')
        f_out.write('\nNewEnt_content:%.4f' % NewEnt_score)


        # # Choices
        # EAS
        score1 = score(A,x_entity,'NewEnt')
        score2 = score(B,x_entity,'NewEnt')
        score3 = score(C,x_entity,'NewEnt')
        score4 = score(D,x_entity,'NewEnt')
        avg = (score1 + score2 + score3 + score4) /4
        f_out.write("\nchoices EAS:")
        # f_out.write("\nA:%.4f" % score1)
        # f_out.write("\nB:%.4f" % score2)
        # f_out.write("\nC:%.4f" % score3)
        # f_out.write("\nD:%.4f" % score4)
        f_out.write("\nAvg:%.4f" % avg)





        # NewEnt
        score1 = score(A,x_entity,'NewEnt')
        score2 = score(B,x_entity,'NewEnt')
        score3 = score(C,x_entity,'NewEnt')
        score4 = score(D,x_entity,'NewEnt')
        avg = (score1 + score2 + score3 + score4) /4
        f_out.write("\nchoices NewEnt:")
        f_out.write("\nA:%.4f" % score1)
        f_out.write("\nB:%.4f" % score2)
        f_out.write("\nC:%.4f" % score3)
        f_out.write("\nD:%.4f" % score4)
        f_out.write("\nAvg:%.4f" % avg)

    elif type == 3:
        # 加载数据
        pred_mat1, pred_mat2,pred_ques1,pred_ques2,pred_ans1,pred_ans2,x,x_entity = [],[],[],[],[],[],[],[]
        for item in pred_list:
            pred_mat1.append(item['mat1'])
            pred_ques1.append(item['ques1'])
            pred_ques2.append(item['ques2'])
            pred_ans1.append(item['ans1'])
            pred_ans2.append(item['ans2'])
            x.append(str(item['x']))
            x_entity.append([item['x']['entity1'],item['x']['entity2']])
            if item['mat2'] == '':
                pred_mat2.append(item["mat1"])
            else:
                pred_mat2.append(item["mat2"]) 
    
        
        # 计算得分


        # # 生成的材料 VS 知识点
        EAS_score = (score(pred_mat1,x,'EAS') + score(pred_mat2,x,'EAS'))/2
        BLEU_score = ( score(pred_mat1,x,'BLEU')+ score(pred_mat2,x,'BLEU'))/2
        F1_score = (score(pred_mat1,x,'F1') + score(pred_mat2,x,'F1'))/2
        f_out.write('\nPred_material VS x')
        f_out.write('\nEAS:%.4f' % EAS_score)
        f_out.write('\nBLEU:%.4f' % BLEU_score)
        f_out.write('\nF1:%.4f' % F1_score)


         


        # # Distinct
        Distinct_score = ( score(pred_mat1,x,'Distinct') + score(pred_mat2,x,'Distinct'))/2
        f_out.write('\nDistinct:%.4f' %Distinct_score)

        # entity
        NewEnt_score =(score(pred_mat1,x_entity,'NewEnt') + score(pred_mat2,x_entity,'NewEnt'))/2
        f_out.write('\nNewEnt:%.4f' % NewEnt_score)


        # 生成的问题 VS 初始问题
        
        NewEnt_score1 =score(pred_ques1,x_entity,'NewEnt') 
        NewEnt_score2 =score(pred_ques2,x_entity,'NewEnt') 
        NewEnt_avg = (NewEnt_score1 + NewEnt_score2) / 2
        f_out.write('\nNewEnt_q1:%.4f' % NewEnt_score1)
        f_out.write('\nNewEnt_q2:%.4f' % NewEnt_score2)
        f_out.write('\nNewEnt_avg:%.4f' % NewEnt_avg)


def file_score():
    f_out = open('a1.txt', 'a', encoding='utf8')
    LLMs = ['Wenxin','Qwen','Hunyuan','SparkDesk','Chatgpt']
    #LLMs = ['Wenxin']
    types = [1,3]
    # types = [1]
    for type in types:
        if type == 1:
                gold_path = 'value1/choice.json'
        elif type == 3:
                gold_path = 'value1/analyse.json'
        with open(gold_path, 'r', encoding='utf8') as f:
            data1 = f.read()
        gold_list = json.loads(data1)

        for LLM in LLMs:
            if type == 1:
                pred_path ='ques_LLM/'+ LLM + '_choice.json'
            elif type == 3:
                pred_path ='ques_LLM/'+ LLM + '_analyse.json'
            f_out.write('\n\n'+pred_path[9:-5])
            with open(pred_path, 'r', encoding='utf8') as f:
                data1 = f.read()
            pred_list = json.loads(data1)
            methods = ['Rand','KP','KKP','KAG']
            for method in methods:
                f_out.write('\n\n\n'+method)
                temp_list = pred_list[method]
                list_score(temp_list,f_out,type)












if __name__ == '__main__':
    start = time.time()  
    file_score()
    end = time.time()
    print("时间",str((end-start)/60))
         