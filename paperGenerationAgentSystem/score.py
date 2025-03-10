# -- coding:utf-8 --
import json
import time
import jieba
from sentence_transformers.util import cos_sim 
from sentence_transformers import SentenceTransformer as SBert
from score1 import calc_bleu, calc_distinct, calc_f1 

model = SBert("/home/suzhangjie/paper/paper/paraphrase-multilingual-MiniLM-L12-v2")

def NewEnt(pred, x):
    # allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v','nr']. if the POS of w is not in this list,it will be filtered.
    pred_entity = jieba.analyse.extract_tags(pred, allowPOS=('ns', 'n', 'nr', 'vn'))
    length = len(pred_entity)
    if length != 0:
        score = 1 - 1 / length
    else:
        score = 0
    return score


def EAS(reference,candidate):
    embeddings1 = model.encode(reference)
    embeddings2 = model.encode(candidate)
    score = cos_sim(embeddings1/2, embeddings2/2)
    return float(score)

# 评分函数
# pred_list 生成的list
# gold_list 参考list
# method 方法参数  F1\BLEU\Distinct\EAS

def score(pred_list,gold_list,method):
    res_list = []
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
            score += NewEnt(pred_list[i], gold_list[i])
        score = score / len(pred_list)

    return score
        


# 选择题题评分
#  pred_path 生成题目的json文件
#  gold_path 初始题目的json文件
def choice_score(pred_path,gold_path):
    with open(gold_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    gold_list = json.loads(data1)
    with open(pred_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    pred_list = json.loads(data1)

    print("Pred_content VS Gold_content:")
    content_pred = [item['content'] for item in pred_list]
    content_gold = [item['content'] for item in gold_list]
    # NewEnt
    score1 = score(A, x_entity, 'NewEnt')
    score2 = score(B, x_entity, 'NewEnt')
    score3 = score(C, x_entity, 'NewEnt')
    score4 = score(D, x_entity, 'NewEnt')
    avg = (score1 + score2 + score3 + score4) / 4
    f_out.write("\nchoices NewEnt:")
    f_out.write("\nA:%.4f" % score1)
    f_out.write("\nB:%.4f" % score2)
    f_out.write("\nC:%.4f" % score3)
    f_out.write("\nD:%.4f" % score4)
    f_out.write("\nAvg:%.4f" % avg)

    # print("Pred_content VS x:")
    # x = [str(item['x']) for item in pred_list]
    # EAS = score(content_pred,x,'EAS')
    # print('EAS:',EAS)
    # BLEU = score(content_pred,x,'BLEU')
    # print('BLEU:',BLEU)
    # # Distinct = score(content_pred,x,'Distinct')
    # # print('Distinct:',Distinct)
    # F1 = score(content_pred,x,'F1')
    # print('F1:',F1)

    print("Pred_content VS choices:")
    A = [item['A'] for item in pred_list]
    EAS_A = score(A,content_pred,'EAS')
    print('EAS_A:',EAS_A)
    B = [item['B'] for item in pred_list]
    EAS_B = score(B,content_pred,'EAS')
    print('EAS_B:',EAS_B)
    C = [item['C'] for item in pred_list]
    EAS_C = score(C,content_pred,'EAS')
    print('EAS_C:',EAS_C)
    D = [item['D'] for item in pred_list]
    EAS_D = score(D,content_pred,'EAS')
    print('EAS_D:',EAS_D)

    print("Gold_content VS choices:")
    A = [item['A'] for item in gold_list]
    EAS_A = score(A,content_pred,'EAS')
    print('EAS_A:',EAS_A)
    B = [item['B'] for item in pred_list]
    EAS_B = score(B,content_pred,'EAS')
    print('EAS_B:',EAS_B)
    C = [item['C'] for item in pred_list]
    EAS_C = score(C,content_pred,'EAS')
    print('EAS_C:',EAS_C)
    D = [item['D'] for item in pred_list]
    EAS_D = score(D,content_pred,'EAS')
    print('EAS_D:',EAS_D)
   

# 材料题评分
#  pred_path 生成题目的json文件
#  gold_path 初始题目的json文件  
def analyse_score(pred_path,gold_path):
    with open(gold_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    gold_list = json.loads(data1)
    with open(pred_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    pred_list = json.loads(data1)

    pred_mat1, pred_mat2,pred_ques1,pred_ques2,pred_ans1,pred_ans2,x = [],[],[],[],[],[],[]
    for item in pred_list:
        pred_mat1.append(item['mat1'])
        pred_ques1.append(item['ques1'])
        pred_ques2.append(item['ques2'])
        pred_ans1.append(item['ans1'])
        pred_ans2.append(item['ans2'])
        x.append(str(item['x']))
        if item['mat2'] == '':
            pred_mat2.append(item["mat1"])
        else:
            pred_mat2.append(item["mat2"]) 
   
    gold_mat1, gold_mat2 = [],[]
    for item in gold_list:
        gold_mat1.append(item['mat1'])
        if item['mat2'] == '':
            gold_mat2.append(item["mat1"])
        else:
            gold_mat2.append(item["mat2"])
  

    

    EAS_score = (score(pred_mat1,gold_mat1,'EAS') + score(pred_mat2,gold_mat2,'EAS'))/2
    BLEU_score =( score(pred_mat1,gold_mat1,'BLEU')+score(pred_mat2,gold_mat2,'BLEU') )/2
    Distinct_score = ( score(pred_mat1,gold_mat1,'Distinct') + score(pred_mat2,gold_mat2,'Distinct'))/2
    F1_score = (score(pred_mat1,gold_mat1,'F1') + score(pred_mat2,gold_mat2,'F1'))/2
    print('Pred_material VS Gold_material')
    print('EAS:',EAS_score)
    print('BLEU:',BLEU_score)
    print('Distinct:',Distinct_score)
    print('F1:',F1_score)

    EAS_score = (score(pred_mat1,x,'EAS') + score(pred_mat2,x,'EAS'))/2
    BLEU_score = ( score(pred_mat1,x,'BLEU')+ score(pred_mat2,x,'BLEU'))/2

    F1_score = (score(pred_mat1,x,'F1') + score(pred_mat2,x,'F1'))/2
    print('Pred_material VS x')
    print('EAS:',EAS_score)
    print('BLEU:',BLEU_score)
    print('F1:',F1_score)


    EAS_q1 = (score(pred_ques1,pred_mat1,'EAS') + score(pred_ques1,pred_mat2,'EAS'))/2
    EAS_q2 = (score(pred_ques2,pred_mat1,'EAS') + score(pred_ques2,pred_mat2,'EAS'))/2
    
    print('Pred_ques VS Pred_material  ')
    print('EAS_q1:',EAS_q1)
    print('EAS_q2:',EAS_q2)

    # EAS_a1 = score(pred_ques1,pred_ans1,'EAS')
    # EAS_a2 = score(pred_ques2,pred_ans2,'EAS')
    
    # print('Pred_ans VS Pred_ques')
    # print('EAS_a1:',EAS_a1)
    # print('EAS_a2:',EAS_a2)
  











if __name__ == '__main__':
    start = time.time()  

    methods = ['Rand','KP','KKP','KAG'] 
    gold_path = 'value1/choice.json'
    for method in methods:
        pred_path = 'ques_choice1/'+ str(method) + '.json'
        choice_score(pred_path,gold_path)
 
 
    end = time.time()
    print ("时间",str((end-start)/60))
         