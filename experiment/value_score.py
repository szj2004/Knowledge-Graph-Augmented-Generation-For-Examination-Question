import time
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import jieba
import json
import torch
from statistics import mean
from rouge import Rouge
from sentence_transformers.util import cos_sim
import bert_score  
from sentence_transformers import SentenceTransformer as SBert
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertModel, BertTokenizer
from nltk.tokenize import word_tokenize
from torch.nn import CrossEntropyLoss
import nltk
from score1 import calc_bleu, calc_distinct, calc_f1 
smooth = SmoothingFunction() 

#model = SBert("D:\\model\paraphrase-multilingual-MiniLM-L12-v2")

model_name = 'D:\\model\\bert_base_chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)



def WER(reference,candidate):
    """编辑距离
    计算两个序列的levenshtein distance，可用于计算 WER/CER
    参考资料：
        https://www.cuelogic.com/blog/the-levenshtein-algorithm
        https://martin-thoma.com/word-error-rate-calculation/

    C: correct
    W: wrong
    I: insert
    D: delete
    S: substitution

    :param hypothesis: 预测序列
    :param reference: 真实序列
    :return: 1: 错误操作，所需要的 S，D，I 操作的次数;
             2: ref 与 hyp 的所有对齐下标
             3: 返回 C、W、S、D、I 各自的数量
    """
    reference = list(jieba.cut(reference))
    hypothesis = list(jieba.cut(candidate))
    len_hyp = len(hypothesis)
    len_ref = len(reference)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

    # 记录所有的操作，0-equal；1-insertion；2-deletion；3-substitution
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    # 生成 cost 矩阵和 operation矩阵，i:外层hyp，j:内层ref
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i-1] == reference[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i-1][j] + 1
                deletion = cost_matrix[i][j-1] + 1

                # compare_val = [insertion, deletion, substitution]   # 优先级
                compare_val = [substitution, insertion, deletion]   # 优先级

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []  # 保存 hyp与ref 中所有对齐的元素下标
    i = len_hyp
    j = len_ref
    nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:     # correct
            if i-1 >= 0 and j-1 >= 0:
                match_idx.append((j-1, i-1))
                nb_map['C'] += 1

            # 出边界后，这里仍然使用，应为第一行与第一列必然是全零的
            i -= 1
            j -= 1
        # elif ops_matrix[i_idx][j_idx] == 1:   # insert
        elif ops_matrix[i_idx][j_idx] == 2:   # insert
            i -= 1
            nb_map['I'] += 1
        # elif ops_matrix[i_idx][j_idx] == 2:   # delete
        elif ops_matrix[i_idx][j_idx] == 3:   # delete
            j -= 1
            nb_map['D'] += 1
        # elif ops_matrix[i_idx][j_idx] == 3:   # substitute
        elif ops_matrix[i_idx][j_idx] == 1:   # substitute
            i -= 1
            j -= 1
            nb_map['S'] += 1

        # 出边界处理
        if i < 0 and j >= 0:
            nb_map['D'] += 1
        elif j < 0 and i >= 0:
            nb_map['I'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    nb_map["W"] = wrong_cnt

    score = (nb_map["S"]+nb_map["D"]+nb_map["I"])/len_ref
    return score








def ppl(candidate):
    inputs = tokenizer.encode_plus(candidate, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)

    logits = outputs.logits
    num_words = input_ids.size(1) 
    log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
    total_log_prob = torch.gather(log_prob, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=1)

    perplexity = torch.exp(-total_log_prob / num_words)

    return perplexity.item()


def ppl1(text):
    text = [text]
    inputs = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    bs, sl = inputs['input_ids'].size()
    outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs[1]
    # Shift so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs['input_ids'][:, 1:].contiguous()
    shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    ppl = torch.exp(meanloss).numpy().tolist()
    return ppl







def distinct(text, n=3):
    # tokens = word_tokenize(text)
    tokens = list(jieba.cut(text))
    ngram_counts = nltk.ngrams(tokens, n)
    unique_ngrams = set(ngram_counts)
    ngram_counts = list(nltk.ngrams(tokens, n))
    diversity = len(unique_ngrams) / len(ngram_counts)
    return diversity




def BLEU(reference,candidate):
    sent1_list = list(jieba.cut(reference))
    sent2_list = list(jieba.cut(candidate))
    score = sentence_bleu(sent1_list, sent2_list, weights=(1, 0, 0, 0),smoothing_function=smooth.method1)
    return score  

# def EAS(reference,candidate):
#     embeddings1 = model.encode(reference)
#     embeddings2 = model.encode(candidate)
#     score = cos_sim(embeddings1/2, embeddings2/2)
#     return float(score)


def self_BLEU(path,type):
    with open(path, 'r', encoding='utf8') as f:
         data1 = f.read()
    list1 = json.loads(data1)
    if type == 1:
        corpus = [item['ques']+item['choices'] for item in list1]
    elif type == 2:
        corpus = [item['ques']+item['answer'] for item in list1]
    elif type == 3:
        corpus = [item['material']+item['ques']+item['answer'] for item in list1]

    score = 0.0
    cnt = 0
    length = len(corpus)

    for index in range(length):
        curr_text = corpus[index]    # string
        other_text = corpus[:index] + corpus[index + 1:]    # list
        temp_score = 0
        for item in other_text:
            temp_score += BLEU(curr_text,item)
        
        temp_score = temp_score / len(other_text)
        score += temp_score
        cnt += 1

    return score / cnt









def BLEU(reference,candidate):
    sent1_list = list(jieba.cut(reference))
    sent2_list = list(jieba.cut(candidate))
    score = sentence_bleu(sent1_list, sent2_list, weights=(1, 0, 0, 0),smoothing_function=smooth.method1)
    return score  

def EAS(reference,candidate):
    embeddings1 = model.encode(reference)
    embeddings2 = model.encode(candidate)
    score = cos_sim(embeddings1/2, embeddings2/2)
    return float(score)


def self_BLEU2(ref_path,cand_path,type):
    with open(ref_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    ref_list = json.loads(data1)
    with open(cand_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    cand_list = json.loads(data1)
    if type == 1:
        ref_corpus = [item['ques']+item['choices'] for item in ref_list]
        cand_corpus = [item['ques']+item['choices'] for item in cand_list]

    elif type == 2:
        ref_corpus = [item['ques']+item['answer'] for item in ref_list]
        cand_corpus = [item['ques']+item['answer'] for item in cand_list]
    elif type == 3:
        ref_corpus = [item['material']+item['ques']+item['answer'] for item in ref_list]
        cand_corpus = [item['material']+item['ques']+item['answer'] for item in cand_list]

    score = 0.0
    cnt = 0
    
    for ref in ref_corpus:
        for cand in cand_corpus:
            score += BLEU(ref,cand)
            cnt += 1

    return score / cnt


def bert(reference,candidate):
    P, R, F1 = bert_score.score([candidate], [reference], lang="zn", model_type = "bert-base-chinese")
    print(float(F1))
    return float(F1)




# def ppl(sent1,sent2):
#     perplexity = evaluate.load("perplexity", module_type="metric")
#     results = perplexity.compute(references=sent1,predictions=sent2)
#     print(results)
#     return results



def sent_score(ref,cand,type,method):

    if method == 'EAS':
        if type == 1:
            ref1 = ref['ques']
            cand1 = cand['ques']
            ref2 = ref['choices']
            cand2 = cand['choices']
            score1 = EAS(ref1,cand1)
            score2 = EAS(ref2,cand2)
            score = (score1 + score2)/2
    
        elif type == 2:
            ref1 = ref['ques']
            cand1 = cand['ques']
            ref2 = ref['answer']
            cand2 = cand['answer']
            score1 = EAS(ref1,cand1)
            score2 = EAS(ref2,cand2)
            score = (score1 + score2)/2

        elif type == 3:
            ref1 = ref['ques']
            cand1 = cand['ques']
            ref2 = ref['answer']
            cand2 = cand['answer']
            ref3 = ref['material']
            cand3 = cand['material']
            score1 = EAS(ref1,cand1)
            score2 = EAS(ref2,cand2)
            score3 = EAS(ref3,cand3)
            score = (score1 + score2 + score3)/3
    elif method == 'WER':
        if type == 1:
            ref1 = ref['ques']
            cand1 = cand['ques']
            ref2 = ref['choices']
            cand2 = cand['choices']
            score1 = WER(ref1,cand1)
            score2 = WER(ref2,cand2)
            score = (score1 + score2)/2
    
        elif type == 2:
            ref1 = ref['ques']
            cand1 = cand['ques']
            ref2 = ref['answer']
            cand2 = cand['answer']
            score1 = WER(ref1,cand1)
            score2 = WER(ref2,cand2)
            score = (score1 + score2)/2

        elif type == 3:
            ref1 = ref['ques']
            cand1 = cand['ques']
            ref2 = ref['answer']
            cand2 = cand['answer']
            ref3 = ref['material']
            cand3 = cand['material']
            score1 = WER(ref1,cand1)
            score2 = WER(ref2,cand2)
            score3 = WER(ref3,cand3)
            score = (score1 + score2 + score3)/3

    else:
        if type == 1:
            candidate = cand['ques'] + cand['choices']
            reference = ref['ques'] + ref['choices']
        elif type == 2:
            candidate = cand['ques'] + cand['answer']
            reference = ref['ques'] + ref['answer']
        elif type == 3:
            candidate = cand['material']+cand['ques'] + cand['answer']
            reference = ref['material']+ ref['ques'] + ref['answer']

        if method == 'distinct':
            score = distinct(candidate)
        elif method == 'ppl':
            score = ppl(candidate)
        # elif method == 'WER':
        #     score = WER(reference,candidate)

    

    


    
    return score



    
  


def file_EAS(type,num,ref_path,cand_path,method):
    
    with open(ref_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    ref_list = json.loads(data1)
    with open(cand_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    cand_list = json.loads(data1)
    score_list = []
    

    for i in range(num):
        score = sent_score(ref_list[i],cand_list[i],type,method)
        score_list.append(score)
    mean_score = mean(score_list)
    # print(cand_path,'   ',method,"分数",str(mean_score))
    return mean_score



# F1,BLEU,Distinct
def file_F1(type,num,ref_path,cand_path):
    with open(ref_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    ref_list = json.loads(data1)
    with open(cand_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    cand_list = json.loads(data1)
    if type == 1:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['choices'] for item in ref_list]
        cand2 = [item['choices'] for item in cand_list]
        list1,list2 = [],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])

        score1 = calc_f1(list1)
        score2 = calc_f1(list2)
        score = (score1+score2)/2

    elif type == 2:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['answer'] for item in ref_list]
        cand2 = [item['answer'] for item in cand_list]
        list1,list2 = [],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])

        score1 = calc_f1(list1)
        score2 = calc_f1(list2)
        score = (score1+score2)/2
    else:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['answer'] for item in ref_list]
        cand2 = [item['answer'] for item in cand_list]
        ref3 = [item['material'] for item in ref_list]
        cand3 = [item['material'] for item in cand_list]
        list1,list2,list3 = [],[],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])
            list3.append([list(jieba.cut(ref3[i])),list(jieba.cut(cand3[i]))])

        score1 = calc_f1(list1)
        score2 = calc_f1(list2)
        score3 = calc_f1(list3)
        score = (score1+score2+score3)/3
    return score

def file_BLUE(type,num,ref_path,cand_path,n=1):
    with open(ref_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    ref_list = json.loads(data1)
    with open(cand_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    cand_list = json.loads(data1)
    if type == 1:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['choices'] for item in ref_list]
        cand2 = [item['choices'] for item in cand_list]
        list1,list2 = [],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])

        score1 = calc_bleu(list1)
        score2 = calc_bleu(list2)
        if n==1:
            score = (score1[0]+score2[0])/2
        elif n==2:
            score = (score1[1]+score2[1])/2


    elif type == 2:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['answer'] for item in ref_list]
        cand2 = [item['answer'] for item in cand_list]
        list1,list2 = [],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])

        score1 = calc_bleu(list1)
        score2 = calc_bleu(list2)
        if n==1:
            score = (score1[0]+score2[0])/2
        elif n==2:
            score = (score1[1]+score2[1])/2
    else:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['answer'] for item in ref_list]
        cand2 = [item['answer'] for item in cand_list]
        ref3 = [item['material'] for item in ref_list]
        cand3 = [item['material'] for item in cand_list]
        list1,list2,list3 = [],[],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])
            list3.append([list(jieba.cut(ref3[i])),list(jieba.cut(cand3[i]))])

        score1 = calc_bleu(list1)
        score2 = calc_bleu(list2)
        score3 = calc_bleu(list3)
        if n==1:
            score = (score1[0]+score2[0]+score3[0])/3
        elif n==2:
            score = (score1[1]+score2[1]+score3[1])/3
        
    return score


# F1,BLEU,Distinct
def file_F1(type,num,ref_path,cand_path):
    with open(ref_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    ref_list = json.loads(data1)
    with open(cand_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    cand_list = json.loads(data1)
    if type == 1:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['choices'] for item in ref_list]
        cand2 = [item['choices'] for item in cand_list]
        list1,list2 = [],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])

        score1 = calc_f1(list1)
        score2 = calc_f1(list2)
        score = (score1+score2)/2

    elif type == 2:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['answer'] for item in ref_list]
        cand2 = [item['answer'] for item in cand_list]
        list1,list2 = [],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])

        score1 = calc_f1(list1)
        score2 = calc_f1(list2)
        score = (score1+score2)/2
    else:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['answer'] for item in ref_list]
        cand2 = [item['answer'] for item in cand_list]
        ref3 = [item['material'] for item in ref_list]
        cand3 = [item['material'] for item in cand_list]
        list1,list2,list3 = [],[],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])
            list3.append([list(jieba.cut(ref3[i])),list(jieba.cut(cand3[i]))])

        score1 = calc_f1(list1)
        score2 = calc_f1(list2)
        score3 = calc_f1(list3)
        score = (score1+score2+score3)/3
    return score

def file_Distinct(type,num,ref_path,cand_path,n=1):
    with open(ref_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    ref_list = json.loads(data1)
    with open(cand_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    cand_list = json.loads(data1)
    if type == 1:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['choices'] for item in ref_list]
        cand2 = [item['choices'] for item in cand_list]
        list1,list2 = [],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])

        score1 = calc_distinct(list1)
        score2 = calc_distinct(list2)
        if n==1:
            score = (score1[0]+score2[0])/2
        elif n==2:
            score = (score1[1]+score2[1])/2


    elif type == 2:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['answer'] for item in ref_list]
        cand2 = [item['answer'] for item in cand_list]
        list1,list2 = [],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])

        score1 = calc_distinct(list1)
        score2 = calc_distinct(list2)
        if n==1:
            score = (score1[0]+score2[0])/2
        elif n==2:
            score = (score1[1]+score2[1])/2
    else:
        ref1 = [item['ques'] for item in ref_list]
        cand1 = [item['ques'] for item in cand_list]
        ref2 = [item['answer'] for item in ref_list]
        cand2 = [item['answer'] for item in cand_list]
        ref3 = [item['material'] for item in ref_list]
        cand3 = [item['material'] for item in cand_list]
        list1,list2,list3 = [],[],[]
        for i in range(num):
            list1.append([list(jieba.cut(ref1[i])),list(jieba.cut(cand1[i]))])
            list2.append([list(jieba.cut(ref2[i])),list(jieba.cut(cand2[i]))])
            list3.append([list(jieba.cut(ref3[i])),list(jieba.cut(cand3[i]))])

        score1 = calc_distinct(list1)
        score2 = calc_distinct(list2)
        score3 = calc_distinct(list3)
        if n==1:
            score = (score1[0]+score2[0]+score3[0])/3
        elif n==2:
            score = (score1[1]+score2[1]+score3[1])/3
        
    return score






# 计算文件之间的得分  EAS F1 BLEU Distinct
# 即每个三元组之间的得分的平均值
def all(method):  
    path1_list = ['choice','shot','analyse']
    path2_list = ['one_','many_']
    path3_list =['_none','_key','_CIL']
    num_list = [100,100,66]
    score_list = []
  
    for i in range(3):
        ref_path = 'value_ques2/' + path1_list[i] + '.json'
        for j in range(2):
            for k in range(3):
                cand_path = 'value_ques2/'+ path2_list[j]+ path1_list[i]+path3_list[k] + '.json'
                type = i + 1
                num = num_list[i]
                 
                if method == 'self_BLEU':
                    score = self_BLEU2(ref_path,cand_path,type)
                elif method == 'F1':
                    score = file_F1(type,num,ref_path,cand_path)
                elif method == 'BLEU':
                    score = file_BLUE(type,num,ref_path,cand_path,n=2)
                elif method == 'Distinct':
                    score = file_Distinct(type,num,ref_path,cand_path,n=1)
                else:                
                    score = file_EAS(type,num,ref_path,cand_path,method)
                print(cand_path,'   ',method,"分数",str(score))
                   
                score_list.append(score)
    return score_list
 





if __name__ == '__main__':
    start = time.time()  
    num = 100
    method = 'Distinct'
    list1 = all(method)
    print(list1)  
    
    end = time.time()
    print ("时间",str((end-start)/60))
         