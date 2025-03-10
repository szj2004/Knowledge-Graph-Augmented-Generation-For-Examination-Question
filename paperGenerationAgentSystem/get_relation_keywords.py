# -- coding:utf-8 --
'''
根据关键词进行关系提取，处理结果
保存：id、文本位置、时间、思想、三元组
关注*开始
处理关系识别的结果，转变为list
'''

import pickle

from baidu import api_answer

# 获得文心一言的关系抽取结果
def get_content(in_path, out_path):
    list = []
    with open(out_path, 'w', encoding='utf8') as f2:
        with open(in_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for (i, item) in enumerate(lines):
                if len(item) != 0:
                    instance = "QQQ:距今约5000年的新石器时代晚期，大汶口文化和仰韶文化被龙山文化所代替。龙山文化的代表器物是黑陶，胎壁薄如蛋壳，被称为“蛋壳陶”。同时，在北方辽河上游有红山文化，长江下游有良渚文化。请首先提取文本关于时间、人物和思想的关键词，然后对关键词进行关系提取，输出三元组形式。\nAAA:关键词提取：\n时间：距今约5000年、新石器时代晚期\n人物：无\n思想：无\n关系提取（三元组形式）：\n(新石器时代晚期, 时间, 距今约5000年)\n(龙山文化, 代表器物, 黑陶)\n(辽河上游, 文化, 红山文化)\n(长江下游, 文化, 良渚文化)"
                    item = instance + "\nQQQ:" + item + "请首先提取文本关于时间、人物和思想的关键词，然后对关键词进行关系提取，输出三元组形式。\n AAA:"
                    res = 'aaa' + str(i) +'\n' + api_answer(item) + '\n\n'
                    f2.write(res)
                    list.append(res)
            print(len(list))






# 获得文心一言的关系抽取结果
# def get_result(in_path, out_path):
#     list = []
#     with open(out_path, 'w', encoding='utf8') as f2:
#         with open(in_path, 'r', encoding='utf8') as f:
#             lines = f.readlines()
#             for (i, item) in enumerate(lines):
#                 if len(item) != 0:
#                     instance = "Q:距今约5000年的新石器时代晚期，大汶口文化和仰韶文化被龙山文化所代替。龙山文化的代表器物是黑陶，胎壁薄如蛋壳，被称为“蛋壳陶”。同时，在北方辽河上游有红山文化，长江下游有良渚文化。请首先提取文本关于时间、人物和思想的关键词，然后对关键词进行关系提取，输出三元组形式。\nA:关键词提取：\n时间：距今约5000年、新石器时代晚期\n人物：无\n思想：无\n关系提取（三元组形式）：\n(新石器时代晚期, 时间, 距今约5000年)\n(龙山文化, 代表器物, 黑陶)\n(辽河上游, 文化, 红山文化)\n(长江下游, 文化, 良渚文化)"
#                     item = instance + "Q:" + item + "请首先提取文本关于时间、人物和思想的关键词，然后对关键词进行关系提取，输出三元组形式。\n"
#                     res = 'aaa' + str(i) +'\n' + api_answer(item) + '\n\n'
#                     f2.write(res)
#                     list.append(res)
#             print(len(list))


# 获得文心一言的关系抽取结果
# def get_choice(in_path, out_path):
#     list = []
#     with open(out_path, 'w', encoding='utf8') as f2:
#         with open(in_path, 'r', encoding='utf8') as f:
#             lines = f.readlines()
#             for (i, item) in enumerate(lines):
#                 if len(item) != 0:
#                     instance = "QQQ:“从周”是指孔子推崇的主张周代的礼制，维护有序的等级秩序，“封建亲戚，以藩屏周”是周代的分封制，维护了等级秩序，A项正确；“海内为郡县，法令由一统”是战国以后的郡县制，排除B项；“罢黜百家，独尊儒术”是西汉董仲舒的儒学独尊主张，排除C项；“上品无寒门，下品无世族”涉及魏晋时期的九品中正制，排除D项。故选A项。请首先提取文本关于时间、人物和思想的关键词，并输出，然后对关键词进行关系提取，输出三元组形式。\nAAA:关键词提取：\n时间：周代、战国以后、西汉、魏晋\n人物：孔子、董仲舒\n思想：推崇周代的礼制、维护有序的等级秩序、分封制、郡县制、法令一统、罢黜百家、独尊儒术、九品中正制\n关系提取（三元组形式）：\n(孔子, 推崇, 周代的礼制)\n(周代, 制度, 分封制)\n(分封制, 内容, 封建亲戚，以藩屏周)\n(分封制, 作用, 维护等级秩序)\n(郡县制, 特点, 海内为郡县，法令由一统)\n(董仲舒, 思想主张, 罢黜百家、独尊儒术)\n(魏晋, 制度, 九品中正制)\n"
#                     item = instance + "QQQ:" + item + "请首先提取这段解析关于时间、人物和思想的关键词，并输出，然后对关键词进行关系提取，输出三元组形式。\n  AAA:"
#                     # print(item)
#                     res = 'aaa' + str(i) +'\n' + api_answer(item) + '\n\n'
#                     f2.write(res)
#                     list.append(res)
#             print(len(list))


# 处理网站获得的结果
def process_relation(in_path, out_path):
    with open(in_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    i = 0
    end = len(lines)
    # print(end)
    list = []
    number = 0
    rel =  [number,'','','']
    new_rel = []
    while i < end:
        line = lines[i].strip()
        if len(line) == 0:
            i = i + 1
            continue
        elif line[:3] == 'aaa':
            rel = [number,'','','']
            new_rel = []
            number = number + 1
        elif line[:2] == '时间' and len(line) > 3 and line[3] != '无':
            rel[1]= line[3:]
        elif line[:2] == '人物' and len(line) > 3 and line[3] != '无':
            rel[2] = line[3:]
        elif line[:2] == '思想' and len(line) > 3 and line[3] != '无':
            rel[3] = line[3:]
        elif line[0] == '(':
            new_rel = rel.copy()
            line = line[1:-1]
            temp = line.split(',')
            if len(temp) == 3:
                for item in temp:
                    new_rel.append(item.strip())
                if new_rel[-3] != new_rel[-1] and new_rel[-1]!='无' and new_rel[-3]!='无':
                    list.append(new_rel)
                # print(new_rel)
        elif line[:3] == '* (':
            new_rel = rel.copy()
            line = line[4:-1]
            temp = line.split(',')
            if len(temp) == 3:
                for item in temp:
                    new_rel.append(item.strip())
                if new_rel[-3] != new_rel[-1] and new_rel[-1]!='无' and new_rel[-3]!='无':
                    list.append(new_rel)
                # print(new_rel)
        
        i = i + 1
       
  
    with open(out_path, 'w', encoding='utf8') as f1:
        for line in list:
            f1.write(str(line))
            f1.write('\n')
   

if __name__ == '__main__':
    # 获得结果
    in_path = 'real_paper/material_detail.txt'
    out_path = 'a1.txt'
    get_content(in_path,out_path)
    # in_path = 'real_paper/choice_detail.txt'
    # out_path = 'key_relation/choice.txt'
    # get_choice(in_path,out_path)

    # 处理结果
    # in_path = 'data1/api_res.txt'
    # out_path = 'data1/relation.txt'
    # process_relation(in_path, out_path)


