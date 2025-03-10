# -- coding:utf-8 --
import json
import time
from deepseek import api_answer

# 运行问题，baidu文件夹的id需要改成新的


# 选择题生成的所有三元组
with open('value2/choice_all_triple.json', 'r', encoding='utf8') as f:
    data1 = f.read()
choice_list = json.loads(data1)

# # 选择题生成的所有三元组
# with open('value2/analyse_triple.json', 'r', encoding='utf8') as f:
#          data1 = f.read()
# analyse_list = json.loads(data1)

# 选择题的相关子图
with open('value2/choice_subKG.json', 'r', encoding='utf8') as f:
    data1 = f.read()
choice_subKG = json.loads(data1)

# 选择题的相关子图
# with open('value1/analyse_subKG.json', 'r', encoding='utf8') as f:
#          data1 = f.read()
# analyse_subKG = json.loads(data1)


"""
i 题目的顺序
id 生成题目三元组的id
method  Rand,KP,KKP,KAG
"""


def get_rel(i, id, method, type=3):
    if type == 1:
        triple_list = choice_list
        subKG = choice_subKG
    elif type == 3:
        triple_list = analyse_list
        subKG = analyse_subKG

    for item in triple_list:
        if item['id'] == id:
            data = item
            break

    if method == 'Rand':
        res_list = [{'实体': data['entity1']}]
    elif method == 'KP':
        rel = '(' + data["entity1"] + ',' + data["relation"] + ',' + data["entity2"] + ')'
        res_list = [{'关系': rel}]
    elif method == 'KKP':
        rel = '(' + data["entity1"] + ',' + data["relation"] + ',' + data["entity2"] + ')'
        res_list = [{'时间': data["time1"], '人物': data["person"], '思想': data["thought"], '关系': rel}]

    elif method == 'KAG':
        # rel_list = get_metarial_all(data)
        rel_list = subKG[i]
        res_list = []
        for temp in rel_list:
            res = {}
            res["时间"] = temp["time1"]
            res["人物"] = temp["person"]
            res["思想"] = temp["thought"]
            res["关系"] = '(' + temp["entity1"] + ',' + temp["relation"] + ',' + temp["entity2"] + ')'
            res_list.append(res)

    return data, res_list


"""
根据数据获得添加样例的prompt
data  get_rel结果
method  Rand,KP,KKP,KAG
type  1选择题  2简答题 3材料题
"""


def get_ICL(data, method, type):
    if method == 'KAG':
        if type == 1:
            content = "QQQ：\n输入：[{'时间': '西周', '人物': '', '思想': '西周社会结构特点通过垄断神权强化王权', '关系': '(《左传•昭公七年》,反映, 西周社会结构)'}]\n参考资料：[{'时间': '西周', '人物': '《左传•昭公七年》', '思想': '', '关系': '(《左传•昭公七年》,内容,天有十日，人有十等。下所以事上，上所以共神也。故王臣公，公臣大夫，大夫臣士，士臣皂 )'}]\n任务定义：请以输入为中心，根据参考资料和相关历史知识设计1道包含四个选项的单选题，并给出答案。\nAAA:\n题目：《左传•昭公七年》：“天有十日，人有十等。下所以事上，上所以共神也。故王臣公，公臣大夫，大夫臣士，士臣皂”，上述材料反映西周社会结构的基本特点是？\nA. 通过垄断神权强化王权\nB. 嫡长子拥有继承特权\nC. 严格的等级关系\nD. 血缘纽带和政治关系紧密结合\n答案：C\n\nQQQ：\n输入："
            content = content + str([data[0]]) + "\n参考资料：" + str(
                data[1:]) + "\n任务定义：请以输入为中心，根据参考资料和相关历史知识设计1道包含四个选项的单选题，并给出答案。\nAAA:\n"
        if type == 2:
            content = "QQQ：[{'时间': '春秋战国时期', '人物': '孔子、其他诸子', '思想': '百家争鸣的背景', '关系': '(百家争鸣, 社会根源, 春秋战国时期的社会动荡与政治、经济大变动)'},{'时间': '春秋战国时期', '人物': '孔子、其他诸子', '思想': '百家争鸣的背景', '关系': '(百家争鸣, 阶层背景, 旧的贵族等级体系瓦解，新兴的士阶层崛起)'},{'时间': '春秋战国时期', '人物': '孔子、其他诸子', '思想': '百家争鸣的背景', '关系': '(百家争鸣, 思想影响, 统治者争相招揽人才，寻找治国新思想)}]请根据这些信息设计1道简答题，并给出答案。\nAAA：问题：请简述百家争鸣的背景。\n答案：（1）春秋战国时期巨大的社会动荡与政治、 经济大变动是“百家争鸣”形成的社会根源。\n（2）旧的贵族等级体系开始瓦解，新兴的士族阶层崛起，提出自己的政治主张，试图影响现实政治。\n（3）统治者出于争霸需要，争相招揽人才，寻找治国新思想。\n（4）文化教育方面，私学兴起，改变“学在官府”的局面。其中孔子影响最大，其他诸子各有特色。\n\nQQQ："
            content = content + str(data) + '请根据这些信息设计1道简答题，并给出答案。\n AAA:'
        if type == 3:
            content = "QQQ：[{'时间': '乾元元年', '人物': '第五琦、刘晏', '思想': '榷盐法', '关系': '(第五琦, 推行, 榷盐法)},{'时间': '乾元元年', '人物': '第五琦、刘晏', '思想': '榷盐法', '关系': '(刘晏, 改革, 榷盐法)'},{'时间': '乾元元年', '人物': '第五琦、刘晏', '思想': '榷盐法', '关系': '(榷盐法, 不同之处, 官营专卖、官府专利、强行加之)'},{'时间': '乾元元年', '人物': '第五琦、刘晏', '思想': '榷盐法', '关系': '(榷盐法, 意义, 促进盐业发展、增加政府财政收入、利商便民、有助于社会稳定)'}]请根据这些信息生成2个材料，并根据生成的材料再生成2个问题，并给出答案。\nAAA：\n材料一：乾元元年（ 758），盐铁铸钱使第五琦初变盐法，就山海井灶近利之地，置盐院。游民业盐者为亭户，免杂徭。盗鬻者论以法。及琦为诸州榷盐铁使，尽榷天下盐，斗加时价百钱而出之，为钱一百一十。\n材料二：刘晏改革榷盐法，调整官营与私商、盐户的关系。在产盐乡 “因旧监置吏 ”，收亭户之盐 ，转卖给商人经销 。其余州县不设盐官 ，在较远州县设置 “常平盐”，“官收厚利而人不知责 ”。刘晏改革是以官商分利代替官方专利，促进了盐业的发展，大大增加了盐税收入。刘晏始榷盐时，盐利年收入40万缗，其后，达600万缗， “天下之赋，盐利过半 ”。\n问题一：根据材料一、二，指出第五琦和刘晏所推行的榷盐法的不同之处。\n答案:不同之处，第五琦：官营专卖，官府专利，强行加之。刘晏：官署民产商销，官商分利：设常平盐。\n问题二：根据材料二，说明刘晏改革榷盐法的意义。\n答案：促进盐业发展：增加政府财政收入：利商，便民；有助于社会稳定。\n\n\nQQQ"
            content = content + str(data) + '请根据这些信息生成2个材料，并根据生成的材料再生成2个问题，并给出答案。\nAAA:'
    elif method == 'KKP':
        if type == 1:
            content = "QQQ:[{'时间': '西周', '人物': '', '思想': '西周社会结构特点通过垄断神权强化王权', '关系': '(西周社会结构, 特点, 通过垄断神权强化王权)'}]根据这些信息设计1道包含四个选项的单选题，并给出答案。\nAAA:\n题目：《左传•昭公七年》：“天有十日，人有十等。下所以事上，上所以共神也。故王臣公，公臣大夫，大夫臣士，士臣皂”，上述材料反映西周社会结构的基本特点是？\nA. 通过垄断神权强化王权\nB. 嫡长子拥有继承特权\nC. 严格的等级关系\nD. 血缘纽带和政治关系紧密结合\n答案：C\n\nQQQ："
            content = content + str(data) + '根据这些信息设计1道包含四个选项的单选题，并给出答案。\n AAA:'
        if type == 2:
            content = "QQQ：[{'时间': '春秋战国时期', '人物': '孔子、其他诸子', '思想': '百家争鸣的背景', '关系': '(百家争鸣, 社会根源, 春秋战国时期的社会动荡与政治、经济大变动)}]请根据这些信息设计1道简答题，并给出答案。\nAAA：问题：请简述百家争鸣的背景。\n答案：（1）春秋战国时期巨大的社会动荡与政治、 经济大变动是“百家争鸣”形成的社会根源。\n（2）旧的贵族等级体系开始瓦解，新兴的士族阶层崛起，提出自己的政治主张，试图影响现实政治。\n（3）统治者出于争霸需要，争相招揽人才，寻找治国新思想。\n（4）文化教育方面，私学兴起，改变“学在官府”的局面。其中孔子影响最大，其他诸子各有特色。\n\nQQQ："
            content = content + str(data) + '请根据这些信息设计1道简答题，并给出答案。\n AAA:'
        if type == 3:
            content = "QQQ：[{'时间': '乾元元年', '人物': '第五琦、刘晏', '思想': '榷盐法', '关系': '(榷盐法, 意义, 促进盐业发展、增加政府财政收入、利商便民、有助于社会稳定)'}]请根据这些信息生成2个材料，并根据生成的材料再生成2个问题，并给出答案。\nAAA：\n材料一：乾元元年（ 758），盐铁铸钱使第五琦初变盐法，就山海井灶近利之地，置盐院。游民业盐者为亭户，免杂徭。盗鬻者论以法。及琦为诸州榷盐铁使，尽榷天下盐，斗加时价百钱而出之，为钱一百一十。\n材料二：刘晏改革榷盐法，调整官营与私商、盐户的关系。在产盐乡 “因旧监置吏 ”，收亭户之盐 ，转卖给商人经销 。其余州县不设盐官 ，在较远州县设置 “常平盐”，“官收厚利而人不知责 ”。刘晏改革是以官商分利代替官方专利，促进了盐业的发展，大大增加了盐税收入。刘晏始榷盐时，盐利年收入40万缗，其后，达600万缗， “天下之赋，盐利过半 ”。 \n问题一：根据材料一、二，指出第五琦和刘晏所推行的榷盐法的不同之处。\n答案:不同之处，第五琦：官营专卖，官府专利，强行加之。刘晏：官署民产商销，官商分利：设常平盐。\n问题二：根据材料二，说明刘晏改革榷盐法的意义。\n答案：促进盐业发展：增加政府财政收入：利商，便民；有助于社会稳定。\n\n\nQQQ："
            content = content + str(data) + '请根据这些信息生成2个材料，并根据生成的材料再生成2个问题，并给出答案。\nAAA：'
    elif method == 'KP':
        if type == 1:
            content = "QQQ：[{'关系': '(西周社会结构, 特点, 通过垄断神权强化王权)'}]根据这些信息设计1道包含四个选项的单选题，并给出答案。\nAAA:\n题目：《左传•昭公七年》：“天有十日，人有十等。下所以事上，上所以共神也。故王臣公，公臣大夫，大夫臣士，士臣皂”，上述材料反映西周社会结构的基本特点是？\nA. 通过垄断神权强化王权\nB. 嫡长子拥有继承特权\nC. 严格的等级关系\nD. 血缘纽带和政治关系紧密结合\n答案：C\n\nQQQ："
            content = content + str(data) + '根据这些信息设计1道包含四个选项的单选题，并给出答案。\n AAA:'
        if type == 2:
            content = "QQQ：[{'关系': '(百家争鸣, 社会根源, 春秋战国时期的社会动荡与政治、经济大变动)}请根据这些信息设计1道简答题，并给出答案。\nAAA：问题：请简述百家争鸣的背景。\n答案：（1）春秋战国时期巨大的社会动荡与政治、 经济大变动是“百家争鸣”形成的社会根源。\n（2）旧的贵族等级体系开始瓦解，新兴的士族阶层崛起，提出自己的政治主张，试图影响现实政治。\n（3）统治者出于争霸需要，争相招揽人才，寻找治国新思想。\n（4）文化教育方面，私学兴起，改变“学在官府”的局面。其中孔子影响最大，其他诸子各有特色。\n\nQQQ："
            content = content + str(data) + '请根据这些信息设计1道简答题，并给出答案。\n AAA:'
        if type == 3:
            content = "QQQ：[{'关系': '(榷盐法, 意义, 促进盐业发展、增加政府财政收入、利商便民、有助于社会稳定)'}]请根据这些信息生成2个材料，并根据生成的材料再生成2个问题，并给出答案。\nAAA：\n材料一：乾元元年（ 758），盐铁铸钱使第五琦初变盐法，就山海井灶近利之地，置盐院。游民业盐者为亭户，免杂徭。盗鬻者论以法。及琦为诸州榷盐铁使，尽榷天下盐，斗加时价百钱而出之，为钱一百一十。\n材料二：刘晏改革榷盐法，调整官营与私商、盐户的关系。在产盐乡 “因旧监置吏 ”，收亭户之盐 ，转卖给商人经销 。其余州县不设盐官 ，在较远州县设置 “常平盐”，“官收厚利而人不知责 ”。刘晏改革是以官商分利代替官方专利，促进了盐业的发展，大大增加了盐税收入。刘晏始榷盐时，盐利年收入40万缗，其后，达600万缗， “天下之赋，盐利过半 ”。 \n问题一：根据材料一、二，指出第五琦和刘晏所推行的榷盐法的不同之处。\n答案:不同之处，第五琦：官营专卖，官府专利，强行加之。刘晏：官署民产商销，官商分利：设常平盐。\n问题二：根据材料二，说明刘晏改革榷盐法的意义。\n答案：促进盐业发展：增加政府财政收入：利商，便民；有助于社会稳定。\n\n\nQQQ："
            content = content + str(data) + '请根据这些信息生成2个材料，并根据生成的材料再生成2个问题，并给出答案。\nAAA：'
    elif method == 'Rand':
        if type == 1:
            content = "QQQ：[{'实体': '西周社会结构'}]根据这些信息设计1道包含四个选项的单选题，并给出答案。\nAAA:\n题目：《左传•昭公七年》：“天有十日，人有十等。下所以事上，上所以共神也。故王臣公，公臣大夫，大夫臣士，士臣皂”，上述材料反映西周社会结构的基本特点是？\nA. 通过垄断神权强化王权\nB. 嫡长子拥有继承特权\nC. 严格的等级关系\nD. 血缘纽带和政治关系紧密结合\n答案：C\n\nQQQ："
            content = content + str(data) + '根据这些信息设计1道包含四个选项的单选题，并给出答案。\n AAA:'
        if type == 2:
            content = "QQQ：[{'实体': '百家争鸣'}]请根据这些信息设计1道简答题，并给出答案。\nAAA：问题：请简述百家争鸣的背景。\n答案：（1）春秋战国时期巨大的社会动荡与政治、 经济大变动是“百家争鸣”形成的社会根源。\n（2）旧的贵族等级体系开始瓦解，新兴的士 阶层崛起，提出自己的政治主张，试图影 响现实政治。\n（3）统治者出于争霸需要，争相招揽人才，寻找治国新思想。\n（4）文化教育方面，私学兴起，改变“学在 官府\"的局面。其中孔子影响最大，其他诸子各有特色。\n\nQQQ："
            content = content + str(data) + '请根据这些信息设计1道简答题，并给出答案。\n AAA:'
        if type == 3:
            content = "QQQ：[{'实体': '榷盐法'}]请根据这些信息生成2个材料，并根据生成的材料再生成2个问题，并给出答案。\nAAA：材料一：乾元元年，盐铁铸钱使第五琦初变盐法，就山海井灶近利之地，置盐院。刘晏改革榷盐法，调整官营与私商、盐户的关系。在产盐乡 “因旧监置吏 ”，收亭户之盐 ，转卖给商人经销 \n问题一：根据材料一、二，指出第五琦和刘晏所推行的榷盐法的不同之处。\n答案:不同之处，第五琦：官营专卖，官府专利，强行加之。刘晏：官署民产商销，官商分利：设常平盐。\n问题二：根据材料二，说明刘晏改革榷盐法的意义。\n答案：促进盐业发展：增加政府财政收入：利商，便民；有助于社会稳定。\n\n\nQQQ："
            content = content + str(data) + '请根据这些信息生成2个材料，并根据生成的材料再生成2个问题，并给出答案。\nAAA：'

    return content


"""
获得问题生成的promot
i : 题目的顺序，主要为了获得subKG
method: Rand, KP, KKP, KAG 
type:    1 单选题; 2 简答题; 3 材料分析题
"""


def get_prompt(i, id, method, type=1):
    x, data = get_rel(i, id, method, type)
    prompt = get_ICL(data, method, type)
    return x, data, prompt


# 分割大语言模型生成的问题和答案
# text 大语言模型生成的结果
def deal(text, type):
    lines = text.split("\n")
    if type == 1:
        state = 1
        content, answer = '', ''
        for (i, line) in enumerate(lines):
            if len(line) == 0:
                continue
            if i == 0 and (line[-1] == '：' or line[-1] == ':'):
                continue

            if '答案' in line[:30]:
                state = 2

            if len(line) == 3:
                continue

            if state == 1:
                content = content + line + '\n'
            elif state == 2:
                answer = answer + line + '\n'
        content = content.strip()
        answer = answer.strip()
        choice_index = [content.find('A'), content.find('B'), content.find('C'), content.find('D')]
        A = content[choice_index[0] + 2:choice_index[1]].strip()
        B = content[choice_index[1] + 2:choice_index[2]].strip()
        C = content[choice_index[2] + 2:choice_index[3]].strip()
        D = content[choice_index[3] + 2:].strip()
        content = content[:choice_index[0]].strip()
        return content, answer, A, B, C, D

    elif type == 3:
        mat1, mat2, ques1, ques2, ans1, ans2 = '', '', '', '', '', ''
        state = 1
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            elif line[:3] == '材料一':
                mat1 = line
            elif line[:3] == '材料二':
                mat2 = line
                state = 2
            elif line[:3] == '问题一':
                ques1 = line
                state = 3
            elif line[:3] == '问题二':
                ques2 = line
                state = 4
            elif line[:2] == '答案' and ans1 == '':
                ans1 = line
                state = 5
            elif line[:2] == '答案':
                ans2 = line
                state = 6
            else:
                if state == 1:
                    mat1 = mat1 + line
                elif state == 2:
                    mat2 = mat2 + line
                elif state == 3:
                    ques1 = ques1 + line
                elif state == 4:
                    ques2 = ques2 + line
                elif state == 5:
                    ans1 = ans1 + line
                elif state == 6:
                    ans2 = ans2 + line

        return mat1, mat2, ques1, ques2, ans1, ans2


"""
id_path  出题id列表
method   算法：Rand KP KKP KAG
out_path  生成题目的json文件
show_path  生成题目的txt文件（json文件中无法根据算法形成规定格式时，根据txt文件手动修改）
type   1 选择题  2 简答题  3 材料题

更换模型时改变 api_answer算法即可
"""


# 根据三元组进行题目生成
#
# 现在是重新生成
def generate(method, id_path, out_path, show_path, type=1):

#如果是从头写
    # res_list = []
    # f2 = open(show_path, 'w', encoding='utf8')  # 便于观察

#追加方式写，继承前者
    with open(out_path, 'r', encoding='utf8') as f:
         data1 = f.read()
    res_list = json.loads(data1)
    f2 = open(show_path, 'a', encoding='utf8')  # 便于观察


    # 获得出题列表
    with open(id_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    id_list = [int(line.strip()) for line in lines]

    for (i, id) in enumerate(id_list):
        # if len(res_list) == 10:
        #     break

        if i <5 :
            continue
        # 1733在choice——id指向104，而其在数据表中不存在，直接从103跳到105.  1756同样的问题
        if i == 1733 | i == 1756:
            continue
        print(i)
        try:
            x, data, prompt = get_prompt(i, id, method, type)
            res = api_answer(prompt)  # 送入LLM
            if "问题" not in res or "答案" not in res:
                with open("log.txt","a",encoding='utf8') as f:
                    f.write(f"材料题生成出现问题\ni:{i}\nid:{id}\nres:{res}\n")
            f2.write("aaa" + str(i) + "\n")
            f2.write(res)
            f2.write("\n\n\n")

            temp = {}
            temp['id'] = i
            temp['x'] = x
            temp['data'] = data
            temp['prompt'] = prompt

            if type == 1:
                content, answer, temp['A'], temp['B'], temp['C'], temp['D'] = deal(res, type)
                temp['content'] = content
                temp['answer'] = answer[3:]
            elif type == 3:
                temp["mat1"], temp["mat2"], temp["ques1"], temp["ques2"], temp["ans1"], temp["ans2"] = deal(res, type)

            res_list.append(temp)
            data1 = json.dumps(res_list, indent=1, ensure_ascii=False)
            with open(out_path, 'w', encoding='utf8') as f:
                f.write(data1)
        except Exception as e:
            with open("log.txt", "a", encoding='utf8') as f:
                f.write(f"材料题生成出现问题\ni:{i}\nid:{id}\nException:{e}\n")



if __name__ == '__main__':
    import tool
    start = time.time()
    tool.set_gpu()
    # 运行时改变
    # 根据id出题
    id_path = 'value2/choice_id.txt'
    methods = ['Rand', 'KP', 'KKP', 'KAG']
    type = 1
    for method in methods:
        out_path = 'ques_choice/' + str(method) + '.json'
        show_path = 'ques_choice/' + str(method) + '.txt'
        generate(method, id_path, out_path, show_path, type)

    end = time.time()
    print("时间", str((end - start) / 60))








