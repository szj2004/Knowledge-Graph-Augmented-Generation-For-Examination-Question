# -- coding:utf-8 --
"""
对数据库进行操作，插入+查找
"""
import pickle
import traceback
import pymysql
import math
import sum_score
import random
import time
import pynvml
import os

host='192.168.21.139'
user='fwq'
password='123456'
database='paper'
charset='utf8'



# 将知识点导入数据库
def insert(path):
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    lists = []
    print("数据条数",len(lines))
    for line in lines:
        line = line[1:-2]
        line = line.split(',')
        lists.append(line)
    
     # 连接数据库
    conn = pymysql.connect(
    host=host,
    user=user,
    password=password,
    database=database,
    charset=charset,
    autocommit=True,    # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
    cursor= conn.cursor()
    # 插入sql语句
    sql = "insert into relationship (id,index1,time1,person,thought,entity1,relation,entity2) values (%s,%s,%s,%s,%s,%s,%s,%s)"
    for (i,item) in enumerate(lists): 
        id = i
        index1 = item[0]
        time1 = item[1][2:-1]
        person = item[2][2:-1]
        thought = item[3][2:-1]
        entity1 = item[4][2:-1]
        relation = item[5][2:-1]
        entity2 = item[6][2:-1]
        parm = (id,index1,time1,person,thought,entity1,relation,entity2)
        # 执行插入操作
        try:           
            cursor.execute(sql,parm)
            conn.commit()
        except Exception as e:
            print("发生异常",i, index1,Exception)
            print(e.args)
            print("\n")
            print(traceback.format_exc())
            conn.rollback()
     
    cursor.close()
    conn.close()



def select_id(id):
    # 连接数据库
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        charset=charset,
        autocommit=True,  # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
    cursor= conn.cursor()
    sql="select * from relationship where id='"+ str(id) + "'"
    try:
        cursor.execute(sql)
        data = cursor.fetchone()
        
    except Exception as e:
            print("发生异常",Exception)
            print(e.args)
            print("\n")
            print(traceback.format_exc())
    cursor.close()
    conn.close()
    return data

# 寻找和文本text相关的知识点
# type   1 ：查找实体一：2：查找实体二
# equal  True 精准匹配； False 模糊查询
def find_entity(text,type,equal):
    data_list = []
    # 连接数据库
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        charset=charset,
        autocommit=True,  # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
    cursor= conn.cursor()
    if type == 1:
        if equal:
            sql = "select * from relationship where entity1='" +text + "'"
        else:
            sql = "select * from relationship where entity1 like  '%" +text +"%' "
    elif type == 2:
        if equal:
            sql = "select * from relationship where entity2='" +text + "'"
        else:
            sql = "select * from relationship where entity2 like  '%" +text +"%' "
    try:
        cursor.execute(sql)
        data_list = cursor.fetchall()
        
    except Exception as e:
            print("发生异常",Exception)
            print(traceback.format_exc())

     
    cursor.close()
    conn.close()
    return data_list



# 找到所有包含text的知识点
# 在id之外的所有属性上均搜索；time1,person,thought,entity1,relation,entity2
def select_related(text):

    data_list = []
    # 连接数据库
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        charset=charset,
        autocommit=True,  # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
    cursor= conn.cursor()
    # time1,person,thought,entity1,relation,entity2
    sql="select * from relationship where person like  '%" +text +"%' or time1 like '%" +text+"%' or thought like '%" +text+"%' or entity1 like '%" +text+"%' or relation like '%" +text+"%' or entity2 like '%" +text +"%'"
    try:
        cursor.execute(sql)
        data_list = cursor.fetchall()
        
    except Exception as e:
            print("发生异常",Exception)
            print(e.args)
            print("\n")
            print(traceback.format_exc())

     
    cursor.close()
    conn.close()
    return data_list







# 从一条数据出发，获得所有和其有关的数据，只考虑得分>0.4的数据，topk + 随机抽样
# 在数据库中寻找相关三元组
def get_metarial_all(data,edge=2,k=3,is_random=True):

    id_list = [data[0]]
    res_list = [[1,data]]
    start = 0
    end = len(res_list)
    for i in range(edge):
        for item1 in res_list[start:end]:
            temp_list = select_related(item1[1][-3]) + select_related(item1[1][-1])
            for item2 in temp_list:
                if item2[0] not in id_list:                    
                    score = sum_score.list_score([data[-3],data[-1]],[item2[-3],item2[-1]])
                    if score > 0.4:
                        id_list.append(item2[0])
                        res_list.append([score,item2])
        start = end
        end = len(res_list)

    res_list.sort(key=lambda x:x[0],reverse=True)
    if is_random:
        res_list.pop(0)
        if len(res_list) > 2*k:
            res_list = res_list[0:2*k]
        if len(res_list) > k-1:
            res_list = random.sample(res_list,k-1) 
        res_list.insert(0,[1,data])
        res_list.sort(key=lambda x:x[0],reverse=True)  
    else:
        if len(res_list) > k:
            res_list = res_list[:k]
    
    return res_list


#
# 从一条kp出发，先获得所有包含kp的三元组list，再遍历该list考虑得分>0.4的数据，得分取知识点中任一部分与kp相似的最大值，最后再topk + 随机抽样
def get_kp_all(kp, edge=1, k=3, is_random=True):
    id_list = []
    res_list = []
    temp_list = select_related(kp)
    for item in temp_list:
        id_list.append(item[0])
        res_list.append([1,item])
    start = 0
    end =len(res_list)
    if end>=k:
        random.shuffle(res_list)
        return res_list[:1]
    else:
        for i in range(edge):
            for item1 in res_list[start:end]:
                temp_list = select_related(item1[1][-1]) + select_related(item1[1][-3])
                for item2 in temp_list:
                    if item2[0] not in id_list:
                        score = sum_score.kp_score(kp,item2[-6:])
                        if score > 0.4:
                            id_list.append(item2[0])
                            res_list.append([score, item2])
            start = end
            end = len(res_list)

        res_list.sort(key=lambda x: x[0], reverse=True)
        if is_random:
            if len(res_list) > 2 * k:
                res_list = res_list[0:2 * k]
            elif len(res_list) > k - 1:
                res_list = random.sample(res_list, k - 1)
            res_list.sort(key=lambda x: x[0], reverse=True)
        return res_list


# 随机挑选知识点，获得足够多的数据
def get_data():
    res_list = []
    id_list =[]
    while len(res_list) < number:
        print("第",len(res_list),"个")
        id = random.randint(0,7000)   # 挑选范围可以改
        if id not in id_list:
            data = select_id(id)
            temp = get_metarial_all(data)
            if len(temp) >=3 :   # 和最后的k有关系
                list1 = []
                for item in temp:
                    list1.append(item[1])
                    id_list.append(item[1][0])
                res_list.append(list1)
    with open('three_neighbour_kg.pkl', 'wb') as f:
        pickle.dump({'id_list':id_list,'res_list':res_list}, f)
    return id_list,res_list


import threading

def get_data2(number=2000):
    res_list = []
    id_list = []
    # with open('three_neighbour_kg.pkl', 'wb') as f:
    #     pickle.dump({'id_list': id_list, 'res_list': res_list}, f)
    for i in range(1055, 7000):
        print("第" + str(i) + "个")
        if i == 104 or i == 124 or i == 157:
            pass
        else:
            data = select_id(i)
            # 如果超过30秒。就跳到下一个。
            import threading
            import time
            def timeout_handler():
                raise TimeoutError("Operation timed out!")
            try:
                timer = threading.Timer(5, timeout_handler)
                timer.start()
                temp=get_metarial_all(data)
                timer.join()
            except TimeoutError as e:
                print("Timeout occurred.")
            finally:
                timer.cancel()
            # 主程序中调用包装后的函数

            if len(temp) >= 3:
                with open('three_neighbour_kg.pkl', 'rb') as f:
                    loaded_data = pickle.load(f)
                    id_list = loaded_data['id_list']
                    res_list = loaded_data['res_list']
                list1 = []
                item=temp[0]
                list1.append(item[1])
                id_list.append(item[1][0])
                res_list.append(list1)
                with open('three_neighbour_kg.pkl', 'wb') as f:
                    pickle.dump({'id_list': id_list, 'res_list': res_list}, f)


#user_list 是最开始的排除list，后来补充的也会加入排除list，排除list主要是防止知识点过于相近
def get_supply(number,pc_list=[]):
    res_list = []
    id_list =[]
    while len(res_list) < number:
        score=1
        def random_data():
            id = random.randint(0,7000)   # 挑选范围可以改
            if id not in id_list:
                data = select_id(id)
            # print("start of paicha:" + str(time.time()))
            for kg in pc_list:
                if kg is None or data is None:
                    score = 0
                    break
                score = sum_score.list_score([data[-3],data[-1]],[kg[1][-3],kg[1][-1]])
                if score > 0.4:
                    break
                if score > 0.4:
                    continue
            return  data

        # print("\nend of paicha:" + str(time.time() ))


        # print("\nstart  of yanzheng:" + str(time.time() ))
        #看选中知识点有几个相似的知识点,比较多才可以后来用KAG
        #一直循环执行。直到执行成功，但每次循环时如果超过30秒就会continu显示状态变量在要执行的程序下面，只有要执行的程序在规定时间内执行完。才会改变状态变量。破除循环
        import threading
        import time
        def timeout_handler():
            raise TimeoutError("Operation timed out!")
        s = 1
        while s:
            timer = threading.Timer(30, timeout_handler)  # 5秒超时
            timer.start()
            try:
                data= random_data()
                temp=get_metarial_all(data)
                s=0
            finally:
                timer.cancel()



        if len(temp) >= 3:   # 这里的k要和 get_metarial_all方法中的k对应
            list1 = temp[0]
            res_list.append(list1)
            pc_list.append(list1)
        # print("\nend of yanzheng :" + str(time.time() ))
    return res_list


def write1(path, data):
    with open(path, 'w', encoding='utf8') as f:
        for line in data:
            f.write(str(line))
            f.write('\n')






if __name__ == '__main__':
    # with open('three_neighbour_kg.pkl', 'rb') as f:
    #     loaded_data = pickle.load(f)
    #     id_list = loaded_data['id_list']
    #     res_list = loaded_data['res_list']
    # print(id_list)
    # select_id(3)



    def set_gpu():
        try:
            # 初始化 NVIDIA 管理库
            pynvml.nvmlInit()
            # 获取 GPU 设备数量
            gpu_count = pynvml.nvmlDeviceGetCount()
            min_utlization = 100
            idlist = []
            min_id = 1;
            # 遍历所有 GPU 设备
            for gpu_id in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                # 获取 GPU 信息
                gpu_utilization = (int)(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                if (gpu_utilization < min_utlization):
                    min_utlization = gpu_utilization
                    min_id = gpu_id
                if (gpu_utilization < 20):
                    idlist.append(gpu_id)
            # 设置
            if (len(idlist) == 0):
                idlist.append(min_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, idlist))
            # 释放 NVIDIA 管理库
            pynvml.nvmlShutdown()
            return min_id, idlist
        except pynvml.NVMLError as error:
            print(f"NVIDIA Management Library Error: {error}")
    set_gpu()
    print(get_metarial_all(select_id(128)))
