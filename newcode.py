# pylint: disable=invalid-name
# -*- coding: utf-8 -*-
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from __future__ import print_function

f = open("./data.csv")
context = f.readlines()
train_day29 = []
offline_candidate_day30 = []
online_candidate_day31 = []

for line in context:
    line = line.replace('\n', '')  # 去除文本中的换行符
    array = line.split(',')  # 以','分割
    if array[0]  == 'user_id':  # 去除第一行
        continue
    day = int(array[-1]) # 倒数第一行是日期
    uid = (array[0], array[1], day + 1)  # 把用户名，商品号，日期+1记录在uid中
    if day == 28:  # 如果是28日，作为训练对象放入
        train_day29.append(uid)
    if day == 29:  # 如果是29日，作为线下评估对象放入
        offline_candidate_day30.append(uid)
    if day == 30:   # 如果是30日， 作为线上评估放入
        online_candidate_day31.append(uid)

train_day29 = list(set(train_day29))  # 排除重复项
offline_candidate_day30 = list(set(offline_candidate_day30))
online_candidate_day31 = list(set(online_candidate_day31))

print("training item number:\t", len(train_day29))  # 输出训练对象数
print("-----------------------\n")
print("offline candidate item number:\t", len(offline_candidate_day30))  # 输出线下评估数
print("-----------------------\n")

ui_dict = [{} for i in range(4)]  # 创建4个表
for line in context:
    line = line.replace('\n', '')
    array = line.split(',')
    if array[0] == 'user_id':
        continue
    day = int(array[-1])
    uid = (array[0], array[1], day)
    type = int(array[2]) - 1  # 查看类型 浏览、收藏、购物车、购买
    if uid in ui_dict[type]:  # 查询当前id对当前商品有无进行当前操作
        ui_dict[type][uid] += 1 # 是则在当前表中加一
    else:
        ui_dict[type][uid] = 1  # 否则在当前表中置一
# for label
ui_buy = {} # 创建一个表
for line in context:
    line = line.replace('\n', '')
    array = line.split(',')
    if array[0] == 'user_id':   # 跳过第一行
        continue
    uid = (array[0], array[1], int(array[-1]))

    if array[2] == '4': # 购买了
        ui_buy[uid] = 1 # 当前id记下来


# get train x,y
x = np.zeros((len(train_day29), 4)) # 开辟train×4的矩阵
y = np.zeros((len(train_day29)), )  # 开辟train×1的矩阵

id = 0
for uid in train_day29:
    last_uid = (uid[0], uid[1], uid[2] - 1)
    for i in range(4):
        x[id][i] = math.log1p(ui_dict[i][last_uid] if last_uid in ui_dict[i] else 0)  # 若28日该用户也进行了该操作 计算操作次数 否则0
    y[id] = 1 if uid in ui_buy else 0  # 买了是1 没买是0
    id += 1


print("x = ", x, '\n\n', 'y = ', y)
print("----------------------\n\n")
print("train number = ", len(y), "positive number = ", sum(y), "\n")

# get predict px for offline_candidate_day30
px = np.zeros((len(offline_candidate_day30), 4))
id = 0
for uid in offline_candidate_day30:
    last_uid = (uid[0], uid[1], uid[2] -1)
    for i in range(4):
        px[id][i] = math.log1p(ui_dict[i][last_uid] if last_uid in ui_dict[i] else 0)
    id += 1

# the same for online_canddate_day31




model = LogisticRegression()
model.fit(x, y)

# evaluate
py = model.predict_log_proba(px)
npy = []
for a in py:
    npy.append(a[1])  # 被判断为1的概率
py = npy    # 重新定义

print('px = ', px)

# combine
lx = zip(offline_candidate_day30, py)  # 把uid 与购买概率混合
print("--------------------------")
# sort by predict score
lx = sorted(lx, key = lambda x: x[1], reverse = True)  # 排序 概率高的放前面
print("--------------------------")

wf = open('ans.csv', 'w')
wf.write('user_id,item_id\n')
for i in range(437):
    item = lx[i]
    wf.write('%s,%s\n'%(item[0][0],item[0][1]))

wf.close()
