import os
import numpy as np
from utils import *
from multiprocessing import Pool
import time

file_path = "./data/"
dataset = "ICEWS05-15/"
# dataset = "YAGO-WIKI50K/"
ts = time.time()
print("unsup_seeds_generation_start:"+str(ts))


def load_dict(file_path):

    train_pair = load_alignment_pair(file_path + 'sup_pairs')
    print("train_pair_len: "+str(len(train_pair)))
    dev_pair = load_alignment_pair(file_path + 'ref_pairs')
    print("dev_pair_len: "+str(len(dev_pair)))
    all_pair = train_pair + dev_pair
    print("all_pair_len: "+str(len(all_pair)))

    entity1,rel1,time1,quadruples1 = load_quadruples(file_path + 'triples_1')
    print("quadruples1_entity_len: "+str(len(entity1)))
    print("quadruples1_rel_len: "+str(len(rel1)))
    print("quadruples1_time_len: "+str(len(time1)))
    print("quadruples1_len: "+str(len(quadruples1)))
    time_point1 = {}
    time_interval1 = {}

    for i in entity1:
        time_point1[i] = []
        time_interval1[i] = []

    for h, r, t, ts, te in quadruples1:
        if ts == te:
            time_point1[h].append(ts);
            time_point1[t].append(ts)
        elif ts == 0:
            time_point1[h].append(te);
            time_point1[t].append(te)
        elif te == 0:
            time_point1[h].append(ts);
            time_point1[t].append(ts)
        else:
            time_interval1[h].append([ts, te]);
            time_interval1[t].append([ts, te])

    entity2,rel2,time2,quadruples2 = load_quadruples(file_path + 'triples_2')
    print("quadruples2_entity_len: "+str(len(entity2)))  
    print("quadruples2_rel_len: "+str(len(rel2)))
    print("quadruples2_time_len: "+str(len(time2)))
    print("quadruples2_len: "+str(len(quadruples2)))
    time_point2 = {}
    time_interval2 = {}

    for i in entity2:
        time_point2[i] = []
        time_interval2[i] = []

    for h, r, t, ts, te in quadruples2:
        if ts == te:
            time_point2[h].append(ts);
            time_point2[t].append(ts)
        elif ts == 0:
            time_point2[h].append(te);
            time_point2[t].append(te)
        elif te == 0:
            time_point2[h].append(ts);
            time_point2[t].append(ts)
        else:
            time_interval2[h].append([ts, te]);
            time_interval2[t].append([ts, te])

    return all_pair,entity1,entity2,time_point1,time_interval1,time_point2,time_interval2


all_pair, entity1, entity2, time_point1,time_interval1,time_point2,time_interval2 = load_dict(file_path+dataset)

thread_num =18

file_name1 = file_path+dataset+"simt/simt_" + dataset[0:-1] + "_ab.npy"
file_name2 = file_path+dataset+"simt/simt_" + dataset[0:-1] + "_ba.npy"
if os.path.exists(file_name1):
    m1 = np.load(file_name1)
    m2 = np.load(file_name2)
else:
    list1 = []
    for k in set(time_point1.keys()) | set(time_interval1.keys()):
        # 处理时间点
        tp1 = time_point1.get(k, [])
        list1.append(list2dict(tp1))
        # 处理时间间隔
        ti1 = time_interval1.get(k, [])
        list1[-1].update(list2dict(ti1))

    list2 = []
    for k in set(time_point2.keys()) | set(time_interval2.keys()):
        # 处理时间点
        tp2 = time_point2.get(k, [])
        list2.append(list2dict(tp2))
        # 处理时间间隔
        ti2 = time_interval2.get(k, [])
        list2[-1].update(list2dict(ti2))
    tsth = time.time()
    print("ICEWS thread_sim_matrix_start:" + str(tsth))
    # print("YAGO-WIKI thread_sim_matrix_start:" + str(tsth))
    m1 = thread_sim_matrix(list1,list2)
    m2 = thread_sim_matrix(list2,list1)
    cost_time = time.time()-tsth
    print("thread cost_time: ", str(cost_time))

    folder_path = "./data/ICEWS05-15/simt"
    # folder_path = "./data/YAGO-WIKI50K/simt"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name1 = os.path.join(folder_path, "simt_ICEWS05-15_ab.npy")
    # file_name1 = os.path.join(folder_path, "simt_YAGO-WIKI50K_ab.npy")
    file_name2 = os.path.join(folder_path, "simt_ICEWS05-15_ba.npy")
    # file_name2 = os.path.join(folder_path, "simt_YAGO-WIKI50K_ba.npy")

    np.save(file_name1, m1)
    np.save(file_name2,m2)

tmp_index1 = []
tmp_index2 = []
THRESHOLD = 0.5
for i in range(m1.shape[0]):
    if ((len(m1[i][m1[i] == np.max(m1[i])]) > 1) | (np.max(m1[i]) < THRESHOLD)):
        continue
    else:
        tmp_index1.append([list(entity1)[i], list(entity2)[np.argmax(m1[i])]])

for j in range(m2.shape[0]):
    if ((len(m2[j][m2[j] == np.max(m2[j])]) > 1) | (np.max(m2[j]) < THRESHOLD)):
        continue
    else:
        tmp_index2.append([list(entity1)[np.argmax(m2[j])], list(entity2)[j]])


sup_pair = []
tmp_index2_set=set([tuple(seed) for seed in tmp_index2])
for i in range(len(tmp_index1)):
    item_tuple=tuple(tmp_index1[i])
    if item_tuple in tmp_index2_set:
        sup_pair.append(tmp_index1[i])

file_name = os.path.join(file_path+dataset, "simt/unsup_seeds_" + dataset[0:-1] + ".npy")
# file_name = file_path+dataset+"simt/unsup_seeds_" + dataset[0:-1] + ".npy"
np.save(file_name,sup_pair)
unsup_seeds_set=set([tuple(seed) for seed in sup_pair])
unsup_dev = []
for i in range(len(all_pair)):
    item_tuple=all_pair[i]
    if item_tuple not in unsup_seeds_set:
        unsup_dev.append(list(all_pair[i]))

file_name = os.path.join(file_path+dataset, "simt/unsup_dev_" + dataset[0:-1] + ".npy")
# file_name = file_path+dataset+"simt/unsup_dev_" + dataset[0:-1] + ".npy"
np.save(file_name, unsup_dev)

cost_time = time.time()-ts
print("program cost_time: ", str(cost_time))
