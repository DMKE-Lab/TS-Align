import numpy as np
import scipy.sparse as sp
import os
from multiprocessing import Pool
import torch
from log import logger


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T


def sp2torch_sparse(X):
    coo = X.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = X.shape
    X=torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return X


def load_alignment_pair(file_name):
    alignment_pair = []
    for line in open(file_name, 'r'):
        e1, e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))
    return alignment_pair


def get_matrix(triples, entity, rel, time):
    ent_size = max(entity) + 1
    rel_size = max(rel) + 1
    time_size = max(time) + 1
    logger.info("entity size & relation_size & timestamp_size: %d, %d, %d." % (ent_size, rel_size, time_size))
    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    rel_features = sp.lil_matrix((ent_size, rel_size))
    time_point = {}
    time_interval = {}

    for i in range(max(entity) + 1):
        adj_features[i, i] = 1
        time_point[i] = []
        time_interval[i] = []

    for h, r, t, ts, te in triples:
        adj_matrix[h, t] = 1;
        adj_matrix[t, h] = 1;
        adj_features[h, t] = 1;
        adj_features[t, h] = 1;
        rel_features[h, r] = 1;
        rel_features[t, r] = 1
        if ts == te:
            time_point[h].append(ts);
            time_point[t].append(ts)
        elif ts == 0:
            time_point[h].append(te);
            time_point[t].append(te)
        elif te == 0:
            time_point[h].append(ts);
            time_point[t].append(ts)
        else:
            time_interval[h].append([ts, te]);
            time_interval[t].append([ts, te])

    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(rel_features)

    adj_matrix = sp2torch_sparse(adj_matrix)
    adj_features = sp2torch_sparse(adj_features)
    rel_features = sp2torch_sparse(rel_features)

    return adj_matrix, adj_features, rel_features, time_point, time_interval


def list2dict(time_list):
    dic = {}
    for i in time_list:
        if isinstance(i, list):
            key = tuple(i)
        else:
            key = i
        dic[key] = time_list.count(i)
    return dic


def pair_simt(time_point, time_interval, pair):
    t1 = []
    t2 = []

    for e1, e2 in pair:
        tp1 = time_point.get(e1, [])
        tp2 = time_point.get(e2, [])

        t1.append(list2dict(tp1))
        t2.append(list2dict(tp2))

        ti1 = time_interval.get(e1, [])
        ti2 = time_interval.get(e2, [])

        t1[-1].update(list2dict(ti1))
        t2[-1].update(list2dict(ti2))

    m = thread_sim_matrix(t1, t2)
    return m


def load_quadruples(file_name):
    quadruples = []
    entity = set()
    rel = set([0])
    time = set()
    for line in open(file_name, 'r'):
        items = line.split()
        if len(items) == 4:
            head, r, tail, t = [int(item) for item in items]
            entity.add(head);
            entity.add(tail);
            rel.add(r);
            time.add(t)
            quadruples.append((head, r, tail, t, t))
        else:
            head, r, tail, tb, te = [int(item) for item in items]
            entity.add(head);
            entity.add(tail);
            rel.add(r);
            time.add(tb);
            time.add(te)
            quadruples.append((head, r, tail, tb, te))
    return entity, rel, time, quadruples


def load_data(path, ratio=1000):
    print(ratio)
    entity1, rel1, time1, quadruples1 = load_quadruples(path + 'triples_1')
    entity2, rel2, time2, quadruples2 = load_quadruples(path + 'triples_2')

    train_pair = load_alignment_pair(path + 'sup_pairs')
    dev_pair = load_alignment_pair(path + 'ref_pairs')
    dev_pair = train_pair[ratio:] + dev_pair
    train_pair = train_pair[:ratio]
    all_pair = train_pair + dev_pair

    adj_matrix, adj_features, rel_features, time_point, time_interval = get_matrix(
        quadruples1 + quadruples2, entity1.union(entity2), rel1.union(rel2), time1.union(time2))

    return np.array(train_pair), np.array(dev_pair), np.array(all_pair), adj_matrix, adj_features, \
           rel_features, time_point, time_interval

thread_num=18


def div_array(arr,n):
    arr_len = len(arr)
    k = arr_len // n
    ls_return = []
    for i in range(n-1):
        ls_return.append(arr[i*k:i*k+k])
    ls_return.append(arr[(n-1)*k:])
    return ls_return


def indicator(value, interval):
    if value >= interval[0] and value <= interval[1]:
        return True
    else:
        return False


def intersection_length(interval1, interval2):
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    return max(0, end - start + 1)


def sim_matrix(t1, t2):
    size_t1 = len(t1)
    size_t2 = len(t2)
    matrix = np.zeros((size_t1, size_t2))

    for i in range(size_t1):
        time_intervals_i = [key for key, value in t1[i].items() if isinstance(key, list) for _ in range(value)]
        time_points_i = [key for key, value in t1[i].items() if not isinstance(key, list) for _ in range(value)]

        for j in range(size_t2):
            time_intervals_j = [key for key, value in t2[j].items() if isinstance(key, list) for _ in range(value)]
            time_points_j = [key for key, value in t2[j].items() if not isinstance(key, list) for _ in range(value)]

            point_overlap = len(set(time_points_i).intersection(set(time_points_j)))
            interval_overlap = 0
            for interval1 in time_intervals_i:
                for interval2 in time_intervals_j:
                    interval_overlap += intersection_length(interval1, interval2)

            total_length_interval = sum([sub_interval[1] - sub_interval[0] + 1 for sub_interval in time_intervals_i]) + sum(
                [sub_interval[1] - sub_interval[0] + 1 for sub_interval in time_intervals_j])
            total_length_point = len(time_points_i) + len(time_points_j)

            if total_length_point==0 :
                if total_length_interval == 0:
                    sim = 0
                else:
                    sim_intervals = interval_overlap / total_length_interval
                    sim = sim_intervals
            elif total_length_interval == 0:
                sim_point = point_overlap / total_length_point
                sim = sim_point
            else:
                sim_point = point_overlap / total_length_point
                sim_intervals = interval_overlap / total_length_interval
                sim = (sim_point + sim_intervals) / 2

            matrix[i, j] = sim
    return matrix


def thread_sim_matrix(t1,t2):
    pool = Pool(processes=thread_num)
    reses = list()
    tasks_t1 = div_array(t1,thread_num)
    for task_t1 in tasks_t1:
        reses.append(pool.apply_async(sim_matrix,args=(task_t1,t2)))
    pool.close()
    pool.join()
    matrix = None
    for res in reses:
        val = res.get()
        if matrix is None:
            matrix = np.array(val)
        else:
            matrix = np.concatenate((matrix,val),axis=0)
    return matrix


def get_simt(file_name,time_point,time_interval,dev_pair):
    if os.path.exists(file_name):
        pair_mt = np.load(file_name, allow_pickle=True)
    else:
        f1 = os.makedirs('data/ICEWS05-15/simt')
        # f1 = os.makedirs('data/YAGO-WIKI50K/simt')
        # f2 = open(file='data/ICEWS05-15/simt/simt_ICEWS05-15_1000_unsup_dev.npy', mode='x')
        # f2 = open(file='data/ICEWS05-15/simt/simt_ICEWS05-15_200_unsup_dev.npy', mode='x')
        # f2 = open(file='data/YAGO-WIKI50K/simt/simt_YAGO-WIKI50K_5000_unsup_dev.npy', mode='x')
        # f2 = open(file='data/YAGO-WIKI50K/simt/simt_YAGO-WIKI50K_1000_unsup_dev.npy', mode='x')
        f2 = open(file='data/ICEWS05-15/simt/simt_ICEWS05-15_1000.npy', mode='x')
        # f2 = open(file='data/ICEWS05-15/simt/simt_ICEWS05-15_200.npy', mode='x')
        # f2 = open(file='data/YAGO-WIKI50K/simt/simt_YAGO-WIKI50K_5000.npy', mode='x')
        # f2 = open(file='data/YAGO-WIKI50K/simt/simt_YAGO-WIKI50K_1000.npy', mode='x')
        pair_mt = pair_simt(time_point,time_interval,dev_pair)
        np.save(file_name, pair_mt)
    return pair_mt
