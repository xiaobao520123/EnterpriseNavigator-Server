# coding=utf-8
import numpy as np

import cfg

def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    return not region.isdisjoint(neighbor)

def region_neighbor(region_set):
    j_min = 100000000
    j_max = -1
    i_m = 0
    for node in region_set:
       i_m = node[0] + 1
       if node[1] > j_max:
           j_max = node[1]
       if node[1] < j_min:
           j_min = node[1]
    j_min = j_min - 1
    j_max = j_max + 2
    neighbor = set()
    for j in range(j_min, j_max):
        neighbor.add((i_m, j))
    return neighbor


def region_group(region_list):
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))
    return D

def rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:
        if n > m and abs(n - m) > 20: # 判断 n > m的目的是：防止n从m后边追上来时，被break，比如：n=44；m=56
            break        
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(rec_region_merge(region_list, e, S))
    return rows

def nms(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
    region_list = []
    region_list_idx = []
    last_i = -1
    current_i = 0
    zipv = zip(activation_pixels[0], activation_pixels[1])
    for i, j in zipv:
        if i != last_i:
            region_list.append({(i, j)})
            region_list_idx.append(i)
            last_i = i
            continue
        merge = False
        for k in range(len(region_list)):
            current_i = region_list_idx[k]
            if i != current_i:
                continue
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:
            region_list.append({(i, j)})
            region_list_idx.append(i)
    D = region_group(region_list)
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        for row in group:
            for ij in region_list[row]:
                score = predict[ij[0], ij[1], 1]
                if score >= threshold:
                    ith_score = predict[ij[0], ij[1], 2:3]
                    if not (cfg.trunc_threshold <= ith_score < 1 -
                            cfg.trunc_threshold):
                        ith = int(np.around(ith_score))
                        total_score[ith * 2:(ith + 1) * 2] += score
                        px = (ij[1] + 0.5) * cfg.pixel_size
                        py = (ij[0] + 0.5) * cfg.pixel_size
                        p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7],
                                              (2, 2))
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + cfg.epsilon)
    return score_list, quad_list
