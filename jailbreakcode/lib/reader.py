import os
import csv
import json
from datasets import Dataset

sst2_index = [
    1730, 979, 292, 434, 1550, 1454, 628, 1285, 1348, 1511, 771, 1263,
    1772, 1474, 737, 547, 120, 553, 336, 87, 1729, 1282, 1048, 810, 
    1115, 351, 846, 1752, 453, 1097, 1735, 275, 1256, 325, 1365, 
    190, 577, 573, 331, 1809, 1556, 102, 679, 861, 48, 108, 47, 
    510, 177, 859, 1216, 546, 674, 1219, 970, 1109, 267, 400, 962, 
    1291, 1203, 181, 1705, 137, 1005, 1461, 1672, 1429, 1045, 424, 
    1695, 194, 1499, 1652, 1506, 348, 285, 1403, 241, 282, 1135, 
    1172, 969, 957, 1463, 793, 1368, 556, 636, 1694, 1531, 1152, 
    572, 1738, 571, 75, 442, 1017, 802, 764
]

agnews_index = [
    3177, 2130, 909, 1010, 1021, 108, 3239, 1050, 772, 2736, 1661, 
    386, 3329, 1650, 1098, 541, 94, 3662, 169, 368, 3683, 3786, 748, 
    678, 487, 808, 605, 1964, 2852, 3017, 1093, 1213, 2589, 19, 3624, 
    1271, 1347, 3536, 2961, 2206, 102, 2748, 161, 3618, 2395, 946, 3690, 
    2097, 331, 3789, 2920, 2618, 1659, 2356, 1884, 17, 98, 3487, 1379, 1540, 
    3138, 725, 2126, 2654, 194, 3020, 1573, 2749, 2369, 2474, 2107, 2152, 661,
    1497, 2318, 2650, 3140, 201, 739, 664, 79, 3521, 2683, 1994, 1781, 1760,
    1794, 2388, 1216, 3223, 1123, 1386, 2582, 2956, 3432, 1467, 3272, 1078,
    2584, 1735, 3177, 2130, 909, 1010, 1021, 108, 3239, 1050, 772, 2736,
    1661, 386, 3329, 1650, 1098, 541, 94, 3662, 169, 368, 3683, 3786, 748,
    678, 487, 808, 605, 1964, 2852, 3017, 1093, 1213, 2589, 19, 3624, 1271,
    1347, 3536, 2961, 2206, 102, 2748, 161, 3618, 2395, 946, 3690, 2097, 331,
    3789, 2920, 2618, 1659, 2356, 1884, 17, 98, 3487, 1379, 1540, 3138, 725,
    2126, 2654, 194, 3020, 1573, 2749, 2369, 2474, 2107, 2152, 661, 1497, 2318,
    2650, 3140, 201, 739, 664, 79, 3521, 2683, 1994, 1781, 1760, 1794, 2388,
    1216, 3223, 1123, 1386, 2582, 2956, 3432, 1467, 3272, 1078, 2584, 1735
]

def read_csv(path, indexes):
    logs = open(path).readlines()
    logs = [log.strip().split('\t') for log in logs] 
    data = {
        'sentences' : [],
        'labels' : [],
    }
    for log in logs:
        data['sentences'].append(log[0].strip())
        data['labels'].append(log[1].strip())
        
    return Dataset.from_dict(data)
