import math
import copy

class Node(object):
    root = None
    children = []
    category = None

class D(object):

    def __init__(self, rid, age, income, is_student, rating, buys):
        self.rid = rid
        self.age = age
        self.income = income
        self.is_student = is_student
        self.rating = rating 
        self.buys = buys
    
    def __repr__(self):
        return 'rid {}, buys {}'.format(self.rid, self.buys)

attr_list = []
data = tuple()
CATEGORY = ['Y', 'N'] # class label
ANOMALY_CATE = []
ANOMALY_ATTR = []
ATTR_LIST = {'age':['youth', 'middle', 'senior'], 'income':['high', 'medium', 'low']} # attr list

def is_same_category(data):
    category = None
    if data:
        category == data[0].category

    for d in data:
        if d.category != category:
            return False
    return True

# calculate info of D
def _info(data):
    data_len = len(data)
    count = {}
    for c in CATEGORY:
        count.update({c:0})
    for d in data:
        if d.category in CATEGORY:
            count[str(d.category)] += 1
        else: # 记录异常分类的数据
            ANOMALY_CATE.append(d)
    info = 0
    for k,v in count.items():
        p = v/data_len
        info = info +  -p * math.log(p, 2)
    return info

def _info_a(data, attr):
    data_len = len(data)
    info_a = 0
    subdata = {}
    for v in ATTR_LIST[attr]:
        subdata.update({str(v):[]})
    for d in data:
        if d[attr] in subdata:
            subdata[d[attr]].append(d)
        else:
            ANOMALY_ATTR.append(d)
    for k, v in subdata.items():
        sub_len = len(v)
        info_a = info_a + sub_len/data_len * _info(v)
    return info_a

# $todo: 目前还是单节点select, 递归的时候需要考虑sub_attr的pop问题
def attr_select(data, method):
    if method == 'gain_info':
        sub_attr = copy.deepcopy(ATTR_LIST)
        count = gain_info(data, sub_attr)
        max_gain = 0
        max_gain_attr = None
        for c in count:
            if max_gain < count[c]:
                max_gain = count[c]
                max_gain_attr = c
        return max_gain_attr
    else:
        return None
    
# attribute selection methods - used in ID3
# 需要sub_attr用于递归调用，否则会子树重复遍历某个属性
# info gain 定义: gain(D, attr) = info(D) - info_a(D,attr)
# 由于info(D)都一样，所以无需计算。只要info_a(D, attr)最小，则gain最大
def gain_info(data, sub_attr):
    # sub_attr = copy.deepcopy(ATTR_LIST)
    count = {} # {attr: info_a}
    for k in sub_attr:
        count.update({k:0})
    for c in count:
        count[c] = _info_a(data, c) * -1 # 乘以-1使顺序倒置，保证选max_gain的逻辑正确
    return count
    
# attribute selection methods - used in C4.5
def gain_ratio(data, attr_list):
    pass

# attribute selection methods
def gini_index(data, attr_list):
    pass

def generate_decision_tree(D, attr_list):
    pass

