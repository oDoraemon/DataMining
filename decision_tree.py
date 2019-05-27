import math

class Node:
    left = None
    right = None

    def __init__(self, name, category):
        self.name = name
        self.category = category

# 测试数据集，见<Data Mining Concepts and Techniques> 2nd edition P299
D = [
    {'rid':1,'age':'youth','income':'high','student':'no','credit_rating':'fair','category':'no'},
    {'rid':2,'age':'youth','income':'high','student':'no','credit_rating':'excellent','category':'no'},
    {'rid':3,'age':'middle_aged','income':'high','student':'no','credit_rating':'fair','category':'yes'},
    {'rid':4,'age':'senior','income':'medium','student':'no','credit_rating':'fair','category':'yes'},
    {'rid':5,'age':'senior','income':'low','student':'yes','credit_rating':'fair','category':'yes'},
    {'rid':6,'age':'senior','income':'low','student':'yes','credit_rating':'excellent','category':'no'},
    {'rid':7,'age':'middle_aged','income':'low','student':'yes','credit_rating':'excellent','category':'yes'},
    {'rid':8,'age':'youth','income':'medium','student':'no','credit_rating':'fair','category':'no'},
    {'rid':9,'age':'youth','income':'low','student':'yes','credit_rating':'fair','category':'yes'},
    {'rid':10,'age':'senior','income':'medium','student':'yes','credit_rating':'fair','category':'yes'},
    {'rid':11,'age':'youth','income':'medium','student':'yes','credit_rating':'excellent','category':'yes'},
    {'rid':12,'age':'middle_aged','income':'medium','student':'no','credit_rating':'excellent','category':'yes'},
    {'rid':13,'age':'middle_aged','income':'high','student':'yes','credit_rating':'fair','category':'yes'},
    {'rid':14,'age':'senior','income':'medium','student':'no','credit_rating':'excellent','category':'no'}
]
attr_list = ['age', 'income', 'student', 'credit_rating']
attr_val_list = {'age': ['youth', 'middle_aged', 'senior'],
                 'income': ['high', 'medium', 'low'],
                 'student': ['yes', 'no'],
                 'credit_rating': ['fair', 'excellent']}
cate_list = ['yes', 'no'] # class 列表

def attr_selection_method(D, attr_list):
    pass

# info - i.e. entropy
# info(D) = -accmulate_sum(pi/log_2(pi)) for i = 1:m
# where pi = |C_i,D|/|D|, C_i,D is the tuple in D with class C_i
def _info(D, cate_list):
    count = {}
    d_len = len(D)
    entropy = 0
    for cate in cate_list:
        count.update({cate:0})
    for item in D:
        count[item['category']] += 1
    for k, v in count.items():
        if v != 0:
            entropy += (-1 * v/d_len * math.log(v/d_len, 2))
    return entropy
    
# info_a
# info_a(D) = accumulate_sum(|Dj|/|D|*info(Dj)) for j = 1:v
# where a refer to the attr to be calculated
def _info_a(D, attr, attr_val):
    d_len = len(D)
    count = {} #{attr1: [subdata], attr2:..}
    info_a = 0
    for v in attr_val:
        count.update({v: []})

    for item in D:
        count[item[attr]].append(item)
    
    for k, v in count.items():
        info_a += len(v)/d_len * _info(v, cate_list)
    
    return info_a
    
# gain(A) = info(D) - info_a(D)
def _gain_info(D, attr):
    return _info(D, cate_list) - _info_a(D, attr, attr_val_list[attr])

# used in C4.5
# to overcome the bias of gain_info: prefer to select attr having large number of values
# split_info_a(D) = -accumulate_sum(|Dj|/|D|*log_2(|Dj|/|D|))
# errata:<Data Mining Concepts and techniques> 2nd edition p301, split_info_a(income) give a wrong answer
def _split_info_a(D, attr, attr_val):
    d_len = len(D)
    count = {} #{attr1: [subdata], attr2:..}
    split_info_a = 0
    for v in attr_val:
        count.update({v: 0})

    for item in D:
        count[item[attr]] += 1
    print(count)
    for k, v in count.items():
        # print(k + ":" + str(len(v)))
        split_info_a += (-1 * v/d_len * math.log(v/d_len, 2))
    
    return split_info_a


# used in C4.5
# to overcome the bias of gain_info: prefer to select attr having large number of values
def _gain_ratio(D, attr):
    return _gain_info(D, attr)/_split_info_a(D, attr, attr_val_list[attr])

# used in CART
# gini_index(D) = 1 - accumulate_sum(pi^2) for i = 1:m
# where i refer to outcome category and pi = |C_i,D|/|D|
# gini index 用于对attr的二元分类
# {low, medium, high}的分类会变成{low, medium}&{high} + {low, high}&{medium} + etc.
# 根据两个类别计算gini
def _gini_index(D):
    count = {}
    d_len = len(D)
    gini_v = 0
    for cate in cate_list:
        count.update({cate:0})
    for item in D:
        count[item['category']] += 1
    for k, v in count.items():
        gini_v +=  (v/d_len)**2
    return 1 - gini_v

def gini_index_a(D, attr):
    pass
    
def generate_decision_tree(D, attr_list):
    pass

# scan之前判断D是否为空
def is_same_category(D):
    category = D[0].category
    for item in D:
        if item['category'] == category:
            continue
        else:
            return False
    return True

def majority_category(D, cate_list):
    count = {}
    max_count = 0
    cate = None

    for cate in cate_list:
        count.update({cate:0})

    for item in D:
        count[item['category']] += 1

    for k, v in count.items():
        if v > max_count:
            cate = k
            max_count = v
    
    return cate

def process():
    pass
    # N = None
    # attr_list = []
    # if not D:
    #     raise ValueError("Empty dataset D")
    # if is_same_category(D):
    #     N = Node('leaf', D[0].category)
    #     return N
    
if __name__ == "__main__":
    print(_gini_index(D))
    # for attr in attr_list:
        # print(attr, _split_info_a(D, attr, attr_val_list[attr]))
        # print(attr, _gain_ratio(D, attr))
    # print(_gain_info(D, 'age'))