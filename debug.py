import codecs
list1 = set()
with codecs.open('./data/final_ans_B_0.txt', 'r', 'utf-8') as f:
    for ln in f.readlines():
        ln = ln.strip()
        list1.add(int(ln))
list2 = set()
with codecs.open('./output/final_ans_B.txt', 'r', 'utf-8') as f:
    for ln in f.readlines():
        ln = ln.strip()
        list2.add(int(ln))
ans = list1 & list2
print(len(list1), len(list2))
print(len(ans))
# import scipy.spatial
# ans = scipy.spatial.distance.euclidean((0, 1), (1, 2))
# print(ans)
