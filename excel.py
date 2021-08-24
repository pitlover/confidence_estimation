import csv
def extract_data():
    f1 = open('./excel/cifar100/sum.csv', 'r')
    f2 = open('./excel/cifar100/cifar100_idx_confidence_300.csv')
    rdr = csv.reader(f1)
    rdr2 = csv.reader(f2)

    sum_line = []
    matrix_idx_confidence = []
    for line in rdr:
        sum_line.append(line)
        print(line)

    for a in rdr2:
        matrix_idx_confidence.append(a)

    print(sum_line[1][0])
    print(len(matrix_idx_confidence))
    f1.close()
    f2.close()

def extract_index():
    f1 = open('./excel/cifar100/cifar100_idx_iscorrect_300.csv', 'r')
    lines = f1.readlines()
    f1.close()

    index = []
    for a in range(1, len(lines)):
        index.append(lines[a].split(",")[0])
    return index
# f2 = open('./excel/cifar100/cifar100_idx_confidence_300.csv')
# rdr = csv.reader(f1)
# # rdr2 = csv.reader(f2)
# sum_line = []
# matrix_idx_confidence = []
# for line in rdr:
#     sum_line.append(line)
#     print(line)
#
# # for a in rdr2:
# #     matrix_idx_confidence.append(a)
#
# print(sum_line[1][0])
# print(len(matrix_idx_confidence))

# f2.close()

