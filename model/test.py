from mscn.data import *
from pytorch2keras.converter import pytorch_to_keras
from torchsummary import summary
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable

def converted_fully_mscn(samples, predicates, joins, sample_masks, predicate_masks, join_masks) :
    # define input tensor
    samples_var = Variable(torch.FloatTensor(samples))
    # print(samples_var.shape)
    predicates_var = Variable(torch.FloatTensor(predicates))
    joins_var = Variable(torch.FloatTensor(joins))
    sample_masks_var = Variable(torch.FloatTensor(sample_masks))
    predicate_masks_var = Variable(torch.FloatTensor(predicate_masks))
    join_masks_var = Variable(torch.FloatTensor(join_masks))
    # get pytorch model
    model_to_transfer = torch.load("/root/tensorflow_c++_test/model/mscn/mscn.pt")
    model_to_transfer.eval()
    # outputs = model_to_transfer(samples[-3:], predicates[-3:], joins[-3:], sample_masks[-3:], predicate_masks[-3:], join_masks[-3:])
    # print(outputs)
    summary(model_to_transfer, [samples.shape, predicates.shape, joins.shape, sample_masks.shape, predicate_masks.shape, join_masks.shape])
    # convert pytorch model to Keras
    # model = pytorch_to_keras(
    #     model_to_transfer,
    #     [samples_var, predicates_var, joins_var, sample_masks_var, predicate_masks_var, join_masks_var],
    #     [samples_var.shape, predicates_var.shape, joins_var.shape, sample_masks_var.shape, predicate_masks_var.shape, join_masks_var.shape],
    #     change_ordering=False, verbose=False, 
    # )

    # return model

trainfilename = "model/mscn/train"
testfilename = "model/mscn/job-light"
num_materialized_samples = 1000
joins_train, predicates_train, tables_train, samples_train, label_train = load_data(trainfilename, num_materialized_samples)
# Get column name dict
column_names = get_all_column_names(predicates_train)
column2vec, idx2column = get_set_encoding(column_names)

# Get table name dict
table_names = get_all_table_names(tables_train)
table2vec, idx2table = get_set_encoding(table_names)

# Get operator name dict
operators = get_all_operators(predicates_train)
op2vec, idx2op = get_set_encoding(operators)

# Get join name dict
join_set = get_all_joins(joins_train)
join2vec, idx2join = get_set_encoding(join_set)
# test data
joins, predicates, tables, samples, label = load_data(testfilename, num_materialized_samples)

# print("tables: " + "%d"%len(tables[0]) + "\n")
# print("joins: " + "%d"%len(joins[0]) + "\n")
# print("label: " + "%d"%len(label) + "\n")
# print("predicates: " + "%d"%len(predicates[0]) + "\n")
# print("samples: " + "%d"%len(samples[0][0]))
# print(samples[0][0])
# with open(filename + ".bitmaps", 'rb') as f:
#     four_bytes = f.read(4)
#     if not four_bytes:
#         print("Error while reading 'four_bytes'")
#         exit(1)
#     num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')

# print("column2vec: " + "%d"%len(column2vec) + "\n")
# print(list(column2vec.values())[0])
# print("\n")
# print("table2vec: " + "%d"%len(table2vec) + "\n")
# print(list(table2vec.values())[0])
# print("\n")
# print("op2vec: " + "%d"%len(op2vec) + "\n")
# print(list(op2vec.values())[0])
# print("\n")
# print("join2vec: " + "%d"%len(join2vec) + "\n")
# print(list(join2vec.values())[0])
# print("\n")

# Get min and max values for each column
with open("model/mscn/column_min_max_vals.csv", 'r', newline=None) as f:
    data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
    column_min_max_vals = {}
    for i, row in enumerate(data_raw):
        if i == 0:
            continue
        column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

# print(list(column_min_max_vals.items())[0])

# Get feature encoding and proper normalization
samples_test = encode_samples(tables, samples, table2vec)
predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
labels_test, _, _ = normalize_labels(label, 0, 1)

# np.set_printoptions(threshold=10000)
# print(len(samples_test))
# print(len(samples_test[0]))
# print(len(samples_test[0][0]))
# print(samples_test[0][0])
# print(len(predicates_test))
# print(len(predicates_test[0]))
# print(len(predicates_test[0][0]))
# print(predicates_test[0][0])
# print(len(joins_test))
# print(len(joins_test[0]))
# print(len(joins_test[0][0]))
# print(joins_test[0][0])

max_num_predicates = max([len(p) for p in predicates_test])
max_num_joins = max([len(j) for j in joins_test])

# print('%d'%max_num_predicates + " " + '%d'%max_num_joins)

test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)
test_data_loader = DataLoader(test_data, batch_size=100)
for batch_idx, data_batch in enumerate(test_data_loader) :
    samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch
    model = converted_fully_mscn(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
    # print(samples.shape)
#     break

# test_data_loader = DataLoader(test_data, 100)
    

outputs = model