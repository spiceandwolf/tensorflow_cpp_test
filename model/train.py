import argparse
import time
import os
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from mscn.util import *
from mscn.tfdata import get_train_datasets, load_data, make_dataset
from mscn.tfmodel import MSCNLayer
from tensorflow import losses

class Qerror_loss(tf.losses.Loss):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        
    def call(self, y_true, y_pred):
        qerror = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        preds = unnormalize_torch(y_pred, self.min_val, self.max_val)
        targets = unnormalize_torch(y_true, self.min_val, self.max_val)

        for i in range(len(targets)):
            if (preds[i] > targets[i]):
                qerror = qerror.write(qerror.size(), preds[i] / targets[i])
            else:
                qerror = qerror.write(qerror.size(), targets[i] / preds[i])
        return tf.reduce_mean(qerror.stack())

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return tf.exp(vals)

# 损失函数需要其他值，不能适用fit
def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return tf.reduce_mean(tf.convert_to_tensor(qerror, dtype=tf.float32))


def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    for data_batch in data_loader.take(1):

        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch
        inputs = [samples, predicates, joins, sample_masks, predicate_masks, join_masks]

        if cuda:
            samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        samples, predicates, joins, targets = tf.Variable(samples), tf.Variable(predicates), tf.Variable(joins), tf.Variable(
            targets)
        sample_masks, predicate_masks, join_masks = tf.Variable(sample_masks), tf.Variable(predicate_masks), tf.Variable(
            join_masks)
        
        t = time.time()
        outputs = model(inputs)
        t_total += time.time() - t

        for i in range(outputs.shape[0]):
            preds.append(outputs[i])

    return preds, t_total


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


def train_and_predict(workload_name, num_queries, num_epochs, batch_size, hid_units, cuda):
    # Load training and validation data
    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts

    # Train model
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = MSCNLayer(sample_feats, predicate_feats, join_feats, hid_units)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    if cuda:
        model.cuda()

    samples_tr, predicates_tr, joins_tr, targets_tr, sample_masks_tr, predicate_masks_tr, join_masks_tr = train_data
    samples_ts, predicates_ts, joins_ts, targets_ts, sample_masks_ts, predicate_masks_ts, join_masks_ts = test_data
    
    train_data_loader = tf.data.Dataset.from_tensor_slices((samples_tr, predicates_tr, joins_tr, targets_tr, sample_masks_tr, predicate_masks_tr, join_masks_tr))
    test_data_loader = tf.data.Dataset.from_tensor_slices((samples_ts, predicates_ts, joins_ts, targets_ts, sample_masks_ts, predicate_masks_ts, join_masks_ts))
    train_data_loader = train_data_loader.batch(batch_size)
    test_data_loader = test_data_loader.batch(batch_size)
    # model.train() 尝试重写下吧
    # for epoch in range(num_epochs) :
    #     loss_total = 0.
    #     for data_batch in train_data_loader.take(1):
    #         with tf.GradientTape() as tape:
    #             samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

    #             if cuda:
    #                 samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
    #                 sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
    #             samples, predicates, joins, targets = tf.Variable(samples), tf.Variable(predicates), tf.Variable(joins), tf.Variable(
    #                 targets)
    #             sample_masks, predicate_masks, join_masks = tf.Variable(sample_masks), tf.Variable(predicate_masks), tf.Variable(
    #                 join_masks)

    #             outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
    #             loss = qerror_loss(outputs, targets, min_val, max_val)
    #             loss_total += loss
            
    #         grads = tape.gradient(loss, model.trainable_variables)
    #         optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    #         print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))
    
    loss = Qerror_loss(min_val, max_val)
    model.compile(optimizer=optimizer, loss=loss)
    inputs = [samples_tr, predicates_tr, joins_tr, sample_masks_tr, predicate_masks_tr, join_masks_tr]
    model.fit(inputs, targets_tr, batch_size=batch_size, epochs=num_epochs)
    model.summary()

    # Get final training and validation set predictions
    preds_train, t_total = predict(model, train_data_loader, cuda)
    print("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per validation sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

    # save model 
    tf.saved_model.save(model, "/root/tensorflow_c++_test/model/mscn/tfmodel")
    print("\nmodel saved")

    # Print metrics
    print("\nQ-Error training set:")
    print_qerror(preds_train_unnorm, labels_train_unnorm)

    print("\nQ-Error validation set:")
    print_qerror(preds_test_unnorm, labels_test_unnorm)
    print("")

    # Load test data
    file_name = "mscn/" + workload_name
    joins, predicates, tables, samples, label = load_data(file_name, num_materialized_samples)

    # Get feature encoding and proper normalization
    samples_test = encode_samples(tables, samples, table2vec)
    predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    print("Number of test samples: {}".format(len(labels_test)))

    max_num_predicates = max([len(p) for p in predicates_test])
    max_num_joins = max([len(j) for j in joins_test])

    # Get test set predictions
    test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)
    samples_JOB, predicates_JOB, joins_JOB, targets_JOB, sample_masks_JOB, predicate_masks_JOB, join_masks_JOB = test_data
    test_data_loader = tf.data.Dataset.from_tensor_slices((samples_JOB, predicates_JOB, joins_JOB, targets_JOB, sample_masks_JOB, predicate_masks_JOB, join_masks_JOB))
    test_data_loader = test_data_loader.batch(batch_size)

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error " + workload_name + ":")
    print_qerror(preds_test_unnorm, label)

    # Write predictions
    file_name = "results/predictions_" + workload_name + ".csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        for i in range(len(preds_test_unnorm)):
            f.write(str(preds_test_unnorm[i]) + "," + label[i] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testset", help="synthetic, scale, or job-light")
    parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
    parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=10)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    args = parser.parse_args()
    train_and_predict(args.testset, args.queries, args.epochs, args.batch, args.hid, args.cuda)


if __name__ == "__main__":
    main()
