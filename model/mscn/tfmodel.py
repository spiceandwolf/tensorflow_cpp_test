import tensorflow as tf
from tensorflow import keras


# Define model architecture

class MSCNLayer(tf.keras.Model):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units):
        super().__init__()
        self.sample_mlp1 = tf.keras.layers.Dense(hid_units, input_shape=(sample_feats,), activation="relu", name='sample_mlp1')
        self.sample_mlp2 = tf.keras.layers.Dense(hid_units, input_shape=(hid_units,), activation="relu", name='sample_mlp2')
        self.predicate_mlp1 = tf.keras.layers.Dense(hid_units, input_shape=(predicate_feats,), activation="relu", name='predicate_mlp1')
        self.predicate_mlp2 = tf.keras.layers.Dense(hid_units, input_shape=(hid_units,), activation="relu", name='predicate_mlp2')
        self.join_mlp1 = tf.keras.layers.Dense(hid_units, input_shape=(join_feats,), activation="relu", name='join_mlp1')
        self.join_mlp2 = tf.keras.layers.Dense(hid_units, input_shape=(hid_units,), activation="relu", name='join_mlp2')
        self.out_mlp1 = tf.keras.layers.Dense(hid_units, input_shape=(hid_units * 3,), activation="relu", name='out_mlp1')
        self.out_mlp2 = tf.keras.layers.Dense(1, input_shape=(hid_units,), activation="sigmoid", name='out_mlp2')

    signature_list = [tf.TensorSpec((None, None, 1006), tf.float32, name="samples"),
                      tf.TensorSpec((None, None, 13), tf.float32, name="predicates"),
                      tf.TensorSpec((None, None, 6), tf.float32, name="joins"),
                      tf.TensorSpec((None, None, 1), tf.float32, name="sample_mask"),
                      tf.TensorSpec((None, None, 1), tf.float32, name="predicate_mask"),
                      tf.TensorSpec((None, None, 1), tf.float32, name="join_mask")]
    @tf.function(input_signature = [signature_list])
    def call(self, inputs):
        samples, predicates, joins, sample_mask, predicate_mask, join_mask = inputs
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]
        # hid_sample = tf.keras.layers.ReLU(self.sample_mlp1(samples))
        hid_sample = self.sample_mlp1(samples)
        # model.add(hid_sample)
        # hid_sample = tf.keras.layers.ReLU(self.sample_mlp2(hid_sample))
        hid_sample = self.sample_mlp2(samples)
        # model.add(hid_sample)
        hid_sample = hid_sample * sample_mask  # Mask
        hid_sample = tf.reduce_sum(hid_sample, axis=1, keepdims=False)
        sample_norm = tf.reduce_sum(sample_mask, axis=1, keepdims=False)
        hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

        # hid_predicate = tf.keras.layers.ReLU(self.predicate_mlp1(predicates))
        hid_predicate = self.predicate_mlp1(predicates)
        # model.add(hid_predicate)
        # hid_predicate = tf.keras.layers.ReLU(self.predicate_mlp2(hid_predicate))
        hid_predicate = self.predicate_mlp2(hid_predicate)
        # model.add(hid_predicate)
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = tf.reduce_sum(hid_predicate, axis=1, keepdims=False)
        predicate_norm = tf.reduce_sum(predicate_mask, axis=1, keepdims=False)
        hid_predicate = hid_predicate / predicate_norm

        # hid_join = tf.keras.layers.ReLU(self.join_mlp1(joins))
        hid_join = self.join_mlp1(joins)
        # model.add(hid_join)
        hid_join = self.join_mlp2(hid_join)
        # model.add(hid_join)
        hid_join = hid_join * join_mask
        hid_join = tf.reduce_sum(hid_join, axis=1, keepdims=False)
        join_norm =  tf.reduce_sum(join_mask, axis=1, keepdims=False)
        hid_join = hid_join / join_norm

        hid = tf.keras.layers.Concatenate(axis=1)([hid_sample, hid_predicate, hid_join])
        # model.add(hid)
        hid = self.out_mlp1(hid)
        # model.add(hid)
        out = self.out_mlp2(hid)
        # model.add(out)
        return out
    
