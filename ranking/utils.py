import tensorflow as tf
import numpy as np
import os
def load_header(model_path):
    header_file = os.path.join(model_path, 'selected_header.txt')
    with open(header_file, 'r') as f:
        header = f.readlines()
    return [h.rstrip() for h in header]

def get_feature_weights(model_path, output_path):
    # load feature list
    header = load_header(model_path)
    # load checkpoint
    ckpt = tf.train.latest_checkpoint(model_path)
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    saver = tf.train.import_meta_graph(ckpt + '.meta')
    saver.restore(sess, ckpt)
    """
    """from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    all_vars = tf.get_collection(tf.ops.GraphKeys.GLOBAL_VARIABLES)
    """
    reader = tf.train.NewCheckpointReader(ckpt)
    # check all variables
    variables = reader.get_variable_to_shape_map()
    print (variables)
    # get demanded weights
    w = reader.get_tensor('net/layer_2/kernel')
    w = np.squeeze(w)
    d = {header[i]:w[i] for i in range(len(w))}
    #s = sorted(d.items(), key=operator.itemgetter(1))
    s = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    # write to file
    with open (output_path, 'w') as f:
        f.writelines(['{}\t{}\n'.format(k, v) for k,v in s])
    return s

def save_top_features(in_files, out_file, n=300):
    top_feat = set()
    for fn in in_files:
        with open(fn, 'r') as f:
            for i in range(n):
                line = f.readline()
                top_feat.add(line.split()[0])
    with open(out_file, 'w') as f:
        for feat in sorted(top_feat):
            f.write(feat+'\n')

def load_dev(header):
    temp_file = 'feat_stat.npy'
    moments = np.load(temp_file)
    moments = moments.item()
    m = [moments[field][0] for field in header]
    m2 = [moments[field][1] for field in header]
    var = m2# - m**2
    return np.sqrt(var)
def weights2freeform(weights_file, model_path, ff_file):
    # load weights
    weights = {}
    with open(weights_file, 'r') as f:
        for line in f:
            tokens = line.rstrip().split()
            weights[tokens[0]] = float(tokens[1])
    header = load_header(model_path)
    dev = load_dev(header)

    ff = ''
    for i, h in enumerate(header):
        coef = weights[h] / dev[i]
        mul = '(* {} {})'.format(coef, h)
        if len(ff) == 0:
            ff = mul
        else:
            ff = '(+ {} {})'.format(mul, ff)
    with open(ff_file, 'w') as f:
        f.write(ff+'\n')


if __name__ == '__main__':
    #model_path = r'models\bingsat_reg_adam_batch_topfeat'
    #weights_file = 'weights_top_4.txt'
    #ff_file = 'freeform_326_longer.txt'
    
    model_path = r'models\bingsat_reg_adam_batch_log_topfeat'
    weights_file = 'weights_top_log_2.txt'
    ff_file = 'freeform_326_log_2.txt'
    
    get_feature_weights(model_path, weights_file)
    weights2freeform(weights_file, model_path, ff_file)