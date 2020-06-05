import os
import sys
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


if len(sys.argv) != 3:
    print('convert.py model.pb model_eport_dir')
    exit(1)

graph_pb = sys.argv[1]
export_dir = sys.argv[2]

assert os.path.exists(graph_pb), 'model.pd must exists'
assert not os.path.exists(export_dir), 'model_eport_dir must NOT exists'

# export_dir = './saved'
# graph_pb = 'edsr_x4.pb'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()

    # import pdb; pdb.set_trace()
    # inp = g.get_tensor_by_name("real_A_and_B_images:0")
    # out = g.get_tensor_by_name("generator/Tanh:0")

    tensor_names = [n.name for n in g.as_graph_def().node]
    if 'sr_input' in tensor_names:
        inp = g.get_tensor_by_name('sr_input:0')
    elif 'Placeholder' in tensor_names:
        inp = g.get_tensor_by_name('Placeholder:0')
    else:
        inp = g.get_tensor_by_name(tensor_names[0] + ':0')
    out = g.get_tensor_by_name('sr_output:0')

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = tf.saved_model.signature_def_utils.predict_signature_def(
        {"in": inp},
        {"out": out}
    )

    # sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = tf.saved_model.signature_def_utils.predict_signature_def(
    #     inp,
    #     out
    # )

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()
