import os
import logging as logger
import torch

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    convert_names = {
        'embeddings/LayerNorm': 'emb_layer_norm',
        'embeddings/position_embeddings': 'position_embeddings',
        'embeddings/token_type_embeddings': 'segment_embeddings',
        'embeddings/word_embeddings': 'word_embeddings',
        'attention/output/LayerNorm': 'multi_head_attention/layer_norm',
        'attention/output/dense': 'multi_head_attention/dense',
        'attention/self/key': 'multi_head_attention/w_k',
        'attention/self/query': 'multi_head_attention/w_q',
        'attention/self/value': 'multi_head_attention/w_v',
        'intermediate/dense': 'feed_forward/intermediate',
        'output/LayerNorm': 'layer_norm',
        'output/dense': 'feed_forward/out_dense',
        'cls/predictions/transform/LayerNorm': 'transform/layer_norm',
        'cls/predictions/transform/dense': 'transform/dense',
        'cls/predictions/output_bias': 'predictions/output_bias',
        'cls/seq_relationship': 'seq_relationship'
    }

    for name, shape in init_vars:

        # logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        name = name.replace('encoder', 'encoders').replace('layer', 'layers')
        for scope_name in convert_names:
            if scope_name in name:
                trans_name = convert_names[scope_name]
                name = name.replace(scope_name, trans_name)
                break
        names.append(name)
        # print(name)
        arrays.append(array)
    # quit()
    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            # print(scope_names)
            # print('++++')
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            
            # print(pointer)
            # print(m_name)
            # print('------')
            # print(name)
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model
