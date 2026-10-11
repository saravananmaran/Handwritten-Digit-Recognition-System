"""
Neural Network intermediate activations extraction and model inspection utilities.

Extracts real feature maps, pooling outputs, dense activations, and layer metadata
from the trained Keras CNN model.
"""

import time
from typing import Dict, Any, List, Tuple
import numpy as np


def get_layer_metadata(layer) -> Dict[str, Any]:
    """Extracts architectural metadata and specifications for a single layer."""
    layer_type = layer.__class__.__name__
    total_params = layer.count_params()
    trainable_params = sum(np.prod(w.shape) for w in layer.trainable_weights) if layer.trainable_weights else 0
    
    # Activation function
    activation_fn = "None"
    if hasattr(layer, "activation") and layer.activation is not None:
        activation_fn = layer.activation.__name__ if hasattr(layer.activation, "__name__") else str(layer.activation)

    meta = {
        "name": layer.name,
        "type": layer_type,
        "params": int(total_params),
        "trainable_params": int(trainable_params),
        "activation": activation_fn,
        "input_shape": getattr(layer, "input_shape", None),
        "output_shape": getattr(layer, "output_shape", None)
    }

    # Type-specific details
    if layer_type == "Conv2D":
        meta["filters"] = getattr(layer, "filters", None)
        meta["kernel_size"] = getattr(layer, "kernel_size", None)
        meta["padding"] = getattr(layer, "padding", None)
        meta["strides"] = getattr(layer, "strides", None)
    elif layer_type == "MaxPooling2D":
        meta["pool_size"] = getattr(layer, "pool_size", None)
        meta["strides"] = getattr(layer, "strides", None)
    elif layer_type == "Dense":
        meta["units"] = getattr(layer, "units", None)
    elif layer_type == "Dropout":
        meta["rate"] = getattr(layer, "rate", None)

    return meta


def build_activation_extractor(model):
    """
    Constructs a multi-output Keras Model that extracts intermediate activations
    for every layer in the network.
    """
    import tensorflow as tf
    outputs = {layer.name: layer.output for layer in model.layers}
    return tf.keras.Model(inputs=model.inputs, outputs=outputs, name="feature_extractor")


def extract_intermediate_activations(model, input_tensor: np.ndarray, extractor=None) -> Dict[str, Any]:
    """
    Feeds input_tensor (1, 28, 28, 1) through the CNN and extracts all genuine intermediate activations.

    Args:
        model: Trained Keras CNN model.
        input_tensor: 4D numpy array with shape (1, 28, 28, 1).
        extractor: Optional pre-built activation extractor.

    Returns:
        Structured dictionary containing:
            - 'predicted_digit': int
            - 'confidence': float
            - 'probabilities': list of 10 floats
            - 'top_3': list of tuples [(digit, prob), ...]
            - 'inference_time_ms': float
            - 'layer_names': list of str
            - 'layers_meta': dict of layer metadata
            - 'activations': dict of numpy arrays for each layer
    """
    if extractor is None:
        extractor = build_activation_extractor(model)

    # Measure exact inference duration
    start_time = time.perf_counter()
    raw_outputs = extractor([input_tensor], training=False)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    activations = {}
    layers_meta = {}
    layer_names = []

    for layer in model.layers:
        lname = layer.name
        layer_names.append(lname)
        layers_meta[lname] = get_layer_metadata(layer)
        tensor_val = raw_outputs[lname].numpy()
        activations[lname] = tensor_val

    # Final prediction probabilities
    last_layer_name = layer_names[-1]
    last_act = activations[last_layer_name][0]
    probabilities = [float(p) for p in last_act]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    # Top 3 predictions
    ranked_indices = np.argsort(probabilities)[::-1]
    top_3 = [(int(idx), float(probabilities[idx])) for idx in ranked_indices[:3]]

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_3": top_3,
        "inference_time_ms": elapsed_ms,
        "layer_names": layer_names,
        "layers_meta": layers_meta,
        "activations": activations
    }


def get_top_active_filters(feature_maps_4d: np.ndarray, top_k: int = 8) -> List[int]:
    """
    Returns filter indices sorted by total activation energy (mean squared intensity).
    feature_maps_4d: shape (1, H, W, Channels)
    """
    channels = feature_maps_4d.shape[-1]
    scores = []
    for c in range(channels):
        fmap = feature_maps_4d[0, :, :, c]
        scores.append((c, float(np.mean(fmap ** 2))))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scores[:top_k]]
