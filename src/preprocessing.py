"""
Image preprocessing utilities for the Handwritten Digit Recognition System.

Includes functions for:
1. Loading the MNIST dataset from local ubyte files or Keras fallback.
2. Normalization and tensor reshaping.
3. Train/validation splitting.
4. Preprocessing user-drawn canvas images into MNIST-compatible 28x28 grayscale tensors.
"""

import os
import struct
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass


def read_idx_images(file_path: str) -> np.ndarray:
    """Reads IDX format images file (e.g., train-images.idx3-ubyte)."""
    with open(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {file_path}. Expected 2051.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)


def read_idx_labels(file_path: str) -> np.ndarray:
    """Reads IDX format labels file (e.g., train-labels.idx1-ubyte)."""
    with open(file_path, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {file_path}. Expected 2049.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_items)


def find_ubyte_dataset_dir(search_dir: str = None) -> str:
    """Locates the folder containing the MNIST ubyte files."""
    candidates = []
    if search_dir:
        candidates.append(search_dir)
        candidates.append(os.path.join(search_dir, "handwritten data set"))

    cwd = os.getcwd()
    candidates.extend([
        cwd,
        os.path.join(cwd, "handwritten data set"),
        r"D:\projects\Handwritten Digit Recognition System\handwritten data set",
        r"D:\projects\Handwritten Digit Recognition System"
    ])

    for candidate in candidates:
        if os.path.isdir(candidate):
            train_img = os.path.join(candidate, "train-images.idx3-ubyte")
            train_lbl = os.path.join(candidate, "train-labels.idx1-ubyte")
            if os.path.exists(train_img) and os.path.exists(train_lbl):
                return candidate
    return None


def load_mnist_raw(data_dir: str = None):
    """
    Loads raw MNIST data.
    First checks local ubyte dataset files; falls back to tf.keras.datasets.mnist.
    Returns:
        (X_train, y_train), (X_test, y_test) as uint8 arrays.
    """
    resolved_dir = find_ubyte_dataset_dir(data_dir)
    if resolved_dir:
        train_img_path = os.path.join(resolved_dir, "train-images.idx3-ubyte")
        train_lbl_path = os.path.join(resolved_dir, "train-labels.idx1-ubyte")
        test_img_path = os.path.join(resolved_dir, "t10k-images.idx3-ubyte")
        test_lbl_path = os.path.join(resolved_dir, "t10k-labels.idx1-ubyte")

        if all(os.path.exists(p) for p in [train_img_path, train_lbl_path, test_img_path, test_lbl_path]):
            try:
                X_train = read_idx_images(train_img_path)
                y_train = read_idx_labels(train_lbl_path)
                X_test = read_idx_images(test_img_path)
                y_test = read_idx_labels(test_lbl_path)
                return (X_train, y_train), (X_test, y_test)
            except Exception as e:
                print(f"[Warning] Failed to load local ubyte files: {e}. Falling back to tf.keras.")

    # Fallback to tf.keras.datasets.mnist
    import tensorflow as tf
    return tf.keras.datasets.mnist.load_data()


def preprocess_mnist_dataset(X: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel values from 0-255 to 0.0-1.0 and reshapes to (N, 28, 28, 1).
    """
    X_norm = X.astype(np.float32) / 255.0
    if X_norm.ndim == 3:
        X_norm = np.expand_dims(X_norm, axis=-1)
    return X_norm


def load_and_preprocess_data(data_dir: str = None, val_split: float = 0.1666667, random_seed: int = 42):
    """
    Loads MNIST dataset, normalizes pixels, creates validation split, and reshapes.
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_mnist_raw(data_dir)

    num_samples = len(X_train_raw)
    indices = np.arange(num_samples)
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(indices)

    val_count = int(num_samples * val_split)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    X_train = preprocess_mnist_dataset(X_train_raw[train_indices])
    y_train = y_train_raw[train_indices].astype(np.int64)

    X_val = preprocess_mnist_dataset(X_train_raw[val_indices])
    y_val = y_train_raw[val_indices].astype(np.int64)

    X_test = preprocess_mnist_dataset(X_test_raw)
    y_test = y_test_raw.astype(np.int64)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_sample_digits(num_per_digit: int = 5, data_dir: str = None) -> dict:
    """
    Returns a dictionary mapping each digit 0-9 to a list of real MNIST sample images (28x28 uint8).
    Useful for quick presets and exploration.
    """
    (_, _), (X_test, y_test) = load_mnist_raw(data_dir)
    samples = {d: [] for d in range(10)}
    for img, lbl in zip(X_test, y_test):
        lbl = int(lbl)
        if len(samples[lbl]) < num_per_digit:
            samples[lbl].append(img)
        if all(len(v) >= num_per_digit for v in samples.values()):
            break
    return samples


def get_mnist_subset(count: int = 50, digit_filter: int = None, data_dir: str = None):
    """
    Returns a list of tuples (image_28x28_uint8, label_int) for the dataset explorer.
    """
    (_, _), (X_test, y_test) = load_mnist_raw(data_dir)
    results = []
    for img, lbl in zip(X_test, y_test):
        lbl = int(lbl)
        if digit_filter is None or lbl == digit_filter:
            results.append((img, lbl))
            if len(results) >= count:
                break
    return results



def is_canvas_empty(image_array: np.ndarray, threshold: float = 10.0, min_active_pixels: int = 15) -> bool:
    """
    Determines whether a drawn canvas image is blank.
    """
    if image_array is None or image_array.size == 0:
        return True
    return bool(np.sum(image_array > threshold) < min_active_pixels)


def center_digit_by_mass(image_28x28: np.ndarray) -> np.ndarray:
    """
    Centers the digit inside the 28x28 frame according to its center of mass,
    closely replicating the original MNIST dataset generation procedure.
    """
    cy, cx = center_of_mass(image_28x28)
    if np.isnan(cx) or np.isnan(cy):
        return image_28x28

    shift_x = int(np.round(14.0 - cx))
    shift_y = int(np.round(14.0 - cy))

    # Clamp shift to prevent shifting completely out of frame
    shift_x = max(-7, min(7, shift_x))
    shift_y = max(-7, min(7, shift_y))

    centered = np.zeros_like(image_28x28)
    h, w = image_28x28.shape

    src_y_start = max(0, -shift_y)
    src_y_end = min(h, h - shift_y)
    src_x_start = max(0, -shift_x)
    src_x_end = min(w, w - shift_x)

    dst_y_start = max(0, shift_y)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, shift_x)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    centered[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image_28x28[src_y_start:src_y_end, src_x_start:src_x_end]
    return centered


def preprocess_drawn_image(raw_image, target_size=(28, 28)) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses user-drawn canvas image into a MNIST-compatible format.
    Steps:
      1. Converts to grayscale.
      2. Handles polarity (white digit on black background).
      3. Validates non-empty drawing.
      4. Crops tightly to the bounding box of the drawn stroke.
      5. Resizes bounding box preserving aspect ratio into a 20x20 box.
      6. Embeds into a 28x28 black frame and centers via center of mass.
      7. Normalizes to [0.0, 1.0] and reshapes to (1, 28, 28, 1).

    Returns:
        (processed_tensor, debug_28x28_image)
    """
    # Convert input to numpy array
    if hasattr(raw_image, "convert"):  # PIL Image
        raw_arr = np.array(raw_image)
    elif isinstance(raw_image, np.ndarray):
        raw_arr = raw_image.copy()
    else:
        raise ValueError(f"Unsupported image input type: {type(raw_image)}")

    if raw_arr.size == 0:
        raise ValueError("Canvas is empty. Please draw a digit before predicting.")

    # Convert to single grayscale channel
    if raw_arr.ndim == 3:
        if raw_arr.shape[2] == 4:
            # RGBA from canvas
            # Check if alpha channel carries strokes or RGB carries strokes
            alpha = raw_arr[:, :, 3]
            rgb = raw_arr[:, :, :3]
            # Standard canvas: black background (0,0,0) and white stroke (255,255,255)
            # or transparent background where alpha indicates stroke
            if np.max(alpha) > 0 and np.mean(rgb) == 0 and np.max(alpha) > 10:
                # Stroke is in alpha or white stroke
                gray = alpha.astype(np.float32)
            else:
                gray = np.mean(rgb, axis=2).astype(np.float32)
        elif raw_arr.shape[2] == 3:
            gray = np.mean(raw_arr, axis=2).astype(np.float32)
        else:
            gray = raw_arr[:, :, 0].astype(np.float32)
    elif raw_arr.ndim == 2:
        gray = raw_arr.astype(np.float32)
    else:
        raise ValueError(f"Unexpected image dimensions: {raw_arr.shape}")

    # Check polarity: MNIST has black background (0) and white digits (255)
    # If the user drew with black ink on white background, average corners will be ~255.
    corner_mean = (gray[0, 0] + gray[0, -1] + gray[-1, 0] + gray[-1, -1]) / 4.0
    if corner_mean > 127.0:
        # Invert: white background -> black background
        gray = 255.0 - gray

    # Check if canvas is empty
    if is_canvas_empty(gray, threshold=20.0, min_active_pixels=15):
        raise ValueError("Canvas is empty or drawing is too faint. Please draw a clear digit.")

    # Find bounding box of digit
    active_rows = np.where(np.any(gray > 20.0, axis=1))[0]
    active_cols = np.where(np.any(gray > 20.0, axis=0))[0]

    if len(active_rows) == 0 or len(active_cols) == 0:
        raise ValueError("Canvas is empty or drawing is too faint. Please draw a clear digit.")

    rmin, rmax = active_rows[0], active_rows[-1] + 1
    cmin, cmax = active_cols[0], active_cols[-1] + 1

    cropped = gray[rmin:rmax, cmin:cmax]

    # Resize preserving aspect ratio into a box of at most 20x20
    crop_h, crop_w = cropped.shape
    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round((crop_w * 20.0) / crop_h)))
    else:
        new_w = 20
        new_h = max(1, int(round((crop_h * 20.0) / crop_w)))

    cropped_pil = Image.fromarray(np.clip(cropped, 0, 255).astype(np.uint8))
    resized_pil = cropped_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized = np.array(resized_pil, dtype=np.float32)

    # Place in 28x28 black canvas
    canvas_28 = np.zeros(target_size, dtype=np.float32)
    start_y = (target_size[0] - new_h) // 2
    start_x = (target_size[1] - new_w) // 2
    canvas_28[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    # Center by center of mass (MNIST standard)
    centered_28 = center_digit_by_mass(canvas_28)

    # Normalize to [0.0, 1.0]
    normalized_28 = np.clip(centered_28 / 255.0, 0.0, 1.0).astype(np.float32)

    # Reshape to (1, 28, 28, 1)
    tensor = np.expand_dims(normalized_28, axis=(0, -1))

    return tensor, normalized_28
