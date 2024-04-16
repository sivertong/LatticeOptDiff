from sklearn.preprocessing import StandardScaler
import numpy as np

def standardize_data(data):
    """
    对输入的NumPy数组进行标准化处理。
    参数:
        data: NumPy数组，形状为(n_samples, n_features)
    返回:
        标准化后的NumPy数组。
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data