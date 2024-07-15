#!/usr/bin/env python3
import numpy as np

import numpy as np

def quaternion_to_rotation_matrix(x, y, z, w):
    """
    Convert a quaternion into a rotation matrix.
    
    Parameters:
    x, y, z, w (float): Quaternion components.
    
    Returns:
    numpy.ndarray: 3x3 rotation matrix.
    """
    # Normalize the quaternion
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    
    # Compute the rotation matrix components
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    yy = y * y
    yz = y * z
    yw = y * w
    zz = z * z
    zw = z * w
    
    rotation_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
        [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
        [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
    ])
    
    return rotation_matrix

def check_points_within_range(points, range_threshold):
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            distance = np.linalg.norm(points[i] - points[j])
            if distance > range_threshold:
                return False
    return True

def get_up_vector(rotation_matrix):
    """
    Extract the up vector from the rotation matrix.
    
    Parameters:
    rotation_matrix (numpy.ndarray): 3x3 rotation matrix.
    
    Returns:
    numpy.ndarray: Up vector (z-axis of the rotation matrix).
    """
    return rotation_matrix[:, 2]  # The third column is the up vector

def calculate_inclination_angle(up_vector):
    """
    Calculate the inclination angle from the global up vector (0, 0, 1).
    
    Parameters:
    up_vector (numpy.ndarray): The up vector of the object.
    
    Returns:
    float: Angle in degrees.
    """
    global_up_vector = np.array([0, 0, 1])
    
    # Dot product and magnitudes
    dot_product = np.dot(up_vector, global_up_vector)
    magnitude_product = np.linalg.norm(up_vector) * np.linalg.norm(global_up_vector)
    
    # Angle in radians and then converted to degrees
    angle_rad = np.arccos(dot_product / magnitude_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


