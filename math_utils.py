import numpy as np


def dir_to_euler_yzx(direction: np.ndarray) -> np.ndarray:
    """
    Convert a world-space direction vector (pointed-to) into YZX-ordered Euler angles:
    [pitch(Y), yaw(Z), roll(X)] that aim +X forward axis along `direction`.
    """
    d = direction.astype(np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)  # no-op if zero vector

    d /= norm
    dx, dy, dz = d

    # Yaw around Z: angle from +X toward +Y in the XY plane
    yaw = np.arctan2(dy, dx)

    # Pitch around Y: elevation angle; positive pitches nose up toward +Z
    horiz = np.hypot(dx, dy)
    if horiz < 1e-6:
        # pointing straight up/down: yaw is undefined; keep whatever, pitch is +/- pi/2
        pitch = np.pi/2 if dz > 0 else -np.pi/2
    else:
        pitch = np.arctan2(dz, horiz)

    roll = 0.0
    return np.array([pitch, yaw, roll], dtype=np.float32)

def normalize(x: np.array) -> np.array:
    norm = np.linalg.norm(x)
    if norm == 0:
       return x
    return x / norm