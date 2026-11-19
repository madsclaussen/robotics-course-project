"""Camera helpers for Problem 11 (OpenCV).

Geometry: the camera sits at frame 5. `R_5C` gives the orientation of the
camera's optical frame (OpenCV convention: x right, y down, z forward) in
frame 5. The default assumes the camera looks along x5 (the stylus
direction). Calibrate it (and the intrinsics K) on the real robot.
"""
import numpy as np
import cv2

from .kinematics import CAM_OFFSET_DEFAULT, D_DEFAULT, T05

# Columns = optical axes expressed in frame 5:
#   x_c (image right) = -y5, y_c (image down) = -z5, z_c (forward) = x5.
# This is a guess -- verify on the robot (move an object right in the image
# and check which way the back-projected point moves).
R_5C_DEFAULT = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float)


def open_camera(index=0, width=None, height=None):
    cap = cv2.VideoCapture(index)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise IOError(f"Camera {index} could not be opened")
    return cap


def grab(cap, flush=3):
    """Read a fresh frame (drop a few buffered ones after the robot moved)."""
    for _ in range(flush):
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        raise IOError("Camera read failed")
    return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

def find_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 3)

    _, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY )

    return gray

def pixels_to_plane(uv, K, q, plane_z=0.0, **pose_kw):
    """Back-project N pixels (N x 2 array of [u, v]) onto the plane z = plane_z [mm].

    Returns (points, valid):
        points : N x 3 array of points in the base frame (NaN where invalid)
        valid  : boolean mask, False where the ray is parallel to / points away from the plane
    """
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    T = camera_pose(q, **pose_kw)
    R, c0 = T[:3, :3], T[:3, 3]

    uv1 = np.column_stack((uv, np.ones(len(uv))))      # N x 3 homogeneous pixels
    rays_c = np.linalg.solve(K, uv1.T)                  # 3 x N rays in camera frame
    rays_0 = R @ rays_c                                 # 3 x N rays in base frame

    dz = rays_0[2]
    valid = np.abs(dz) > 1e-9
    s = np.full(len(uv), np.nan)
    s[valid] = (plane_z - c0[2]) / dz[valid]
    valid &= s > 0

    points = c0[:, None] + s * rays_0                   # 3 x N
    points = points.T
    points[~valid] = np.nan
    return points, valid


def image_to_plane(binary, K, q, plane_z=0.0, **pose_kw):
    """White pixels in a binary image -> 3D points on the plane, in the base frame."""
    uv = uv_coordinates(binary)
    points, valid = pixels_to_plane(uv, K, q, plane_z, **pose_kw)
    return points[valid]

def uv_coordinates(edges):
    """"Takes the input image and returns coordinates of non black elements"""
    v, u = np.nonzero(edges)
    return np.column_stack((u, v))


def camera_pose(q, d=D_DEFAULT, cam_offset=CAM_OFFSET_DEFAULT, R_5c=R_5C_DEFAULT):
    """4x4 pose of the optical camera frame in the base frame."""
    T = T05(q, d, cam_offset).copy()
    T[:3, :3] = T[:3, :3] @ R_5c
    return T


def pixel_to_plane(u, v, K, q, plane_z=0.0, **pose_kw):
    """Back-project pixel (u, v) onto the horizontal plane z = plane_z [mm].

    Returns the 3D point in the base frame, or None if the ray is parallel
    to / pointing away from the plane.
    """
    T = camera_pose(q, **pose_kw)
    ray_c = np.linalg.solve(K, [u, v, 1.0])
    ray_0 = T[:3, :3] @ ray_c
    c0 = T[:3, 3]
    if abs(ray_0[2]) < 1e-9:
        return None
    s = (plane_z - c0[2]) / ray_0[2]
    return None if s <= 0 else c0 + s * ray_0


def triangulate(rays):
    """Least-squares intersection of rays [(origin, direction), ...] from several
    poses (Hint 2: depth from multiple end-effector poses)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, r in rays:
        r = r / np.linalg.norm(r)
        P = np.eye(3) - np.outer(r, r)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def pixel_ray(u, v, K, q, **pose_kw):
    """(origin, direction) of the viewing ray of pixel (u, v) in the base frame."""
    T = camera_pose(q, **pose_kw)
    return T[:3, 3], T[:3, :3] @ np.linalg.solve(K, [u, v, 1.0])
