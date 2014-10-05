import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product,combinations


def quatmult(p, q):
    """
    Perform multiplication of quaternion p * q
    :param p: [p0,p1,p2,p3]
    :param q: [q0,q1,q2,q3]
    :return: quaternion after multiplication -> [r0,r1,r2,r3]
    """
    result = np.zeros(4)
    result[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]
    result[1] = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2]
    result[2] = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1]
    result[3] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
    return result


def quat(w, deg):
    """
    Get a quaternion for rotation along w-axis by angle deg
    :param w: axis of rotation -> unit vector [x,y,z]
    :param deg: angle of rotation in degree
    :return: quaternion
    """
    rad = np.radians(deg)
    return np.array([np.cos(rad / 2.0), np.sin(rad/2.0)*w[0], np.sin(rad/2.0)*w[1], np.sin(rad/2.0)*w[2]])


def quat_conj(q):
    """
    Get the conjugate of quaternion q
    :param q:
    :return:
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def rotate(p, axis, deg):
    """
    Rotate point p wrt to the axis for angle deg
    :param p: point to be rotated
    :param axis: axis of rotation -> {'x','y','z'}
    :param deg: angle of rotation in degree
    :return: point after rotation
    """
    if axis != 'y':
        return ValueError('Only rotation wrt to y-axis is implemented')

    p = np.insert(p, 0, 0)  # Convert p to quaternion form
    w = np.array([0,1,0])  # y-axis of rotation
    q = quat(w, deg)
    return quatmult(q, quatmult(p, quat_conj(q)))[1:]


def quat2rot(q):
    """
    Convert quaternion to rotation matrix
    :param q: quaternion [q0,q1,q2,q3]
    :return: 3x3 rotation matrix
    """
    q2 = np.power(q, 2)  # q squared
    rot = np.zeros((3, 3))
    rot[0][0] = q2[0] + q2[1] - q2[2] - q2[3]
    rot[0][1] = 2 * (q[1] * q[2] - q[0] * q[3])
    rot[0][2] = 2 * (q[1] * q[3] + q[0] * q[2])
    rot[1][0] = 2 * (q[1] * q[2] + q[0] * q[3])
    rot[1][1] = q2[0] + q2[2] - q2[1] - q2[3]
    rot[1][2] = 2 * (q[2] * q[3] - q[0] * q[1])
    rot[2][0] = 2 * (q[1] * q[3] - q[0] * q[2])
    rot[2][1] = 2 * (q[2] * q[3] + q[0] * q[1])
    rot[2][2] = q2[0] + q2[3] - q2[1] - q2[2]
    return rot


def get_translation(p, deg):
    """
    Get the camera translation after rotating by angle deg
    :param p: point before rotation
    :param deg: angle of rotation in degree
    :return: point after rotation
    """
    return rotate(p, axis='y', deg=deg)  # Only rotate wrt to y-axis


def get_orientation(orig, deg):
    """
    Get the camerat orientation after rotating by angle deg.
    :param orig: orientation matrix before rotation -> 3x3 matrix
    :param deg: angle of rotation in deg
    :return: orientation matrix after rotation -> 3x3 matrix
        The returned matrix:
        1st-row -> x-axis dir
        2nd-row -> y-axis dir
        3rd-row -> z-axis dir
    """
    y_axis = np.array([0,1,0])
    q = quat(y_axis, deg)
    return np.dot(quat2rot(q), orig)


def draw_3d(pts):
    plt.cla()
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    fig.canvas.draw()


def perspective(p, translation, orientation):
    # TODO check the orientation index in u,v calculation: what exactly is camera optical axis
    u0, v0, bu, bv, ku, kv, f = 0, 0, 1, 1, 1, 1, 1
    # p,translation,orientation = p.astype('float32'),translation.astype('float32'),orientation.astype('float32')
    u = (f * np.dot((p - translation), orientation[0].T) * bu) / np.dot((p - translation), orientation[2].T) + u0
    v = (f * np.dot((p - translation), orientation[1].T) * bu) / np.dot((p - translation), orientation[2].T) + v0
    return u,v


def orthographic(p, translation, orientation):
    u0, v0, bu, bv = 0, 0, 1, 1
    u = np.dot((p - translation), orientation[0].T) * bu + u0
    v = np.dot((p - translation), orientation[1].T) * bv + v0
    return u,v



def get_sample_2():

    def create_intermediate_points(pt1, pt2, granularity):
        new_pts = []
        vector = np.array([(x[0] - x[1]) for x in zip(pt1, pt2)])
        return [(np.array(pt2) + (vector * (float(i)/granularity))) for i in range(1, granularity)]

    pts = []
    granularity = 20

    # Create cube wireframe
    pts.extend([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], \
              [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])

    pts.extend(create_intermediate_points([-1, -1, 1], [1, -1, 1], granularity))
    pts.extend(create_intermediate_points([1, -1, 1], [1, 1, 1], granularity))
    pts.extend(create_intermediate_points([1, 1, 1], [-1, 1, 1], granularity))
    pts.extend(create_intermediate_points([-1, 1, 1], [-1, -1, 1], granularity))

    pts.extend(create_intermediate_points([-1, -1, -1], [1, -1, -1], granularity))
    pts.extend(create_intermediate_points([1, -1, -1], [1, 1, -1], granularity))
    pts.extend(create_intermediate_points([1, 1, -1], [-1, 1, -1], granularity))
    pts.extend(create_intermediate_points([-1, 1, -1], [-1, -1, -1], granularity))

    pts.extend(create_intermediate_points([1, 1, 1], [1, 1, -1], granularity))
    pts.extend(create_intermediate_points([1, -1, 1], [1, -1, -1], granularity))
    pts.extend(create_intermediate_points([-1, -1, 1], [-1, -1, -1], granularity))
    pts.extend(create_intermediate_points([-1, 1, 1], [-1, 1, -1], granularity))

    # Create triangle wireframe
    pts.extend([[-0.5, -0.5, -1], [0.5, -0.5, -1], [0, 0.5, -1]])
    pts.extend(create_intermediate_points([-0.5, -0.5, -1], [0.5, -0.5, -1], granularity))
    pts.extend(create_intermediate_points([0.5, -0.5, -1], [0, 0.5, -1], granularity))
    pts.extend(create_intermediate_points([0, 0.5, -1], [-0.5, -0.5, -1], granularity))

    return np.array(pts)


def draw(pts,frame):
    plt.subplot(2,2,frame)
    ax = plt.gca()
    for i, (x,y) in enumerate(pts):
        plt.scatter(x,y)
        ax.annotate(i, (x+0.005,y+0.005))
    plt.title('Frame {}'.format(frame))

def draw2(pts,frame):
    plt.subplot(2,2,frame)
    for i, (x,y) in enumerate(pts):
        plt.scatter(x,y)
    plt.title('Frame {}'.format(frame))


def main():
    # pts = get_sample()  # Test data points
    pts = get_sample_2()
    # pts = get_cube()
    rot_angle_per_frame = 30
    frames = range(1,5)


    # Original camera orientation and position
    orientation = np.identity(3)  # Original orientation
    translation = np.array([0,0,-5])

    for frame in frames:
        print('{}\nFrame {}\n{}'.format('='*50, frame, '='*50))
        perspective_pts = []
        X = Y = []

        if frame > 1:
            translation = get_translation(translation, -rot_angle_per_frame)
            # NOTE: rot_angle_per_frame is negated below because we're on camera coord system
            orientation = get_orientation(orientation, rot_angle_per_frame)

        for p in pts:
            x, y = perspective(p, translation, orientation)
            # x, y = orthographic(p, translation,orientation)
            perspective_pts.append((x,y))
            X.append(x)
            Y.append(y)
            print('\t{:16} -> ({:6.3f}, {:6.3f})'.format(p, x, y))

        draw2(perspective_pts,frame)


if __name__ == '__main__':
    main()
    plt.show()