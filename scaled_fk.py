import numpy as np
from math import cos, sin
import casadi as ca

# Scaled-down UR5e parameters (1/2 scaling)
d1 = 0.163 / 2
a2 = -0.425 / 2
a3 = -0.392 / 2
d4 = 0.127 / 2
d5 = 0.1 / 2
d6 = 0.1 / 2

# Mounting transformation
car_to_base_translation = np.array([-0.05, 0.0, 0.2])  # x, y, z
car_to_base_rotation = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
car_to_base_homogeneous = np.eye(4)
car_to_base_homogeneous[:3, :3] = car_to_base_rotation
car_to_base_homogeneous[:3, 3] = car_to_base_translation

def forward(q):
    s = [sin(q[0]), sin(q[1]), sin(q[2]), sin(q[3]), sin(q[4]), sin(q[5])]
    c = [cos(q[0]), cos(q[1]), cos(q[2]), cos(q[3]), cos(q[4]), cos(q[5])]

    q23 = q[1]+q[2]
    q234 = q[1]+q[2]+q[3]

    s23 = sin(q23)
    c23 = cos(q23)
    s234 = sin(q234)
    c234 = cos(q234)
    T = np.matrix(np.identity(4))
    T[0, 0] = c234*c[0]*s[4] - c[4]*s[0]
    T[0, 1] = c[5]*(s[0]*s[4] + c234*c[0]*c[4]) - s234*c[0]*s[5]
    T[0, 2] = -s[5]*(s[0]*s[4] + c234*c[0]*c[4]) - s234*c[0]*c[5]
    T[0, 3] = d6*c234*c[0]*s[4] - a3*c23*c[0] - a2*c[0]*c[1] - d6*c[4]*s[0] - d5*s234*c[0] - d4*s[0]
    T[1, 0] = c[0]*c[4] + c234*s[0]*s[4]
    T[1, 1] = -c[5]*(c[0]*s[4] - c234*c[4]*s[0]) - s234*s[0]*s[5]
    T[1, 2] = s[5]*(c[0]*s[4] - c234*c[4]*s[0]) - s234*c[5]*s[0]
    T[1, 3] = d6*(c[0]*c[4] + c234*s[0]*s[4]) + d4*c[0] - a3*c23*s[0] - a2*c[1]*s[0] - d5*s234*s[0]
    T[2, 0] = -s234*s[4]
    T[2, 1] = -c234*s[5] - s234*c[4]*c[5]
    T[2, 2] = s234*c[4]*s[5] - c234*c[5]
    T[2, 3] = d1 + a3*s23 + a2*s[1] - d5*(c23*c[3] - s23*s[3]) - d6*s[4]*(c23*s[3] + s23*c[3])
    T[3, 0] = 0.0
    T[3, 1] = 0.0
    T[3, 2] = 0.0
    T[3, 3] = 1.0

    # Apply the mounting transformation
    T = car_to_base_homogeneous @ T
    return T

def forward_all_links(q):
    """
    Compute the transformation matrices for all links of the scaled-down UR5e arm.
    :param q: List of 6 joint angles [q1, q2, q3, q4, q5, q6].
    :return: List of 4x4 transformation matrices [T1, T2, T3, T4, T5, T6].
    """
    s1, c1 = sin(q[0]), cos(q[0])
    q23 = q[1] + q[2]
    q234 = q23 + q[3]
    s2, c2 = sin(q[1]), cos(q[1])
    s3, c3 = sin(q[2]), cos(q[2])
    s23, c23 = sin(q23), cos(q23)
    s234, c234 = sin(q234), cos(q234)
    s5, c5 = sin(q[4]), cos(q[4])
    s6, c6 = sin(q[5]), cos(q[5])

    # Initialize transformation matrices
    T1 = np.eye(4)
    T2 = np.eye(4)
    T3 = np.eye(4)
    T4 = np.eye(4)
    T5 = np.eye(4)
    T6 = np.eye(4)

    # Compute T1
    T1[:3, :3] = [[c1, 0, s1], [s1, 0, -c1], [0, 1, 0]]
    T1[:3, 3] = [0, 0, d1]

    # Compute T2
    T2[:3, :3] = [[c1 * c2, -c1 * s2, s1],
                  [c2 * s1, -s1 * s2, -c1],
                  [s2, c2, 0]]
    T2[:3, 3] = [a2 * c1 * c2, a2 * c2 * s1, d1 + a2 * s2]

    # Compute T3
    T3[:3, :3] = [[c23 * c1, -s23 * c1, s1],
                  [c23 * s1, -s23 * s1, -c1],
                  [s23, c23, 0]]
    T3[:3, 3] = [c1 * (a3 * c23 + a2 * c2),
                 s1 * (a3 * c23 + a2 * c2),
                 d1 + a3 * s23 + a2 * s2]

    # Compute T4
    T4[:3, :3] = [[c234 * c1, s1, s234 * c1],
                  [c234 * s1, -c1, s234 * s1],
                  [s234, 0, -c234]]
    T4[:3, 3] = [c1 * (a3 * c23 + a2 * c2) + d4 * s1,
                 s1 * (a3 * c23 + a2 * c2) - d4 * c1,
                 d1 + a3 * s23 + a2 * s2]

    # Compute T5
    T5[:3, :3] = [[s1 * s5 + c234 * c1 * c5, -s234 * c1, c5 * s1 - c234 * c1 * s5],
                  [c234 * c5 * s1 - c1 * s5, -s234 * s1, -c1 * c5 - c234 * s1 * s5],
                  [s234 * c5, c234, -s234 * s5]]
    T5[:3, 3] = [c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1,
                 s1 * (a3 * c23 + a2 * c2) - d4 * c1 + d5 * s234 * s1,
                 d1 + a3 * s23 + a2 * s2 - d5 * c234]

    # Compute T6
    T6[:3, :3] = [[c6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * s6,
                   -s6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * c6,
                   c5 * s1 - c234 * c1 * s5],
                  [-c6 * (c1 * s5 - c234 * c5 * s1) - s234 * s1 * s6,
                   s6 * (c1 * s5 - c234 * c5 * s1) - s234 * c6 * s1,
                   -c1 * c5 - c234 * s1 * s5],
                  [c234 * s6 + s234 * c5 * c6,
                   c234 * c6 - s234 * c5 * s6,
                   -s234 * s5]]
    T6[:3, 3] = [d6 * (c5 * s1 - c234 * c1 * s5) + c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1,
                 s1 * (a3 * c23 + a2 * c2) - d4 * c1 - d6 * (c1 * c5 + c234 * s1 * s5) + d5 * s234 * s1,
                 d1 + a3 * s23 + a2 * s2 - d5 * c234 - d6 * s234 * s5]
    
    # Flip the sign of x and y axes only of the transformation matrices
    T1[0, 3] *= -1
    T2[0, 3] *= -1
    T3[0, 3] *= -1
    T4[0, 3] *= -1
    T5[0, 3] *= -1
    T6[0, 3] *= -1
    T1[1, 3] *= -1
    T2[1, 3] *= -1
    T3[1, 3] *= -1
    T4[1, 3] *= -1
    T5[1, 3] *= -1
    T6[1, 3] *= -1
    
    # Apply the mounting transformation
    T1 = car_to_base_homogeneous @ T1
    T2 = car_to_base_homogeneous @ T2
    T3 = car_to_base_homogeneous @ T3
    T4 = car_to_base_homogeneous @ T4
    T5 = car_to_base_homogeneous @ T5
    T6 = car_to_base_homogeneous @ T6

    return [T1, T2, T3, T4, T5, T6]


def casadi_forward_all_links(q):
    """
    Compute the transformation matrices for all links of the scaled-down UR5e arm using CasADi.
    :param q: List of 6 joint angles [q1, q2, q3, q4, q5, q6].
    :param car_to_base_homogeneous: 4x4 mounting transformation matrix (numpy array).
    :param d1, a2, a3, d4, d5, d6: Robot-specific DH parameters (floats).
    :return: List of 4x4 transformation matrices [T1, T2, T3, T4, T5, T6].
    """

    global car_to_base_homogeneous

    # Define shorthand for CasADi trigonometric functions
    sin, cos = ca.sin, ca.cos

    # Joint angle sums for compact expressions
    q23 = q[1] + q[2]
    q234 = q23 + q[3]

    # Precompute sines and cosines
    s1, c1 = sin(q[0]), cos(q[0])
    s2, c2 = sin(q[1]), cos(q[1])
    s23, c23 = sin(q23), cos(q23)
    s234, c234 = sin(q234), cos(q234)
    s5, c5 = sin(q[4]), cos(q[4])
    s6, c6 = sin(q[5]), cos(q[5])

    # Helper function to build transformation matrices
    def create_transformation(R, t):
        """
        Create a 4x4 transformation matrix from a rotation matrix and translation vector.
        :param R: 3x3 rotation matrix (CasADi MX).
        :param t: 3x1 translation vector (CasADi MX).
        :return: 4x4 transformation matrix (CasADi MX).
        """
        T = ca.MX.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    # Compute T1
    R1 = ca.horzcat(
        ca.vertcat(c1, s1, 0),
        ca.vertcat(0, 0, 1),
        ca.vertcat(s1, -c1, 0)
    )
    t1 = ca.vertcat(0, 0, d1)
    T1 = create_transformation(R1, t1)

    # Compute T2
    R2 = ca.horzcat(
        ca.vertcat(c1 * c2, s1 * c2, s2),
        ca.vertcat(-c1 * s2, -s1 * s2, c2),
        ca.vertcat(s1, -c1, 0)
    )
    t2 = ca.vertcat(a2 * c1 * c2, a2 * s1 * c2, d1 + a2 * s2)
    T2 = create_transformation(R2, t2)

    # Compute T3
    R3 = ca.horzcat(
        ca.vertcat(c23 * c1, s23 * c1, s1),
        ca.vertcat(c23 * s1, s23 * s1, -c1),
        ca.vertcat(s23, c23, 0)
    )
    t3 = ca.vertcat(
        c1 * (a3 * c23 + a2 * c2),
        s1 * (a3 * c23 + a2 * c2),
        d1 + a3 * s23 + a2 * s2
    )
    T3 = create_transformation(R3, t3)

    # Compute T4
    R4 = ca.horzcat(
        ca.vertcat(c234 * c1, c234 * s1, s234),
        ca.vertcat(s1, -c1, 0),
        ca.vertcat(s234 * c1, s234 * s1, -c234)
    )
    t4 = ca.vertcat(
        c1 * (a3 * c23 + a2 * c2) + d4 * s1,
        s1 * (a3 * c23 + a2 * c2) - d4 * c1,
        d1 + a3 * s23 + a2 * s2
    )
    T4 = create_transformation(R4, t4)

    # Compute T5
    R5 = ca.horzcat(
        ca.vertcat(s1 * s5 + c234 * c1 * c5, c234 * c5 * s1 - c1 * s5, s234 * c5),
        ca.vertcat(-s234 * c1, -s234 * s1, c234),
        ca.vertcat(c5 * s1 - c234 * c1 * s5, -c1 * c5 - c234 * s1 * s5, -s234 * s5)
    )
    t5 = ca.vertcat(
        c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1,
        s1 * (a3 * c23 + a2 * c2) - d4 * c1 + d5 * s234 * s1,
        d1 + a3 * s23 + a2 * s2 - d5 * c234
    )
    T5 = create_transformation(R5, t5)

    # Compute T6
    R6 = ca.horzcat(
        ca.vertcat(
            c6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * s6,
            -c6 * (c1 * s5 - c234 * c5 * s1) - s234 * s1 * s6,
            c234 * s6 + s234 * c5 * c6
        ),
        ca.vertcat(
            -s6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * c6,
            s6 * (c1 * s5 - c234 * c5 * s1) - s234 * c6 * s1,
            c234 * c6 - s234 * c5 * s6
        ),
        ca.vertcat(
            c5 * s1 - c234 * c1 * s5,
            -c1 * c5 - c234 * s1 * s5,
            -s234 * s5
        )
    )
    t6 = ca.vertcat(
        d6 * (c5 * s1 - c234 * c1 * s5) + c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1,
        s1 * (a3 * c23 + a2 * c2) - d4 * c1 - d6 * (c1 * c5 + c234 * s1 * s5) + d5 * s234 * s1,
        d1 + a3 * s23 + a2 * s2 - d5 * c234 - d6 * s234 * s5
    )
    T6 = create_transformation(R6, t6)

    # Flip the sign of x and y axes only of the transformation matrices
    for T in [T1, T2, T3, T4, T5, T6]:
        T[0, 3] *= -1
        T[1, 3] *= -1

    # Apply the mounting transformation
    car_to_base_homogeneous = ca.MX(car_to_base_homogeneous)  # Convert to CasADi matrix
    T1 = car_to_base_homogeneous @ T1
    T2 = car_to_base_homogeneous @ T2
    T3 = car_to_base_homogeneous @ T3
    T4 = car_to_base_homogeneous @ T4
    T5 = car_to_base_homogeneous @ T5
    T6 = car_to_base_homogeneous @ T6

    return [T1, T2, T3, T4, T5, T6]
