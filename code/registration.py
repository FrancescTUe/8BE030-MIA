"""
Registration module main code.
"""

import numpy as np
from scipy import ndimage
import registration_util as util


# SECTION 1. Geometrical transformations


def identity():
    # 2D identity matrix.
    # Output:
    # T - transformation matrix

    T = np.eye(2)

    return T


def scale(sx, sy):
    # 2D scaling matrix.
    # Input:
    # sx, sy - scaling parameters
    # Output:
    # T - transformation matrix

    T = np.array([[sx,0],[0,sy]])

    return T


def rotate(phi):
    # 2D rotation matrix.
    # Input:
    # phi - rotation angle
    # Output:
    # T - transformation matrix

    #------------------------------------------------------------------#
    # TODO: Implement transformation matrix for rotation.
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    T = np.array([[cos_p, -sin_p],[sin_p,  cos_p]]) # Completed rotation matrix
    #------------------------------------------------------------------#
    return T


def shear(cx, cy):
    # 2D shearing matrix.
    # Input:
    # cx - horizontal shear
    # cy - vertical shear
    # Output:
    # T - transformation matrix

    #------------------------------------------------------------------#
    # TODO: Implement transformation matrix for shear.
    T = np.array([[1,cx],[cy,1]]) # completed shear function
    #------------------------------------------------------------------#

    return T


def reflect(rx, ry):
    # 2D reflection matrix.
    # Input:
    # rx - horizontal reflection (must have value of -1 or 1)
    # ry - vertical reflection (must have value of -1 or 1)
    # Output:
    # T - transformation matrix

    allowed = [-1, 1]
    if rx not in allowed or ry not in allowed:
        T = 'Invalid input parameter'
        return T

    #------------------------------------------------------------------#
    # TODO: Implement transformation matrix for reflection
    T = np.array([[rx,0],[0,ry]]) # completed reflection matrix
    #------------------------------------------------------------------#

    return T


# SECTION 2. Image transformation and least squares fitting


def image_transform(I, Th,  output_shape=None):
    # Image transformation by inverse mapping.
    # Input:
    # I - image to be transformed
    # Th - homogeneous transformation matrix
    # output_shape - size of the output image (default is same size as input)
    # Output:
    # It - transformed image
	# Xt - remapped coordinates
    # we want double precision for the interpolation, but we want the
    # output to have the same data type as the input - so, we will
    # convert to double and remember the original input type

    input_type = type(I)

    # default output size is same as input
    if output_shape is None:
        output_shape = I.shape

    # spatial coordinates of the transformed image
    x = np.arange(0, output_shape[1])
    y = np.arange(0, output_shape[0])
    xx, yy = np.meshgrid(x, y)

    # convert to a 2-by-p matrix (p is the number of pixels)
    X = np.concatenate((xx.reshape((1, xx.size)), yy.reshape((1, yy.size))))
    # convert to homogeneous coordinates
    Xh = util.c2h(X)

    #------------------------------------------------------------------#
    # TODO: Perform inverse coordinates mapping.

    Xt_h = np.dot(Th, Xh)

    Xt = Xt_h[:2, :] / Xt_h[2, :] # we transform from homogeneous to cartesian

    #------------------------------------------------------------------#

    It = ndimage.map_coordinates(I, [Xt[1,:], Xt[0,:]], order=1, mode='constant').reshape(output_shape)

    return It, Xt


def ls_solve(A, b):
    # Least-squares solution to a linear system of equations.
    # Input:
    # A - matrix of known coefficients
    # b - vector of known constant term
    # Output:
    # w - least-squares solution to the system of equations
    # E - squared error for the optimal solution

    #------------------------------------------------------------------#
    # TODO: Implement the least-squares solution for w.

    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    #------------------------------------------------------------------#

    # compute the error
    E = np.transpose(A.dot(w) - b).dot(A.dot(w) - b)

    return w, E


def ls_affine(X, Xm):
    # Least-squares fitting of an affine transformation.
    # Input:
    # X - Points in the fixed image
    # Xm - Corresponding points in the moving image
    # Output:
    # T - affine transformation in homogeneous form.

    A = np.transpose(Xm)

    #------------------------------------------------------------------#
    # TODO: Implement least-squares fitting of an affine transformation.
    # Use the ls_solve() function that you have previously implemented.

    ones = np.ones((A.shape[0], 1))
    A = np.hstack((A, ones)) # We initialize the design matrix for the affine transformation

    w1, E1 = ls_solve(A, X[0, :]) # We solve for the first row of A (x-map)
    w2, E3 = ls_solve(A, X[1, :]) # We solve for the second row of A (y-map)

    # We now construct the transformation matrix from the coefficients:
    #[[a,b,tx],[c,d,ty],[0,0,1]]
    T = np.array([
            w1,
            w2,
            [0, 0, 1]
        ])
    
    return T
    #------------------------------------------------------------------#

# SECTION 3. Image simmilarity metrics


def correlation(I, J):
    # Compute the normalized cross-correlation between two images.
    # Input:
    # I, J - input images
    # Output:
    # CC - normalized cross-correlation
    # it's always good to do some parameter checks

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    u = I.reshape((I.shape[0]*I.shape[1],1))
    v = J.reshape((J.shape[0]*J.shape[1],1))

    # subtract the mean
    u = u - u.mean(keepdims=True)
    v = v - v.mean(keepdims=True)

    #------------------------------------------------------------------#
    # TODO: Implement the computation of the normalized cross-correlation.
    # This can be done with a single line of code, but you can use for-loops instead.

    CC = np.vdot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)) # We compute the dot product between u and v and divide my the product of their norms

    #------------------------------------------------------------------#

    return CC


def joint_histogram(I, J, num_bins=16, minmax_range=None):
    # Compute the joint histogram of two signals.
    # Input:
    # I, J - input images
    # num_bins: number of bins of the joint histogram (default: 16)
    # range - range of the values of the signals (defaul: min and max
    # of the inputs)
    # Output:
    # p - joint histogram

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    # make sure the inputs are column-vectors of type double (highest
    # precision)
    I = I.reshape((I.shape[0]*I.shape[1],1)).astype(float)
    J = J.reshape((J.shape[0]*J.shape[1],1)).astype(float)

    # if the range is not specified use the min and max values of the
    # inputs
    if minmax_range is None:
        minmax_range = np.array([min(min(I),min(J)), max(max(I),max(J))])

    # this will normalize the inputs to the [0 1] range
    I = (I-minmax_range[0]) / (minmax_range[1]-minmax_range[0])
    J = (J-minmax_range[0]) / (minmax_range[1]-minmax_range[0])

    # and this will make them integers in the [0 (num_bins-1)] range
    I = np.round(I*(num_bins-1)).astype(int)
    J = np.round(J*(num_bins-1)).astype(int)

    n = I.shape[0]
    hist_size = np.array([num_bins,num_bins])

    # initialize the joint histogram to all zeros
    p = np.zeros(hist_size)

    for k in range(n):
        p[I[k], J[k]] = p[I[k], J[k]] + 1

    #------------------------------------------------------------------#
    # TODO: At this point, p contains the counts of cooccuring
    # intensities in the two images. You need to implement one final
    # step to make p take the form of a probability mass function
    # (p.m.f.).

    p = p / n # We normalize by the total number of pixels so that the sum is 1

    #------------------------------------------------------------------#

    return p


def mutual_information(p):
    # Compute the mutual information from a joint histogram.
    # Input:
    # p - joint histogram
    # Output:
    # MI - mutual information in nat units
    # a very small positive number

    EPSILON = 10e-10

    # add a small positive number to the joint histogram to avoid
    # numerical problems (such as division by zero)
    p += EPSILON

    # we can compute the marginal histograms from the joint histogram
    p_I = np.sum(p, axis=1)
    p_I = p_I.reshape(-1, 1)
    p_J = np.sum(p, axis=0)
    p_J = p_J.reshape(1, -1)

    #------------------------------------------------------------------#
    # TODO: Implement the computation of the mutual information from p,
    # p_I and p_J. This can be done with a single line of code, but you
    # can use a for-loop instead.
    # HINT: p_I is a column-vector and p_J is a row-vector so their
    # product is a matrix. You can also use the sum() function here.

    MI = np.sum(p*np.log(p/(p_I*p_J)))

    #------------------------------------------------------------------#

    return MI


def mutual_information_e(p):
    # Compute the mutual information from a joint histogram.
    # Alternative implementation via computation of entropy.
    # Input:
    # p - joint histogram
    # Output:
    # MI - mutual information in nat units
    # a very small positive number

    EPSILON = 10e-10

    # add a small positive number to the joint histogram to avoid
    # numerical problems (such as division by zero)
    p += EPSILON

    # we can compute the marginal histograms from the joint histogram
    p_I = np.sum(p, axis=1)
    p_I = p_I.reshape(-1, 1)
    p_J = np.sum(p, axis=0)
    p_J = p_J.reshape(1, -1)

    #------------------------------------------------------------------#
    # TODO: Implement the computation of the mutual information via
    # computation of entropy.
    # MI formula is: MI = H(I) + H(J) - H(I,J)
    H_I = -np.sum(p_I*np.log(p_I))
    H_J = -np.sum(p_J*np.log(p_J))
    H_IJ = -np.sum(p*np.log(p))
    MI = H_I + H_J - H_IJ
    #------------------------------------------------------------------#

    return MI


# SECTION 4. Towards intensity-based image registration


def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    # Input:
    # fun - function for which the gradient is computed
    # x - vector of parameter values at which to compute the gradient
    # h - a small positive number used in the finite difference formula
    # Output:
    # g - vector of partial derivatives (gradient) of fun

    g = np.zeros_like(x)

    #------------------------------------------------------------------#
    # TODO: Implement the  computation of the partial derivatives of
    # the function at x with numerical differentiation.
    # g[k] should store the partial derivative w.r.t. the k-th parameter

    for k in range(len(x)):
        x_p = x.copy()
        x_m = x.copy()

        # We perturb the kth element by a small amount
        x_p[k] += h
        x_m[k] -= h

        # We use the central difference formula: [f(x+h) - f(x-h)]/(2*h)
        g[k] = (fun(x_p)-fun(x_m))/(2*h)

    #------------------------------------------------------------------#

    return g


def rigid_corr(I, Im, x, return_transform=True):
    # Computes normalized cross-correlation between a fixed and
    # a moving image transformed with a rigid transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)

    SCALING = 100

    # the first element is the rotation angle
    T = rotate(x[0])

    # the remaining two element are the translation
    #
    # the gradient ascent/descent method work best when all parameters
    # of the function have approximately the same range of values
    # this is  not the case for the parameters of rigid registration
    # where the transformation matrix usually takes  much smaller
    # values compared to the translation vector this is why we pass a
    # scaled down version of the translation vector to this function
    # and then scale it up when computing the transformation matrix
    Th = util.t2h(T, x[1:]*SCALING)

    # transform the moving image
    Im_t, Xt = image_transform(Im, Th)

    # compute the similarity between the fixed and transformed
    # moving image
    C = correlation(I, Im_t)

    if return_transform:
        return C, Im_t, Th
    else:
        return C


def affine_corr(I, Im, x, return_transform=True):
    # Computes normalized cross-corrleation between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the roation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)
    
    NUM_BINS = 64
    SCALING = 100

    #------------------------------------------------------------------#
    # TODO: Implement the missing functionality

    # We first extract all parameters from x
    phi = x[0] # Rotation angle
    sx, sy = x[1], x[2] # Scaling factors
    shx, shy = x[3], x[4]# Shearing factors
    tx, ty = x[5], x[6] # Translation factors

    # We generate each component and compose the linear transformation matrix
    R = rotate(phi)
    Sh = shear(shx, shy)
    S = np.array([[sx, 0], [0, sy]])
    
    A = Sh@S@R

    # We then create the homoeneous function
    Th = util.t2h(A, np.array([[tx], [ty]]))

    # We transform the moving image using Th
    Im_t, Xt = image_transform(Im, Th)

    # We compute the similarity metric
    C = correlation(I, Im_t)
    #------------------------------------------------------------------#

    if return_transform:
        return C, Im_t, Th
    else:
        return C


def affine_mi(I, Im, x, return_transform=True):
    # Computes mutual information between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)

    NUM_BINS = 64
    SCALING = 100
    
    #------------------------------------------------------------------#
    # TODO: Implement the missing functionality
    # We first extract all parameters from x
    phi = x[0] # Rotation angle
    sx, sy = x[1], x[2] # Scaling factors
    shx, shy = x[3], x[4]# Shearing factors
    tx, ty = x[5], x[6] # Translation factors

    # We generate each component and compose the linear transformation matrix
    R = rotate(phi)
    Sh = shear(shx, shy)
    S = np.array([[sx, 0], [0, sy]])
    
    A = Sh@S@R

    # We then create the homoeneous function
    Th = util.t2h(A, np.array([[tx], [ty]]))

    # We transform the moving image using Th
    Im_t, Xt = image_transform(Im, Th)

    # We compute the similarity metric:
    p = joint_histogram(I, Im_t) # joint_histogram
    MI = mutual_information(p)
    #------------------------------------------------------------------#

    if return_transform:
        return MI, Im_t, Th
    else:
        return MI