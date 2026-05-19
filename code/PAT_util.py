import numpy as np
import registration as reg
import registration_util as util

def compute_centre_of_mass(I):
    """
    Compute the intensity-weighted centre of mass of a 2D image.

    Returns
    -------
    com : ndarray, shape (2,)
        [cx, cy] where cx is the weighted mean column index
        and cy is the weighted mean row index.
    """
    rows, cols = np.indices(I.shape)   # rows[r,c]=r,  cols[r,c]=c
    total = I.sum()
    cx = (I * cols).sum() / total
    cy = (I * rows).sum() / total
    return np.array([cx, cy])

def compute_second_moment_matrix(I, com):
    """
    Compute the 2x2 intensity-weighted second moment (inertia) matrix.

    Parameters
    ----------
    I   : 2D ndarray — image
    com : ndarray shape (2,) — [cx, cy] from compute_centre_of_mass

    Returns
    -------
    M : ndarray, shape (2, 2)
    """
    cx, cy = com
    rows, cols = np.indices(I.shape)
    total = I.sum()

    dx = cols - cx   # column displacement  (x direction)
    dy = rows - cy   # row displacement     (y direction)

    mu20 = (I * dx**2).sum() / total
    mu02 = (I * dy**2).sum() / total
    mu11 = (I * dx * dy).sum() / total

    return np.array([[mu20, mu11],
                     [mu11, mu02]])

def compute_principal_axes(M):
    """
    Compute principal axes from a 2x2 second moment matrix.

    Returns
    -------
    S : ndarray, shape (2, 2)
        Columns are eigenvectors sorted by descending eigenvalue.
        S[:, 0] is the major (principal) axis.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # eigh returns ascending order — reverse to get descending
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx]

def principal_axes_transform(I, J):
    """
    Compute the PAT homogeneous matrix that registers moving image J to fixed image I.

    Returns
    -------
    Th_best : ndarray, shape (3, 3)
        Homogeneous transformation matrix for use with image_transform.
        Maps fixed-image coordinates to moving-image coordinates (inverse mapping).
    """
    # -- centres of mass --
    com_I = compute_centre_of_mass(I)
    com_J = compute_centre_of_mass(J)

    # -- second moment matrices --
    M_I = compute_second_moment_matrix(I, com_I)
    M_J = compute_second_moment_matrix(J, com_J)

    # -- principal axes --
    S_I = compute_principal_axes(M_I)
    S_J = compute_principal_axes(M_J)

    # -- rotation that maps J's axes onto I's axes --
    R = S_I @ S_J.T

    # report rotation angle
    theta = np.arctan2(R[1, 0], R[0, 0])
    print(f'Rotation angle: {theta:.4f} rad  ({np.degrees(theta):.2f} deg)')

    def build_Th(R_mat):
        # inverse mapping: fixed -> moving
        R_inv = R_mat.T
        t_inv = com_J - R_inv @ com_I
        return util.t2h(R_inv, t_inv)

    Th      = build_Th(R)
    Th_flip = build_Th(-R)   # 180-degree flipped solution

    J_t,      _ = reg.image_transform(J, Th)
    J_t_flip, _ = reg.image_transform(J, Th_flip)

    ncc      = reg.correlation(I, J_t)
    ncc_flip = reg.correlation(I, J_t_flip)

    if ncc >= ncc_flip:
        return Th
    else:
        theta_flip = np.arctan2(-R[1, 0], -R[0, 0])
        return Th_flip
    
def color_overlay(fixed_im, moving_im):
    fixed_norm = (fixed_im - fixed_im.min()) / (fixed_im.max() - fixed_im.min()) # Norm of fixed image
    moving_norm = (moving_im - moving_im.min()) / (moving_im.max() - moving_im.min()) # Norm of moving image
    
    overlay = np.zeros((fixed_im.shape[0], fixed_im.shape[1], 3))
    overlay[..., 0] = moving_norm  # Red
    overlay[..., 1] = fixed_norm   # Green
    overlay[..., 2] = moving_norm  # Blue
    
    return overlay  

def params_to_Th(x):
    """
    Convert parameter vector x = [angle, scale_x, scale_y, shear_x, shear_y, translation_x, translation_y] to a
    homogeneous affine transformation matrix Th.
    """
    phi = x[0] # Rotation angle
    sx, sy = x[1], x[2] # Scaling factors
    shx, shy = x[3], x[4]# Shearing factors
    tx, ty = x[5], x[6] # Translation factors

    # We generate each component and compose the linear transformation matrix
    R = reg.rotate(phi)
    Sh = reg.shear(shx, shy)
    S = np.array([[sx, 0], [0, sy]])
    
    A = Sh@S@R

    # We then create the homoeneous function
    Th = util.t2h(A, np.array([[tx], [ty]]))
    return Th

def gradient_ascent_registration(I, J, x_init, similarity_fn,
                                   n_iterations=50, learning_rate=1e-3, tol=1e-4):
    """
    Intensity-based registration via gradient ascent on similarity_fn.

    Parameters
    ----------
    I, J          : fixed and moving images 
    x_init        : initial parameter vector [rot angle, sx, sy, shx, shy, tx, ty]
    similarity_fn : similarity function
    n_iterations  : number of gradient ascent steps
    learning_rate : step size
    tol           : convergence tolerance for similarity change

    Returns
    -------
    x_opt   : final parameter vector
    history : list of similarity scores, one per iteration
    """
    x = x_init.copy().astype(np.float64)
    history = []

    current_sim = similarity_fn(x)
    history.append(current_sim)

    print(f'  Initial similarity : {current_sim:.4f}')
    print(f'  Initial parameters : angle={np.degrees(x[0]):.2f} deg, '
          f'sx={x[1]:.4f}, sy={x[2]:.4f}, shx={x[3]:.4f}, shy={x[4]:.4f}, tx={x[5]:.4f}, ty={x[6]:.4f}')

    for i in range(n_iterations):
         g = reg.ngradient(similarity_fn, x)
         step = learning_rate * g
         x = x + step

         new_sim = similarity_fn(x)
         history.append(new_sim)
       
         sim_change = abs(new_sim - current_sim)
         if sim_change < tol:
            print(f' Converged at iteration {i+1}')
            break
        
         current_sim = new_sim

    print(f'  Final similarity   : {history[-1]:.4f}')
    print(f'  Final parameters   : angle={np.degrees(x[0]):.2f} deg, '
          f'sx={x[1]:.4f}, sy={x[2]:.4f}, shx={x[3]:.4f}, shy={x[4]:.4f}, tx={x[5]:.4f}, ty={x[6]:.4f}')
    return x, history