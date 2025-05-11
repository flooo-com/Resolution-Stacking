import numpy as np
from numba import njit, prange
from tqdm import trange
import rawpy as rp

def transform_imageCoords(SrcImage_XY: np.ndarray, TrgImage_XY: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms source image coordinates to target image coordinates using a given homography matrix.
    
    Parameters:
    -----------
    SrcImage_XY : numpy.ndarray
        A 2D array with shape (2, M, N) representing the X and Y coordinates of the source image.
        - SrcImage_XY[0]: 2D array of X-coordinates.
        - SrcImage_XY[1]: 2D array of Y-coordinates.
    H : numpy.ndarray
        A 3x3 homography matrix used to transform the source image coordinates to the target image coordinates.
    
    Returns:
    --------
    numpy.ndarray
        TrgImage_XY_transformed: A 2D array with shape (2, M, N) containing the transformed X and Y coordinates.
    
    Notes:
    ------
    - The function assumes that the input coordinates are in homogeneous form and applies the transformation
      using the provided homography matrix.
    - The transformation is performed for each point in the source image coordinates, and the result is
      returned in the same shape as the input coordinates.
    """

    x = SrcImage_XY[0].flatten()
    y = SrcImage_XY[1].flatten()
    w = np.ones_like(x)

    N = len(x)
    print("Check1")
    X_Src = np.vstack([x, y, w])  # homogeneous Coordinates
    X_trg = np.zeros_like(X_Src).T

    for i in range(N):
        # Make transformation.
        X_trg_temp = H @ X_Src[:, i]
        X_trg[i, :] = X_trg_temp / X_trg_temp[2]

    TrgImage_XY_transformed = np.array([X_trg[:, 0].reshape(SrcImage_XY[0].shape), X_trg[:, 1].reshape(SrcImage_XY[0].shape)])  # Put X and Y layer back in image coord array

    print("Check2")

    residual_xy = np.sqrt((TrgImage_XY[0].ravel()-X_trg[:, 0])**2+(TrgImage_XY[1].ravel()-X_trg[:, 1])**2)

    return TrgImage_XY_transformed, residual_xy


def HomographyReprojection_error(SrcKeypoints_XY: np.ndarray, TrgKeypoints_XY: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the reprojection error for a set of source and target keypoints using a given homography matrix.

    Parameters:
    -----------
    SrcKeypoints_XY : list
        A list of two 1D arrays representing the X and Y coordinates of the source keypoints.
        - SrcKeypoints_XY[0]: 1D array of X-coordinates.
        - SrcKeypoints_XY[1]: 1D array of Y-coordinates.
    TrgKeypoints_XY : list
        A list of two 1D arrays representing the X and Y coordinates of the target keypoints.
        - TrgKeypoints_XY[0]: 1D array of X-coordinates.
        - TrgKeypoints_XY[1]: 1D array of Y-coordinates.
    H : numpy.ndarray
        A 3x3 homography matrix used to transform the source keypoints to the target keypoints.

    Returns:
    --------
    tuple:
        - TrgKeypoints_XY_transformed (numpy.ndarray): A 2D array with shape (2, N) containing the transformed X and Y coordinates.
        - residual_xy (numpy.ndarray): A 1D array with shape (N,) containing the Euclidean distance between the transformed and actual target keypoints.
        - residual_x (numpy.ndarray): A 1D array with shape (N,) containing the residuals in the X direction.
        - residual_y (numpy.ndarray): A 1D array with shape (N,) containing the residuals in the Y direction.

    Notes:
    ------
    - The residuals are computed as the difference between the transformed source keypoints and the actual target keypoints.
    """

    x = SrcKeypoints_XY[0]
    y = SrcKeypoints_XY[1]
    w = np.ones_like(x)

    N_keypoints = len(x)

    X_Src = np.vstack([x, y, w])  # homogeneous Coordinates
    X_trg = np.zeros_like(X_Src).T

    for i in range(N_keypoints):
        # Make transformation.
        X_trg_temp = H @ X_Src[:, i]
        X_trg[i, :] = X_trg_temp / X_trg_temp[2]

    TrgKeypoints_XY_transformed = np.array([X_trg[:, 0].reshape(SrcKeypoints_XY[0].shape), X_trg[:, 1].reshape(SrcKeypoints_XY[0].shape)])  # Put X and Y layer back in image coord array

    residual_x = TrgKeypoints_XY[0].ravel()-X_trg[:, 0]
    residual_y = TrgKeypoints_XY[1].ravel()-X_trg[:, 1]
    residual_xy = np.sqrt((TrgKeypoints_XY[0].ravel()-X_trg[:, 0])**2+(TrgKeypoints_XY[1].ravel()-X_trg[:, 1])**2)

    return TrgKeypoints_XY_transformed, residual_xy, residual_x, residual_y


@njit(parallel=True)
def transform_imageCoords_numba(SrcImage_XY: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Transforms source image coordinates to target image coordinates using a given homography matrix.
    This version parallelizes the transformation over each point for maximum CPU utilization.
    
    Parameters:
    -----------
    SrcImage_XY : numpy.ndarray
        A 3D array with shape (2, M, N) representing the X and Y coordinates of the source image.
    H : numpy.ndarray
        A 3x3 homography matrix.
    
    Returns:
    --------
    numpy.ndarray
        A 3D array with shape (2, M* N) containing the transformed X and Y coordinates as lists.
    """
    # Get image dimensions
    M = SrcImage_XY.shape[1]
    N = SrcImage_XY.shape[2]
    total = M * N

    # Flatten coordinate arrays for easier processing
    x = SrcImage_XY[0].ravel()
    y = SrcImage_XY[1].ravel()
    TrgImage_XY_transformed = np.empty((2, total))
    
    # Parallel loop over each point
    for idx in prange(total):

        x_at_point = x[idx] # one time acces at every point
        y_at_point = y[idx]
        
        # Compute homogeneous coordinate transformation
        w = H[2, 0] * x_at_point + H[2, 1] * y_at_point + H[2, 2] # scale factor for every point
        TrgImage_XY_transformed[0, idx] = (H[0, 0] * x_at_point + H[0, 1] * y_at_point + H[0, 2]) / w
        TrgImage_XY_transformed[1, idx] = (H[1, 0] * x_at_point + H[1, 1] * y_at_point + H[1, 2]) / w

    # Reshape back to original image dimensions
    return TrgImage_XY_transformed

@njit()
def transform_imagePixels(Trg_Image_coordinatesXY2D, Image):
    """
    Transforms the pixel values of an image based on target 2D coordinates.
    
    Args:
        Trg_Image_coordinatesXY2D (numpy.ndarray): A 3D array of shape (2, Nv, Nh) containing
            the target 2D coordinates for each pixel in the input image. The first dimension
            represents the x and y coordinates.
        Image (numpy.ndarray): A 2D array of shape (Nv, Nh) representing the input image.

    Returns:
        numpy.ndarray: A 2D array of shape (Nv, Nh) representing the transformed image, where
        pixel values are mapped to their new positions based on the target coordinates. Pixels
        that fall outside the valid range remain as NaN.
    
    Notes:
        The image coordiantes are define by the left upper corner of each pixel.
    """
    Nv, Nh = Image.shape
    img_coords = Trg_Image_coordinatesXY2D # Coords are H,V=X,Y
    Image_new = np.full((Nv, Nh), np.nan)

    for i in prange(Nv):
        for j in range(Nh):
            x = int(round(img_coords[0, i, j], 0)) # x is h
            y = int(round(img_coords[1, i, j], 0)) # y is v
            if 0 <= y < Nv and 0 <= x < Nh:
                Image_new[y, x] = Image[i, j]

    return Image_new



# CENTER Image Coords
def generateImageCoordinates_centers(ImageShape):
    """
    Generates a coordinate grid for the centers of pixels in an image based on its shape.

    Args:
        ImageShape (tuple): A tuple (Nv, Nh) representing the shape of the image, where:
                            - Nv is the number of vertical pixels (rows).
                            - Nh is the number of horizontal pixels (columns).

    Returns:
        numpy.ndarray: A 3D array of shape (2, Nv, Nh), where:
                       - The first dimension (index 0) contains the horizontal coordinates.
                       - The second dimension (index 1) contains the vertical coordinates.
                       - The origin is at the top-left corner of the image, and each pixel 
                         is represented by its center coordinates.

    Notes:
        - The function uses `np.mgrid` to generate a grid of vertical and horizontal 
          coordinates for pixel centers.
        - The output array is structured as [horizontal_centers, vertical_centers].
    """

    Nv, Nh = ImageShape # Shape of images
    vv_pix, hh_pix = np.mgrid[0:Nv,0:Nh] # 
    pixel_centers = np.array([hh_pix+0.5,vv_pix+0.5]) # stack grids to one layered array # center coordinates

    return pixel_centers
