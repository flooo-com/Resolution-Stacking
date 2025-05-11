# Feb 2025, Spatial Data Analysis, Uni Potsdam, Florian Josephowitz

# Import Packages
import numpy as np
import rawpy as rp
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
from tqdm import trange
from numba import njit, prange

import print_settings

# Clear the console output
os.system('cls' if os.name == 'nt' else 'clear')

# Set printopt for matrix output
np.set_printoptions(precision=6, suppress=True, linewidth=200)

## Settings <<<<<<<<<<<<<<<<<<<<<<<<<<
exp_shift = 5.0 # exposure shift in linear scale. Usable range from 0.25 (2-stop darken) to 8.0 (3-stop lighter)
bright = 1.3 # brightness scaling
image_number_selector = 10 # Number of images used for Upscaling
baseImageIndex = 0 # Wich image is the base image for all the image alingment # should be kept at 0 for now
nfeatures = 20000 # limit fir Sift features
sift_distance_threshold = 60 # Similarity betwenn 128 element SIFT Markers
ransac_reprojection_threshold = 0.7 # how critically RANSAC is for finding the Homography H based on back-projected keypoints in pixels
plot = False # if plots will be made for the beginning
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Find Images
fps_all = glob.glob("Data_Kuchen_2\*.dng")
fps_all = glob.glob("Data_Lab\Phone\*.dng")
fps_all = glob.glob("Data_House\Phone\*.dng")
fps_all = glob.glob("Data_House\Camera\*.dng")
fps_all = glob.glob("Data\*.dng")


names_all = [os.path.basename(fp) for fp in fps_all] # Extract the Filenames
for name in names_all : print(name)

# Select
fps = fps_all[:image_number_selector]
names = names_all[:image_number_selector]

# Print
print("images selected: " +str(len(fps)) +"/"+str(len(fps_all)))

## -----------------------------------
## Read Images 
def read_images(fps,exp_shift,bright):
    
    # number of images in paths
    N_img = len(fps) 

    # Set arrays
    rgb = [None] * N_img
    gray = [None] * N_img

    # Load images
    for i in trange(N_img):
        with rp.imread(fps[i]) as raw:
            rgb_temp = raw.postprocess(highlight_mode=rp.HighlightMode.Reconstruct(4), exp_shift=exp_shift, exp_preserve_highlights=1.0,bright=bright)

    # Convert grayscale and save
        gray[i] = cv.cvtColor(rgb_temp, cv.COLOR_RGB2GRAY)
        rgb[i] = rgb_temp

    return rgb, gray

# read images
print("...read images")
rgb, gray = read_images(fps,exp_shift,bright)

# General Information
Input_Image_Dims = gray[0].shape
N_Images = len(gray)

print(f"Input Image Dimensions: {Input_Image_Dims}")
print(f"Number of Images: {N_Images}")

if plot:
    plt.imshow(rgb[0])
    plt.show()

# -----------------------------------
## Sift Detection

def detect_SIFT(gray,nfeatures):

    # number of Iterations
    N_img = len(gray)
    
    # Initiate SIFT detector
    sift = cv.SIFT_create(nfeatures=nfeatures)

    # Set Arrays
    keypoints = [None] * N_img
    descriptors = [None] * N_img

    # Find SIFTs
    for i in trange(N_img):
        keypoints[i], descriptors[i] = sift.detectAndCompute(gray[i],None)

    return keypoints, descriptors

# Detect SIFT features
print("...detect SIFT Features")
kp, des = detect_SIFT(gray,nfeatures=nfeatures)


# -----------------------------------
##  Select Base Image, all other images are compared to this one

baseImageIdx = baseImageIndex
otherImageIdx = list(range(0,baseImageIdx)) + list(range(baseImageIdx+1,len(fps)))
otherImageIdx

# Descriptors
des_base = des[baseImageIdx]
des_other = des[:baseImageIdx]+ des[baseImageIdx+1:] # should work as expected

# Image File Names
name_base = names[baseImageIdx]
names_other = names[:baseImageIdx]+ names[baseImageIdx+1:]
print(f"Base Image is {name_base}, the connecting images are {names_other}")


# -----------------------------------
## Find Matches
# Notes: 
# https://stackoverflow.com/questions/22272283/opencv-feature-matching-for-multiple-images

# Notes For Speedup: Use FLANN for Approximate Matching

# Find Matches
def find_matches_old(descriptor_base,descriptor_other):

    # Number of Iterations
    N_comparisons = len(descriptor_other)

    # Matcher Object
    bf = cv.BFMatcher.create(crossCheck=True) 

    # Find matches
    matches = [None] * N_comparisons
    matches_count = [None] * N_comparisons
    for i in trange(N_comparisons):
        # matches
        matches_temp= bf.match(descriptor_base,descriptor_other[i])
        matches[i] = matches_temp
        matches_count[i] = len(matches_temp)
    
    return matches, matches_count

# def find_matches(descriptor_base,descriptor_other):

#     # Number of Iterations
#     N_comparisons = len(descriptor_other)

#     # Matcher Object
#     FLANN_INDEX_KDTREE = 0
#     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#     search_params = dict(checks = 50)

#     flann_matcher = cv.FlannBasedMatcher(index_params, search_params)
#     # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

#     # Find matches
#     matches = [None] * N_comparisons
#     matches_count = [None] * N_comparisons
#     for i in trange(N_comparisons):
#         # matches
#         matches_temp = flann_matcher.knnMatch(descriptor_base,descriptor_other[i],k=2)
#         matches[i] = matches_temp
#         matches_count[i] = len(matches_temp)
    
#     return matches, matches_count

print("...find matches")
matches, N_matches = find_matches_old(des_base,des_other)

print("Matches to base Image: "+names[0]+" (0)")
for i in range(len(matches)) : print(names[i]+" ("+str(i+1)+") Matches: "+str(N_matches[i]))

print("...get Keypoints")
# Filter matches
def filter_matches(matches,distance_threshold):
    
    N_matches = len(matches) # numer of matches with the base image

    # filter good matches
    good_matches = [None] * N_matches
    good_matches_count = [None] * N_matches
    for i in range(N_matches):
        good_matches[i] = [m for m in matches[i] if m.distance < distance_threshold]
        good_matches_count[i] = len(good_matches[i])
        # TODO Do RANSAC here instead?

    # TODO just sort the Matches?
    # TODO Alternative Sorting approach ?
    # Sort them in the order of their distance.
    # matches_sort = sorted(matches[i], key = lambda x:x.distance) # sort by distance
    # good_matches[i] = matches_sort[:N_treshold]

    return good_matches, good_matches_count

good_matches, good_matches_count = filter_matches(matches,distance_threshold=sift_distance_threshold)

def print_matches(matches):
    # not knn with k >1
    # prints first 10 matches
    print(f"Len of Matches {len(matches)}")
    for m in matches[0:10]:
        print(f"  QueryIdx: {m.queryIdx}, TrainIdx: {m.trainIdx}, ImgIdx: {m.imgIdx}, Distance: {m.distance}")

print_matches(good_matches[0])


# Plot of Matches and filtered matches in first image pair
if plot:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    plt.suptitle(f"Matches n = {len(matches[0])} Keypoints, Filtered matches n = {len(good_matches[0])}")
    # Plot all matches
    plot3 = cv.drawMatches(gray[0], kp[0], gray[1], kp[1], matches[0], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axes[0].set_title("All Matches Between Pictures \n(" +str(names[0]) +" and "+str(names[1])+")")
    axes[0].imshow(plot3)
    axes[0].axis('off')

    # Plot filtered matches
    plot4 = cv.drawMatches(gray[0], kp[0], gray[1], kp[1], good_matches[0], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axes[1].set_title("Filtered Matches Between Pictures \n(" +str(names[0]) +" and "+str(names[1])+")")
    axes[1].imshow(plot4)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# -----------------------------------
## Get Keypoints

# Index matches to image
def get_MatchIdx(matches):
    QueryIdx = [m.queryIdx for m in matches]
    TrainIdx = [m.trainIdx for m in matches]
    return QueryIdx, TrainIdx

# Get Keypoint Coordinates
def get_keypoints(good_matches,keypoints,baseImageIdx): 

    # Note 1: 
        # For every pair of images (base<>other) witht the number of pairs is len(fps) - 1, the keypoints for the Base image, and the 1 other images is calculated. 
        # The base keypoints are different for every image pair.

    # Note 2:
        # That woul be an alternative for getting the points
        # Index Points from Match index list:
        # kp1[QIdx[1]].pt # https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
    
    N_matches = len(good_matches)

    print(N_matches)
    keypointsXY_base = [None] * N_matches
    keypointsXY_other = [None] * N_matches
    for i in range(N_matches):
        # Call external function for image Coordinate extraction
        QIdx, TIdx = get_MatchIdx(good_matches[i])

        # Get kypoints in pixel coordinates of the right picture
        keypointsXY_base[i] = cv.KeyPoint_convert(keypoints[baseImageIdx],QIdx)
        keypointsXY_other[i] = cv.KeyPoint_convert(keypoints[otherImageIdx[i]],TIdx)

    return keypointsXY_base, keypointsXY_other

kpXY_base, kpXY_other = get_keypoints(good_matches,kp,baseImageIdx)


QIdx, TIdx = get_MatchIdx(matches[0])
kps_all_baseXY = cv.KeyPoint_convert(kp[baseImageIdx],QIdx)

# Plot all keypoints
if plot:
    # Plot
    fig, axes = plt.subplots( figsize=(12, 6),sharex=True,sharey=True)
    
    idx = 0 # <<< Change for other image

    plt.suptitle(f"Keypoints in base Images with n = {len(matches[0])}  Keypoints")

    # Plot match1 on the left
    c=np.random.rand(len(kps_all_baseXY)) # random values on the cmap
    
    axes.scatter(kps_all_baseXY[:, 0],kps_all_baseXY[:, 1], c=c, s=10,cmap="turbo")
    axes.imshow(gray[baseImageIdx],cmap="Greys_r")
    axes.set_title(f'Image {baseImageIdx}')
    axes.set_xlabel('Pixel X Coordinate')
    axes.set_ylabel('Pixel Y Coordinate')


    plt.tight_layout()
    plt.show()

# Keypoints for one image Pair
if plot:
    # Plot
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6),sharex=True,sharey=True)
    
    idx = 1 # <<< Change for other image

    plt.suptitle(f"Keypoints in Both Images with n = {len(good_matches[idx])} good matches")

    # Plot match1 on the left
    c=np.random.rand(len(good_matches[idx])) # random values on the cmap
    
    axes[0].scatter(kpXY_base[idx][:, 0],kpXY_base[idx][:, 1], c=c, s=10,cmap="turbo")
    axes[0].imshow(gray[baseImageIdx],cmap="Greys_r")
    axes[0].set_title(f'Image {baseImageIdx}')
    axes[0].set_xlabel('Pixel X Coordinate')
    axes[0].set_ylabel('Pixel Y Coordinate')

    # Plot match2 on the right
    axes[1].scatter(kpXY_other[idx][:, 0],kpXY_other[idx][:, 1], c=c, s=10, cmap="turbo")
    axes[1].imshow(gray[otherImageIdx[idx]],cmap="Greys_r")
    axes[1].set_title(f'Image {otherImageIdx[idx]}')
    axes[1].set_xlabel('Pixel X Coordinate')
    axes[1].set_ylabel('Pixel Y Coordinate')

    plt.tight_layout()
    plt.show()

# -----------------------------------
## Get Homographies

def findAllHomographies(keypointsXY_base,keypointsXY_other):

    # Number of iterations
    N_connections = len(keypointsXY_other)
    
    # Prepare Arrays
    H = [None] * N_connections
    mask = [None] * N_connections
    for i in range(N_connections):
        # Find projective matrix. All projections from other image into base image. All --> Base
        H[i], mask[i] = cv.findHomography(keypointsXY_other[i],keypointsXY_base[i], method=cv.RANSAC, ransacReprojThreshold=ransac_reprojection_threshold) # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gafd3ef89257e27d5235f4467cbb1b6a63
        
        # H_alt[i], mask_alt[i] = cv.findHomography(keypointsXY_base[i],keypointsXY_other[i], method=cv.RANSAC, ransacReprojThreshold=3.0)
    return H, mask


print("...find Homography")
H, mask = findAllHomographies(kpXY_base,kpXY_other)

for i, n in enumerate(mask): 
    print(f"Number of used Keypoints for H in RANSAC: {np.sum(n)}  (Total Keypoints input: {len(kpXY_other[i])})")



# -----------------------------------
## Validate Projection Accuracy

# ------
from functions import HomographyReprojection_error
# ------

for i in range(len(kpXY_base)):
    TrgKeypoints_XY_transformed, residual_xy, residual_x, residual_y = HomographyReprojection_error(kpXY_other[i],kpXY_base[i],H[i])
    print(f"Summed Error between image Keypoints: {np.sum(residual_xy)}")


# -----------------------------------
## Generate Image Coordinate Raster

# NORMAL Image Coords
def generateImageCoordinates(ImageShape):
    """
    Generates a coordinate grid for the borders of pixels in an image based on its shape.
    Args:
        ImageShape (tuple): A tuple representing the shape of the image 
                            (number of vertical pixels, number of horizontal pixels), 
                            typically obtained from `img.shape`.
    Returns:
        numpy.ndarray: A 3D array of shape (2, Nv+1, Nh+1), where:
                       - The first dimension (index 0) contains the horizontal coordinates.
                       - The second dimension (index 1) contains the vertical coordinates.
                       - The origin is at the top-left corner of the image, and each pixel 
                         is bounded by its top-left and bottom-right corners.
    Notes:
        - The function uses `np.mgrid` to generate a grid of vertical and horizontal 
          coordinates for pixel borders.
        - The output array is structured as [horizontal_borders, vertical_borders].
    """

    Nv, Nh = ImageShape # Shape of images
    vv_pix, hh_pix = np.mgrid[0:Nv,0:Nh] # 
    pixel_borders = np.array([hh_pix,vv_pix]) # stack grids to one layered array

    return pixel_borders

# CENTER Image Coords
def generateImageCoordinates_centers(ImageShape):
    """
    Generates a coordinate grid for the centers of pixels in an image based on its shape.
    Args:
        Image (numpy.ndarray): A 2D array representing the image.
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


def ImageCoordinates(OtherImage_list):
    
    # List of grayscale images is used to determine Shape and Count of images Coords rasters

    N_images = len(OtherImage_list)
    img_coords = [None] * N_images
    img_coords_centers = [None] * N_images
    for i in range(N_images):
        #img_shape = (OtherImage_list[0].shape[0]-1,OtherImage_list[0].shape[1]-1)
        img_shape = OtherImage_list[0].shape
        img_coords[i] = generateImageCoordinates(img_shape)
        img_coords_centers[i] = generateImageCoordinates_centers(img_shape)

    return img_coords,img_coords_centers

# Just Pick "other" images from the image list. That means "base" image is left out
gray_other = [gray[i] for i in otherImageIdx]
rgb_other = [rgb[i] for i in otherImageIdx] # nur vorläufig, später alle rgb for interpolation

img_coords, img_coords_centers = ImageCoordinates(gray_other) # Get all image Coordinates in one List


# -----------------------------------
## Transform all Image Coordinates to Base Coordinate System

# -----------------------------------
from functions import transform_imageCoords_numba
# -----------------------------------


def TransformPixelBorders(img_coords_list,H):
    
    N_transforms = len(img_coords_list)
    otherImagesXYTranfomed = [None] * N_transforms
    ImageDims = img_coords_list[0][0].shape

    # warmup
    _ = transform_imageCoords_numba(img_coords_list[0][0:10,0:10], H[0])

    # Execute all Transformations
    for i in range(N_transforms):
        otherImagesXYTranfomed_temp = transform_imageCoords_numba(img_coords_list[i], H[i])
        otherImagesXYTranfomed[i] = otherImagesXYTranfomed_temp.reshape(2, ImageDims[0], ImageDims[1])

    return otherImagesXYTranfomed

otherImagesCoordinatesTransformed = TransformPixelBorders(img_coords,H)
otherImagesCoordinates_centersTransformed = TransformPixelBorders(img_coords_centers,H)

# -----------------------------------
## Transform Image Pixels to the ne Coordinate system

## --------------------------------------
from functions import transform_imagePixels
## --------------------------------------

def TransformPixel(ImageCoordinates,Image):

    N_transformations = len(ImageCoordinates)
    ImageTransformed = [None] * N_transformations

    # Warmup Numba Function
    _ = transform_imagePixels(ImageCoordinates[0][:10,:10], Image[0][:10,:10])

    for i in range(N_transformations):
        ImageTransformed[i] = transform_imagePixels(ImageCoordinates[i], Image[i])

    return ImageTransformed

otherImagesTransformed = TransformPixel(otherImagesCoordinatesTransformed,gray_other)


# -----------------------------------


if plot:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8),sharex=True,sharey=True)
    # Left plot: gray[0]
    axes[0].set_title("Base Image (gray[0])")
    axes[0].imshow(gray[0], cmap="Greys")

    # Right plot: Overlay of all transformed images
    axes[1].set_title("Overlay of Transformed Images")
    axes[1].imshow(gray[0], cmap="Greys", alpha=0.3)  # Base image for reference
    for i, img in enumerate(otherImagesTransformed):
        axes[1].imshow(img, alpha=0.3, cmap="Greys")

    plt.tight_layout()
    plt.show()


## ------------------------------------------------------------------------------------


# def ge
gray_other = [gray[i] for i in otherImageIdx] 


# -----------------------------------
from functions import transform_imageCoords_numba
# -----------------------------------


def TransformPixelCenters(img_coords_center_list,H):
    
    N_transforms = len(img_coords_center_list)
    otherImagesXY_centers_Transformed = [None] * N_transforms
    ImageDims = img_coords_center_list[0][0].shape

    # warmup
    _ = transform_imageCoords_numba(img_coords_center_list[0][0:10,0:10], H[0])

    # Execute all Transformations
    for i in range(N_transforms):
        otherImagesXYTranfomed_temp = transform_imageCoords_numba(img_coords_center_list[i], H[i])
        otherImagesXY_centers_Transformed[i] = otherImagesXYTranfomed_temp.reshape(2, ImageDims[0], ImageDims[1])

    return otherImagesXY_centers_Transformed

otherImagesPixelCentersTransformed = TransformPixelCenters(img_coords_centers,H)

## ------------------------------------------------------------------
## First unsuccesfull attempt

## this has to be edited for the highres upsaling
# it should take:
    # 1. all image Borders Coords
    # 2. all image values
    # 3. Upscale Factor
    # >> then is should fill every high res image based on the underlaying pixel values
    # >> weighting according to homography error?

## 1. )
# First Attempt to fill accordint to pixels
@njit(parallel=True)
def FillUpscaledImage_numba2(Trg_Image_coordinatesXY2D_array, Images, ImageShapeHighres, upscale_factor,warmup=False):
    
    N_Images = len(Images)
    Nv, Nh = ImageShapeHighres
    sr = 8 # search range

    Image_new = np.empty((Nv, Nh), dtype=np.float32)
    Image_new.fill(np.nan)

    v_range = range(20,Nv-20)
    h_range = range(20,Nh-20)

    if warmup:
        v_range = range(2000,2050)
        h_range = range(2000,2050)

    for k in v_range:
        for j in h_range:
            pixel_list = np.empty(N_Images, dtype=np.float32)
            pixel_list.fill(np.nan)
            for i in prange(N_Images):
                k_small = k // upscale_factor
                j_small = j // upscale_factor

                x = Trg_Image_coordinatesXY2D_array[i, 0, k_small - sr:k_small + sr, j_small - sr:j_small + sr]
                y = Trg_Image_coordinatesXY2D_array[i, 1, k_small - sr:k_small + sr, j_small - sr:j_small + sr]

                x = x * upscale_factor
                y = y * upscale_factor

                pixel_values = Images[i][k_small - sr:k_small + sr, j_small - sr:j_small + sr]

                # For each pixel in search range check if in actual pixel k,j of highres image
                for m in range(x.shape[0]):
                    for n in range(x.shape[1]):
                        if (k < y[m, n] < (k + 1)) and (j < x[m, n] < (j + 1)):
                            pixel_list[i] = pixel_values[m, n]
                            break  # only take first match
            
            # fill actual highres pixel with mean of all detected pixels
            sum_val = 0.0
            count = 0
            for pixel_val in pixel_list:
                if not np.isnan(pixel_val):
                    sum_val += pixel_val
                    count += 1

            if count > 0:
                Image_new[k, j] = sum_val / count
            else:
                Image_new[k, j] = np.nan

    return Image_new


upscale_factor = 2 # on each side
ImageShapeHighres = (gray[0].shape[0]*upscale_factor ,gray[0].shape[1]*upscale_factor )


# FillUpscaledImage_numba2(np.asarray(otherImagesPixelCentersTransformed[0:1]), gray_other[0:1], ImageShapeHighres,upscale_factor=1,warmup=False)
# HighresGrey = FillUpscaledImage_numba2(np.asarray(otherImagesPixelCentersTransformed), gray_other, ImageShapeHighres,upscale_factor=2,warmup=False)

# Save HighresGrey to disk
# output_path = "DataGenerated/HighresGrey.npy"
# np.save(output_path, HighresGrey)
# print(f"HighresGrey saved to {output_path}")


# plt.figure()
# plt.imshow(HighresGrey,cmap="Greys")
# plt.show()


#Save HighresGrey to disk
# output_path = "DataGenerated/HighresGrey_Lab.npy"
# np.save(output_path, HighresGrey)
# print(f"HighresGrey saved to {output_path}")

# # Save the final high-resolution image as a PNG file
# output_path = "DataGenerated/HighresGrey_Lab.png"
# cv.imwrite(output_path, HighresGrey)
# print(f"High-resolution image saved as PNG to {output_path}")


# Interpolation Points
Nv, Nh = gray_other[0].shape # Shape of images
vv_pix, hh_pix = np.mgrid[0:Nv:0.5,0:Nh:0.5] # 
vv_pix += 0.5
hh_pix += 0.5 # centers

## ----------------------------------------------------------
## End First attempt

# Simple Upscale Function
@njit(parallel=True)
def upscale_image_gray(Image, upscale_factor):

    # Original Shape
    Nv, Nh = Image.shape[0:2]

    # Upscale shape
    Nv = Nv * upscale_factor
    Nh = Nh * upscale_factor

    if True:
        # Initialize
        Image_new = np.zeros((Nv, Nh),dtype=Image.dtype)
        for k in prange(Nv):
            for j in range(Nh):
                Image_new[k, j] = Image[k // upscale_factor, j // upscale_factor]
    return Image_new

@njit(parallel=True)
def upscale_image_rgb(Image, upscale_factor):


    # Original Shape
    Nv, Nh = Image.shape[0:2]

    # Upscale shape
    Nv = Nv * upscale_factor
    Nh = Nh * upscale_factor

    if True:
        # Initialize
        Image_new = np.zeros((Nv, Nh, 3),dtype=Image.dtype)
        for k in prange(Nv):
            for j in range(Nh):
                for c in range(3):  # Loop over channels
                    Image_new[k, j, c] = Image[k // upscale_factor, j // upscale_factor, c]
    if False:
        # Initialize
        Image_new = np.zeros((Nv, Nh),dtype=Image.dtype)
        for k in range(Nv):
            for j in range(Nh):
                Image_new[k, j] = Image[k // upscale_factor, j // upscale_factor]
    
    return Image_new

# Upscale for comprison
gray_0_upsc = upscale_image_gray(gray_other[0],2)
rgb_0_upsc = upscale_image_rgb(rgb_other[0],2)


## 2.)
## Do second attempt


print("...Generate Pixel Coordinates and Transform")

# Center Image Coords
def generateHighresImageCoordinates_centers(ImageShape,edge_cut):

    edge_cut = int(edge_cut) # in pixels

    Nv, Nh = ImageShape # Shape of images
    vv_pix, hh_pix = np.mgrid[0+edge_cut:Nv-edge_cut:0.5,0+edge_cut:Nh-edge_cut:0.5] # 
    pixel_centers = np.array([hh_pix+0.5,vv_pix+0.5]) # stack grids to one layered array # center coordinates

    return pixel_centers

edge_cut = 50 # pix
pixel_centers_highres = generateHighresImageCoordinates_centers(ImageShape=gray_other[0].shape,edge_cut=edge_cut)

pixel_centers_highres

# Tranform to every other Image
def TransformPixelCenters(HighRes_Center_xy,H):
    
    N_transforms = len(H)
    HighRes_Center_XY_transformed_to_otherImages = [None] * N_transforms
    ImageDims = HighRes_Center_xy[0].shape # take from x array

    # warmup
    _ = transform_imageCoords_numba(HighRes_Center_xy[0:10,0:10], H[0])

    # Execute all Transformations
    for i in range(N_transforms):
        CenterXY_temp = transform_imageCoords_numba(HighRes_Center_xy, np.linalg.inv(H[i]))
        HighRes_Center_XY_transformed_to_otherImages[i] = CenterXY_temp.reshape(2, ImageDims[0], ImageDims[1])

    return HighRes_Center_XY_transformed_to_otherImages


otherImagesPixelCentersTransformed = TransformPixelCenters(pixel_centers_highres,H)


# Fit in the Base image at the right postion
allImagesPixelCentersTransformed = otherImagesPixelCentersTransformed[:baseImageIdx] +[pixel_centers_highres] +  otherImagesPixelCentersTransformed[baseImageIdx:] # shoul functio0n as expected


print("...remap pixel values")

def RemapToBasePixels(ImagesPixelCentersTransformed,gray,rgb):

    N_remap = len(ImagesPixelCentersTransformed)
    GrayRemap_Highres = [None] * N_remap
    RGBRemap_Highres = [None] * N_remap
    for i in range(N_remap):

        x_float = ImagesPixelCentersTransformed[i][0].astype(np.float32)
        y_float = ImagesPixelCentersTransformed[i][1].astype(np.float32)
        #cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        GrayRemap_Highres[i] = cv.remap(gray[i], x_float, y_float, interpolation=cv.INTER_CUBIC)
        RGBRemap_Highres[i] = cv.remap(rgb[i], x_float, y_float, interpolation=cv.INTER_CUBIC)

    return GrayRemap_Highres, RGBRemap_Highres

GrayRemap_Highres, RGBRemap_Highres = RemapToBasePixels(allImagesPixelCentersTransformed,gray,rgb)

plt.title("Single Frame cubic interpolation")
plt.imshow(RGBRemap_Highres[0])
plt.show()

RGBRemap_Highres

# Stack Together
HighRes_gray = np.mean(np.stack(GrayRemap_Highres),axis=0)
HighRes_gray = HighRes_gray.astype(np.uint8)

HighRes_rgb = np.zeros((HighRes_gray.shape[0],HighRes_gray.shape[1],3))
HighRes_rgb[:,:,0] = np.mean(np.stack(RGBRemap_Highres)[:,:,:,0],axis=0)
HighRes_rgb[:,:,1] = np.mean(np.stack(RGBRemap_Highres)[:,:,:,1],axis=0)
HighRes_rgb[:,:,2] = np.mean(np.stack(RGBRemap_Highres)[:,:,:,2],axis=0)
HighRes_rgb = HighRes_rgb.astype(np.uint8)


# Plot
print("...plot results")
if True:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8),sharex=True,sharey=True)
    # Left plot: gray[0]
    axes[0].set_title("Base Image upscaled")
    axes[0].imshow(gray_0_upsc, cmap="Greys_r",extent=[0,ImageShapeHighres[1],ImageShapeHighres[0],0])

    # Right plot: Overlay of all transformed images
    axes[1].set_title("Stacked Highres Images")
    axes[1].imshow(HighRes_gray, cmap="Greys_r",extent=[edge_cut,ImageShapeHighres[1]-edge_cut,ImageShapeHighres[0]-edge_cut,edge_cut])  # Base image for reference

    plt.tight_layout()
    plt.show()

if True:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8),sharex=True,sharey=True)
    # Left plot: gray[0]
    axes[0].set_title("Base Image upscaled")
    axes[0].imshow(rgb_0_upsc, cmap="Greys",extent=[0,ImageShapeHighres[1],ImageShapeHighres[0],0])

    # Right plot: Overlay of all transformed images
    axes[1].set_title("Stacked Highres Images")
    axes[1].imshow(HighRes_rgb,extent=[edge_cut,ImageShapeHighres[1]-edge_cut,ImageShapeHighres[0]-edge_cut,edge_cut])  # Base image for reference

    plt.tight_layout()
    plt.show()


## Safe Results with appr filenames

## Highres -----------------------------------------
# Save HighRes_gray as a PNG image
fp_out_gray = f"DataGenerated/Image_HighRes_gray_Base_{name_base[:-4]}_imgCount_{image_number_selector}_nfeatures_{nfeatures}_siftThr_{sift_distance_threshold}.png"
cv.imwrite(fp_out_gray, HighRes_gray)
print(f"...HighRes_gray saved to {fp_out_gray}")

# Save HighRes_rgb as a PNG image
fp_out_rgb = f"DataGenerated/Image_HighRes_rgb_Base_{name_base[:-4]}_imgCount_{image_number_selector}_nfeatures_{nfeatures}_siftThr_{sift_distance_threshold}.png"
cv.imwrite(fp_out_rgb, cv.cvtColor(HighRes_rgb, cv.COLOR_RGB2BGR))
print(f"...HighRes_rgb saved to {fp_out_rgb}")

## BiCubic interpolated -----------------------------
# Save bicubic itnerpolated
fp_out_gray = f"DataGenerated/Image_Bicubic_gray_Base_{name_base[:-4]}_imgCount_{image_number_selector}_nfeatures_{nfeatures}_siftThr_{sift_distance_threshold}.png"
cv.imwrite(fp_out_gray, GrayRemap_Highres[baseImageIdx])
print(f"...Bicubic_gray saved to {fp_out_gray}")

# Save HighRes_rgb as a PNG image
fp_out_rgb = f"DataGenerated/Image_Bicubic_rgb_Base_{name_base[:-4]}_imgCount_{image_number_selector}_nfeatures_{nfeatures}_siftThr_{sift_distance_threshold}.png"
cv.imwrite(fp_out_rgb, cv.cvtColor(RGBRemap_Highres[baseImageIdx], cv.COLOR_RGB2BGR))
print(f"...Bicubic_rgb saved to {fp_out_rgb}")

## Save Original image with simple pixel filling ----
# Save Simple itnerpolated
fp_out_gray = f"DataGenerated/Image_SimpleUpscale_gray_Base_{name_base[:-4]}_imgCount_{image_number_selector}_nfeatures_{nfeatures}_siftThr_{sift_distance_threshold}.png"
cv.imwrite(fp_out_gray, gray_0_upsc)
print(f"...SimpleUpscale_gray saved to {fp_out_gray}")

# Save simple upscaled as a PNG image
fp_out_rgb = f"DataGenerated/Image_SimpleUpscale_rgb_Base_{name_base[:-4]}_imgCount_{image_number_selector}_nfeatures_{nfeatures}_siftThr_{sift_distance_threshold}.png"
cv.imwrite(fp_out_rgb, cv.cvtColor(rgb_0_upsc, cv.COLOR_RGB2BGR))
print(f"...SimpleUpscale_rgb saved to {fp_out_rgb}")

## TODO
# include base image with base coords
# Querschnitt durch das bild 
# evtl fine alignement
# extend bei plot

# safe out 1) upscaled base image 2) interpolated base image, 3) highres product


# Generate a report of all print statements
def report():
    print("\n\n--- Report of All Print Statements ---")

    # Include the state of the "Settings"
    print("\n--- Settings ---")
    print(f"Exposure Shift: {exp_shift}")
    print(f"Brightness Scaling: {bright}")
    print(f"Image Number Selector: {image_number_selector}")
    print(f"Number of SIFT Features: {nfeatures}")
    print(f"SIFT Distance Threshold: {sift_distance_threshold}")
    print(f"Plot Enabled: {plot}\n")
    # Re-run all print statements
    print("Data:")
    print("images selected: " + str(len(fps)) + "/" + str(len(fps_all)))
    print(f"Input Image Dimensions: {Input_Image_Dims}")
    print(f"Number of Images: {N_Images}")
    print(f"Base Image is {name_base}, the connecting images are {names_other}")
    print(f"Matches:")
    print("Matches to base Image: " + names[0] + " (0)")
    for i in range(len(matches)):
        print(names[i] + " (" + str(i + 1) + ") Matches: " + str(N_matches[i]) + " (Good Matches: "+str(len(good_matches[i]))+")")
    print(f"Homographic Projection:")
    for i, n in enumerate(mask):
        print(f"Number of used Keypoints for H in RANSAC: {np.sum(n)}  (Total Keypoints input: {len(kpXY_other[i])})")
    print("Alignment Error:")
    for i in range(len(kpXY_base)):
        TrgKeypoints_XY_transformed, residual_xy, residual_x, residual_y = HomographyReprojection_error(kpXY_other[i], kpXY_base[i], H[i])
        print(f"Summed Error between image Keypoints: {np.sum(residual_xy)}")
    print("Safe to disk:")
    print(f"HighRes_gray saved to {fp_out_gray}")
    print(f"HighRes_rgb saved to {fp_out_rgb}")
    print(f"Bicubic_gray saved to {fp_out_gray}")
    print(f"Bicubic_rgb saved to {fp_out_rgb}")
    print(f"SimpleUpscale_gray saved to {fp_out_gray}")
    print(f"SimpleUpscale_rgb saved to {fp_out_rgb}")

    print("\n--- End of Report ---")

report()