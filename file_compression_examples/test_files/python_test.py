"""
This is my implementation of the Canny edge detection algorithm. Note that the
first 3 steps bear striking similarity to the Sobel edge detection method with
the exception of the calculation of theta, which we use here later on.
"""

from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import click
from helper_blur import gaussian_blur, median_blur
from helper_greyscale import greyscale
from helper_rainbow_fill import get_color


def canny_edge_detect(img_arr: np.ndarray, color: bool) -> np.ndarray:
    """
    Performs Canny edge detection on an image array.

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        color (bool): option to color edges based on edge gradient direction

    Returns:
        np.ndarray: Array representation of the edge detector applied image
    """
    # image array dimensions
    HEIGHT, WIDTH, _ = img_arr.shape
    
    
    ### GREYSCALING IMAGE ###
    
    greyscaled_img_arr = greyscale(img_arr)
    print("\nGREYSCALE APPLIED\n")
    
    
    ### PERFORMING NOISE REDUCTION WITH MEDIAN AND GAUSSIAN BLUR ###
    
    # median blur, radius depends on image size to clear 
    r = 0 if WIDTH < 500 else 1
    median_blurred_img_arr = median_blur(greyscaled_img_arr, radius=r)
    print("\nMEDIAN BLUR APPLIED\n")
    
    # Gaussian blur, sigma = 2
    noise_reduced_img_arr = gaussian_blur(median_blurred_img_arr, sigma=2)
    print("\nGAUSSIAN BLUR APPLIED")
    
    
    ### GRADIENT CALCULATION ###
    
    # turning image array into 2-d array (shows only intensity)
    intensity_arr = noise_reduced_img_arr[:, :, 0]
    
    # X and Y Sobel filters (discrete derivative approximations in dx and dy)
    X_KERNEL = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    Y_KERNEL = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    
    # creating array to be convolved with estimated derivative gradients
    convolved_arr = np.zeros((HEIGHT, WIDTH))
    # creating array to store theta (direction) values for intensity change
    theta_vals = np.zeros((HEIGHT, WIDTH))
    
    # iterating through intensity_arr to fill convolved array
    for y, row in enumerate(intensity_arr):
        # ignore edge
        if y == 0 or y == HEIGHT - 1:
            continue
        
        for x, _ in enumerate(row):
            # ignore edge
            if x == 0 or x == WIDTH - 1:
                continue
            
            # get Hadamard product and array sum to calculate intensity change
            g_x = np.sum(X_KERNEL * intensity_arr[y-1:y+2, x-1:x+2])
            g_y = np.sum(Y_KERNEL * intensity_arr[y-1:y+2, x-1:x+2])
            
            # assignment of intensity change value (from gradient)
            convolved_arr[y, x] = np.sqrt(g_x**2 + g_y**2)
            # specialized arctan function to store direction of gradient change
            theta_vals[y, x] = np.arctan2(g_y, g_x)

    # removing edges that weren't calculated
    convolved_arr = np.squeeze(convolved_arr)
    theta_vals = np.squeeze(theta_vals)
    
    # redirecting theta values to lie only between 0 and 180 (mainting direction)
    theta_vals = theta_vals.astype(np.float64)
    theta_vals[theta_vals < 0] += np.pi
          
    print("\nGRADIENT AND IMAGE ARRAY CONVOLVED\n\nTHETA VALUES CALCULATED")
    
    
    ### APPLYING NON-MAXIMUM SUPRESSION ###
    
    # creating new array to store values from non-maximum suppression
    suppressed_arr = convolved_arr.copy()
    
    # iterating through convolved_arr to find perpendicular points and determine
    # if point is valid edge according to theta and adjacent intensities
    for y, row in enumerate(convolved_arr):
        # ignore edge
        if y == 0 or y == HEIGHT - 1:
            continue
            
        for x, _ in enumerate(row):
            # ignore edge
            if x == 0 or x == WIDTH - 1:
                continue
                
            # finding which adjacent angles to check using adjusted theta
            theta = theta_vals[y, x]
            
            if 0 <= theta < np.pi / 8 or \
            7 * np.pi / 8 <= theta <= np.pi:
                adj_1 = convolved_arr[y, x - 1]
                adj_2 = convolved_arr[y, x + 1]
            
            elif np.pi / 8 <= theta <= 3 * np.pi / 8:
                adj_1 = convolved_arr[y - 1, x + 1]
                adj_2 = convolved_arr[y + 1, x - 1]
            
            elif 3 * np.pi / 8 <= theta <= 5 * np.pi / 8:
                adj_1 = convolved_arr[y - 1, x]
                adj_2 = convolved_arr[y + 1, x]
            
            elif 5 * np.pi / 8 <= theta <= 7 * np.pi / 8:
                adj_1 = convolved_arr[y + 1, x + 1]
                adj_2 = convolved_arr[y - 1, x - 1]
                
            # assigning new values to suppressed array if center intensity is
            # larger than neighbors (otherwise 0)
            if convolved_arr[y, x] >= adj_1 and convolved_arr[y, x] >= adj_2:
                suppressed_arr[y, x] = convolved_arr[y, x]
            else:
                suppressed_arr[y, x] = 0
            
    print("\nNON-MAXIMUM SUPPRESSION APPLIED")
    
    
    ### APPLYING DOUBLE THRESHOLD AND HYSTERESIS ###
    
    double_threshold_arr = np.zeros_like(suppressed_arr)
    
    high_th = suppressed_arr.max() * 0.1
    low_th = high_th * 0.5
    strong = 255
    weak = 50
    
    for y, row in enumerate(suppressed_arr):
        for x, intensity in enumerate(row):
            
            if intensity > high_th:
                # strong edge
                double_threshold_arr[y, x] = strong
            elif intensity < low_th:
                # false edge
                double_threshold_arr[y, x] = 0
            else:
                # weak edge
                double_threshold_arr[y, x] = weak
                
    print("\nDOUBLE THRESHOLD APPLIED")
    
    # applying hysteresis
    
    # creating final image array
    final_arr = np.zeros_like(double_threshold_arr)
    # directions to check for strong edge
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1), 
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for y, row in enumerate(double_threshold_arr):
        # ignore edge
        if y == 0 or y == HEIGHT - 1:
            continue
            
        for x, val in enumerate(row):
            # ignore edge
            if x == 0 or x == WIDTH - 1:
                continue
            
            if val == strong or val == 0:
                final_arr[y, x] = val
            
            # determining whether to include weak edge based on neighbors being
            # strong edges    
            else:
                include = False
                for dx, dy in directions:
                    if double_threshold_arr[y+dy, x+dx] == strong:
                        include = True
                        break
                final_arr[y, x] = strong if include else 0
    print("\nHYSTERESIS APPLIED")
                    
    
    ### RETURNING FINAL IMAGE ARRAY ###
    
    final_image_arr = np.zeros((HEIGHT - 1, WIDTH - 1, 3), dtype=np.uint8)
    # putting intensity values from final array back into RGB format
    for y, row in enumerate(final_image_arr):    
        for x, _ in enumerate(row):
            if not color:
                val = final_arr[y, x]
                pixel_val = np.array([val, val, val])
            else:
                # optionally assigning edge color based on gradient slope
                if final_arr[y, x] != 0:
                    pixel_val = get_color(theta_vals[y, x])
                else:
                    pixel_val = np.array([0, 0, 0])
            final_image_arr[y, x] = pixel_val
    print("\nTHETA GRADIENT COLORING COMPLETE")
    print("\nIMAGE COMPLETE")
    return final_image_arr


# click commands
@click.command(name="canny_edge_detector")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option("--color/--no-color", default=True)

def edge_detect(filename: str, color: bool) -> None:
    """
    command
    """
    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = canny_edge_detect(img_arr, color)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    edge_detect()
    
"""
This is a simple program that takes a gif and returns its canny counterpart. I'm
just making this to have a pretty header for my README, lol.
"""

from typing import List, Tuple, Optional
import numpy as np
import time
from PIL import Image
import click
from canny_edge_detector import canny_edge_detect


def canny_gif(img_arr_seq: List[np.ndarray], color: bool) -> List[np.ndarray]:
    """
    Performs Canny edge detection on an image array.

    Args:
        img_arr_seq (List[np.ndarray]): sequence of 3-d array representations of 
            images which constitute the gif.
        color (bool): option to color edges based on edge gradient direction

    Returns:
        List[np.ndarray]: Sequence of array representations of the edge 
            detector applied images which constitute the gif
    """
    canny_seq = []
    for i, img_arr in enumerate(img_arr_seq):
        print(f"\nPROCESSING IMAGE {i+1}/{len(img_arr_seq)}")
        canny_seq.append(canny_edge_detect(img_arr, color))
        print(f"\nAPPLIED CANNY EDGE DETECTION TO IMAGE {i+1}/{len(img_arr_seq)}\n")
    
    return canny_seq

# click commands
@click.command(name="canny_gif_maker")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option("--color/--no-color", default=True)

def canny_animate(filename: str, color: bool) -> None:
    """
    command
    """
    gif_name = filename.split('/')[-1][:-4]
    if color:
        gif_name += "_colored"
    
    # changing gif into list of image arrays
    img_arr_seq = []
    with Image.open(filename) as gif:
        for n in range(gif.n_frames):
            gif.seek(n)
            img_arr = np.array(gif.convert("RGB"))
            img_arr_seq.append(img_arr)
    
    # applying Canny edge detection to all images in list
    canny_gif_arr = canny_gif(img_arr_seq, color)
    # changing each image array in list to image and storing in new list
    img_seq = []
    for img_arr in canny_gif_arr:
        img_seq.append(Image.fromarray(img_arr))
    
    # saving new gif (from list of images) to canny_animation folder
    img_seq[0].save(f"canny_animations/canny_{gif_name}.gif", 
                    save_all=True, append_images=img_seq[1:], loop=0)

if __name__ == "__main__":
    canny_animate()


"""
This is a from-scratch implementation of the Gaussian Blur function, which
utilizes the array-form of a given image and the standard Gaussian distribution 
in 2 dimensions to evenly blur an image with a kernel matrix. This is a 
computationally expensive operation which will take quite some time to run on 
larger images or with large sigma values.

Note that this file uses click for command line support. You can run it from 
the root with: 'python3 src/filename.py --help' to get started.
"""

from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import click


def gaussian_blur(img_arr: np.ndarray, sigma: int, msg: bool) -> np.ndarray:
    """
    Operates on array representation of image to return a guassian blurred array

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        sigma (int): Standard deviation in guassian distribution. Serves as the
            strength of the blur for our purposes.
        msg (bool): Option to display a progress message every time a row is 
            completed

    Returns:
        np.ndarray: Array representation of the blurred image
    """
    # new image array creation
    new_img_arr = img_arr.copy()
    # creates kernel with weights according to gaussian distribution
    kernel = get_kernel(sigma)
    
    for y, row in enumerate(img_arr):
        if msg:
            print(f"{y}/{img_arr.shape[0]} pixel rows calculated")
        for x, _ in enumerate(row):
            
            # getting image_piece dimensions
            height, width, _ = img_arr.shape
            img_range = find_range((height, width), (x, y), 3 * sigma)
            x_min, x_max = img_range[0]
            y_min, y_max = img_range[1]
            img_piece = img_arr[y_min:y_max, x_min:x_max]
            
            # cropping kernel to match image_piece
            x_bounds, y_bounds = crop_kernel(kernel, img_range)
            x_min, x_max = x_bounds
            y_min, y_max = y_bounds
            kernel_piece = kernel[y_min:y_max, x_min:x_max]
            
            # normalizing kernel values so that total sum is 3.00 
            # (1.00 for each RGB)
            kernel_piece = kernel_piece / np.sum(kernel_piece) * 3
            
            # putting blurred pixel into new image array
            new_img_arr[y, x] = pixel_calculate(kernel_piece, img_piece)
    
    if msg: 
        print("done!")
    return new_img_arr


def find_range(dimensions: Tuple[int, int], center: Tuple[int, int],
                      radius: int) -> List[Tuple[int, int]]:
    """
    Finds coordinate range for finding the kernel with a given image size and
        pixel coordinate of interest. 

    Args:
        dimensions (Tuple[int, int]): Image dimensions (height x width)
        center (Tuple[int, int]): Coordinate of pixel of interest
        radius (int): Distance in each cardinal direction that the kernel would
            extend into. Tells us how far cropping should go.

    Returns:
        List[Tuple[int, int]]: [x-range, y-range]
    """
    y_max, x_max = dimensions
    x, y = center
    
    x_min = 0 if x - radius < 0 else x - radius
    x_max = x_max - 1 if x + radius > x_max - 1 else x + radius
    y_min = 0 if y - radius < 0 else y - radius
    y_max = y_max - 1 if y + radius > y_max - 1 else y + radius   
     
    return [(x_min, x_max + 1), (y_min, y_max + 1)]


def get_kernel(sigma: int) -> np.ndarray:
    """
    Pre-calculates the kernel (which will be used for convolution) using the
        2 dimensional guassian distribution. Creates a matrix that extends for
        3 * sigma in each direction (gaussian weights past that are negligible).

    Args:
        sigma (int): Standard deviation in guassian distribution. Serves as the
            strength of the blur for our purposes.

    Returns:
        np.ndarray: Generic kernel with guassin distribution in 2-d.
    """
    # creating kernel using 2-d guassian distribution
    kernel = np.zeros((6 * sigma + 1, 6 * sigma + 1, 3))
    for y in range(-3 * sigma, 3 * sigma + 1):
        for x in range(-3 * sigma, 3 * sigma + 1):
            coef = 1 / (2 * np.pi * sigma**2)
            exp_term = np.exp(- (x**2 + y**2) / (2 * sigma**2))
            weight = coef * exp_term
            kernel[y + 3 * sigma, x + 3 * sigma] = np.array([weight] * 3)
    return kernel


def crop_kernel(kernel: np.ndarray, img_range: List[Tuple[int, int]]
                ) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Normalizes and crops kernel based on respective pixel position so that its
        dimensions match with the image piece for hadamard product.

    Args:
        kernel (np.ndarray): Generic uncropped kernel.
        img_range (List[Tuple[int, int]]): Coordinate ranges to apply crop.

    Returns:
        np.ndarray: Coordinate ranges within the kernel to apply crop in 
            order to get kernel_piece
    """
    x_min, x_max = img_range[0]
    y_min, y_max = img_range[1]
    new_x_min = new_x_max = new_y_min = new_y_max = None

    # vertical cropping
    if y_max - y_min < kernel.shape[0]:
        if y_min == 0:
            new_y_min, new_y_max = kernel.shape[0] - (y_max - y_min), None
        else:
            new_y_min, new_y_max = None, y_max - y_min

    # horizontal cropping
    if x_max - x_min < kernel.shape[1]:
        if x_min == 0:
            new_x_min, new_x_max = kernel.shape[1] - (x_max - x_min), None
        else:
            new_x_min, new_x_max = None, x_max - x_min

    return [(new_x_min, new_x_max), (new_y_min, new_y_max)]    


def pixel_calculate(kernel: np.ndarray, og_img: np.ndarray) -> np.ndarray:
    """
    Calculates RGB value for given pixel based on identically dimensioned
        kernel and original image. Utilizes Hadamard product before we take the
        pseudo-sum to obtain 1x3 array for RGB.

    Args:
        kernel (np.ndarray): Kernel to apply convolution from.
        og_img (np.ndarray): Original image array

    Returns:
        np.ndarray: RGB value of pixel
    """
    had_product = kernel * og_img
    pixel = np.zeros((1, 3))
    for row in had_product:
        for rgb_piece in row:
            pixel += rgb_piece
    return pixel


# click commands
@click.command(name="gaussian_blur")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option('-s', '--sigma-value', type=int, default=2)
@click.option("--progress/--hide-progress", default=True)

def blur(filename: str, sigma_value: int, progress: bool) -> None:
    """
    command line operation
    """
    if type(sigma_value) is not int and not 1 <= sigma_value <= 10:
        raise ValueError("sigma value must be int from 1 to 10")
    
    with Image.open(filename) as img:
        img_arr = np.array(img)
        if max(img_arr.shape) > 500:
            raise ValueError("Image is too large for gaussian blur to be "
                             "efficient. Try another image file or another "
                             "algorithm like box blur")
        new_img_arr = gaussian_blur(img_arr, sigma_value, progress)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    blur()