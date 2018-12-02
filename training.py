from imutils import contours
import imutils
import cv2
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
import scipy.ndimage.filters as flt

import argparse


def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):
	"""
	Anisotropic diffusion.

	Usage:
	imgout = anisodiff(im, niter, kappa, gamma, option)

	Arguments:
	        img    - input image
	        niter  - number of iterations
	        kappa  - conduction coefficient 20-100 ?
	        gamma  - max value of .25 for stability
	        step   - tuple, the distance between adjacent pixels in (y,x)
	        option - 1 Perona Malik diffusion equation No 1
	                 2 Perona Malik diffusion equation No 2
	        ploton - if True, the image will be plotted on every iteration

	Returns:
	        imgout   - diffused image.

	kappa controls conduction as a function of gradient.  If kappa is low
	small intensity gradients are able to block conduction and hence diffusion
	across step edges.  A large value reduces the influence of intensity
	gradients on conduction.

	gamma controls speed of diffusion (you usually want it at a maximum of
	0.25)

	step is used to scale the gradients in case the spacing between adjacent
	pixels differs in the x and y axes

	Diffusion equation 1 favours high contrast edges over low contrast ones.
	Diffusion equation 2 favours wide regions over smaller ones.

	Reference: 
	P. Perona and J. Malik. 
	Scale-space and edge detection using ansotropic diffusion.
	IEEE Transactions on Pattern Analysis and Machine Intelligence, 
	12(7):629-639, July 1990.

	Original MATLAB code by Peter Kovesi  
	School of Computer Science & Software Engineering
	The University of Western Australia
	pk @ csse uwa edu au
	<http://www.csse.uwa.edu.au>

	Translated to Python and optimised by Alistair Muldal
	Department of Pharmacology
	University of Oxford
	<alistair.muldal@pharm.ox.ac.uk>

	June 2000  original version.       
	March 2002 corrected diffusion eqn No 2.
	July 2012 translated to Python
	"""

	# ...you could always diffuse each color channel independently if you
	# really want
	if img.ndim == 3:
		warnings.warn("Only grayscale images allowed, converting to 2D matrix")
		img = img.mean(2)

	# initialize output array
	img = img.astype('float32')
	imgout = img.copy()

	# initialize some internal variables
	deltaS = np.zeros_like(imgout)
	deltaE = deltaS.copy()
	NS = deltaS.copy()
	EW = deltaS.copy()
	gS = np.ones_like(imgout)
	gE = gS.copy()

	# create the plot figure, if requested
	if ploton:
		import pylab as pl
		from time import sleep

		fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
		ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

		ax1.imshow(img,interpolation='nearest')
		ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
		ax1.set_title("Original image")
		ax2.set_title("Iteration 0")

		fig.canvas.draw()

	for ii in np.arange(1,niter):

		# calculate the diffs
		deltaS[:-1,: ] = np.diff(imgout,axis=0)
		deltaE[: ,:-1] = np.diff(imgout,axis=1)

		if 0<sigma:
			deltaSf=flt.gaussian_filter(deltaS,sigma);
			deltaEf=flt.gaussian_filter(deltaE,sigma);
		else: 
			deltaSf=deltaS;
			deltaEf=deltaE;
			
		# conduction gradients (only need to compute one per dim!)
		if option == 1:
			gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
			gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
		elif option == 2:
			gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
			gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]

		# update matrices
		E = gE*deltaE
		S = gS*deltaS

		# subtract a copy that has been shifted 'North/West' by one
		# pixel. don't as questions. just do it. trust me.
		NS[:] = S
		EW[:] = E
		NS[1:,:] -= S[:-1,:]
		EW[:,1:] -= E[:,:-1]

		# update the image
		imgout += gamma*(NS+EW)

		if ploton:
			iterstring = "Iteration %i" %(ii+1)
			ih.set_data(imgout)
			ax2.set_title(iterstring)
			fig.canvas.draw()
			# sleep(0.01)

	return imgout

def anisodiff3(stack,niter=1,kappa=50,gamma=0.1,step=(1.,1.,1.),option=1,ploton=False):
	"""
	3D Anisotropic diffusion.

	Usage:
	stackout = anisodiff(stack, niter, kappa, gamma, option)

	Arguments:
	        stack  - input stack
	        niter  - number of iterations
	        kappa  - conduction coefficient 20-100 ?
	        gamma  - max value of .25 for stability
	        step   - tuple, the distance between adjacent pixels in (z,y,x)
	        option - 1 Perona Malik diffusion equation No 1
	                 2 Perona Malik diffusion equation No 2
	        ploton - if True, the middle z-plane will be plotted on every 
	        	 iteration

	Returns:
	        stackout   - diffused stack.

	kappa controls conduction as a function of gradient.  If kappa is low
	small intensity gradients are able to block conduction and hence diffusion
	across step edges.  A large value reduces the influence of intensity
	gradients on conduction.

	gamma controls speed of diffusion (you usually want it at a maximum of
	0.25)

	step is used to scale the gradients in case the spacing between adjacent
	pixels differs in the x,y and/or z axes

	Diffusion equation 1 favours high contrast edges over low contrast ones.
	Diffusion equation 2 favours wide regions over smaller ones.

	Reference: 
	P. Perona and J. Malik. 
	Scale-space and edge detection using ansotropic diffusion.
	IEEE Transactions on Pattern Analysis and Machine Intelligence, 
	12(7):629-639, July 1990.

	Original MATLAB code by Peter Kovesi  
	School of Computer Science & Software Engineering
	The University of Western Australia
	pk @ csse uwa edu au
	<http://www.csse.uwa.edu.au>

	Translated to Python and optimised by Alistair Muldal
	Department of Pharmacology
	University of Oxford
	<alistair.muldal@pharm.ox.ac.uk>

	June 2000  original version.       
	March 2002 corrected diffusion eqn No 2.
	July 2012 translated to Python
	"""

	# ...you could always diffuse each color channel independently if you
	# really want
	if stack.ndim == 4:
		warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
		stack = stack.mean(3)

	# initialize output array
	stack = stack.astype('float32')
	stackout = stack.copy()

	# initialize some internal variables
	deltaS = np.zeros_like(stackout)
	deltaE = deltaS.copy()
	deltaD = deltaS.copy()
	NS = deltaS.copy()
	EW = deltaS.copy()
	UD = deltaS.copy()
	gS = np.ones_like(stackout)
	gE = gS.copy()
	gD = gS.copy()

	# create the plot figure, if requested
	if ploton:
		import pylab as pl
		from time import sleep

		showplane = stack.shape[0]//2

		fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
		ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

		ax1.imshow(stack[showplane,...].squeeze(),interpolation='nearest')
		ih = ax2.imshow(stackout[showplane,...].squeeze(),interpolation='nearest',animated=True)
		ax1.set_title("Original stack (Z = %i)" %showplane)
		ax2.set_title("Iteration 0")

		fig.canvas.draw()

	for ii in np.arange(1,niter):

		# calculate the diffs
		deltaD[:-1,: ,:  ] = np.diff(stackout,axis=0)
		deltaS[:  ,:-1,: ] = np.diff(stackout,axis=1)
		deltaE[:  ,: ,:-1] = np.diff(stackout,axis=2)

		# conduction gradients (only need to compute one per dim!)
		if option == 1:
			gD = np.exp(-(deltaD/kappa)**2.)/step[0]
			gS = np.exp(-(deltaS/kappa)**2.)/step[1]
			gE = np.exp(-(deltaE/kappa)**2.)/step[2]
		elif option == 2:
			gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
			gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
			gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

		# update matrices
		D = gD*deltaD
		E = gE*deltaE
		S = gS*deltaS

		# subtract a copy that has been shifted 'Up/North/West' by one
		# pixel. don't as questions. just do it. trust me.
		UD[:] = D
		NS[:] = S
		EW[:] = E
		UD[1:,: ,: ] -= D[:-1,:  ,:  ]
		NS[: ,1:,: ] -= S[:  ,:-1,:  ]
		EW[: ,: ,1:] -= E[:  ,:  ,:-1]

		# update the image
		stackout += gamma*(UD+NS+EW)

		if ploton:
			iterstring = "Iteration %i" %(ii+1)
			ih.set_data(stackout[showplane,...].squeeze())
			ax2.set_title(iterstring)
			fig.canvas.draw()
			# sleep(0.01)

	return stackout

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# from skimage.transform import hough_ellipse

# Load dictionary
clf = joblib.load("digits_cls.pkl")

# load the example image
orig_image_name = args["image"]

new_image_name = orig_image_name+"-mod.png"
image = cv2.imread(new_image_name)

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=600, width=800)
image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)


cv2.imshow("Resized", image)

cv2.waitKey()

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# blur = cv2.blur(image,(9,9))
# blur2 = cv2.GaussianBlur(image,(3,3),0)

# absd=cv2.equalizeHist(cv2.cvtColor(cv2.absdiff(blur2,blur),cv2.COLOR_BGR2GRAY))
# cv2.imshow("blur", blur)
# cv2.waitKey()
# cv2.imshow("blur2", blur2)
# cv2.waitKey()
# cv2.imshow("blur2", blur2)
# cv2.waitKey()    

# gray = cv2.GaussianBlur(gray, (1,1), 0)
# cv2.imshow( 'GaussianBlur',gray)
# cv2.waitKey()

# gray = cv2.bilateralFilter(gray,105,5,5)


# gray = cv2.fastNlMeansDenoising(gray,None,4,7,91)

# cv2.imshow( 'bilateralFilter',gray)
# cv2.waitKey()

# for x in range(0,100):
# gray=anisodiff(gray,2,80,0.075,(1,1),2.5,1)
# cv2.imshow("anisodiff", gray)
# cv2.waitKey()


# for x in range(70,90):
#     print x
thresh = cv2.threshold(gray, 76, 140, cv2.THRESH_BINARY)[1]
cv2.imshow("thresh", thresh)
cv2.waitKey()
    

# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# thresh = cv2.filter2D(thresh, -1, kernel)
# cv2.imshow("filter2D", thresh)
# cv2.waitKey()



# for x in range(0,10):
#     print x
edged = cv2.Canny(thresh,  100, 255, apertureSize = 5)
cv2.imshow("Canny", edged)
cv2.waitKey()


# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
# # dilated = cv2.dilate(edged.copy(),kernel,iterations = 2)
# # cv2.imshow("dilated", dilated)
# # cv2.waitKey()



# find contours in the edge map, then sort them by their
# size in descending order
# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# displayCnt = None

contours_dict = dict()
blank_background = np.zeros_like(edged)
dd = cv2.drawContours(blank_background, cnts, -1, (255,255,255), thickness=2)
cv2.imshow( 'dd',dd)
cv2.waitKey()

# loop over the contours
prev_x = 0
prev_y = 0
for cont in cnts:
    x, y, w, h = cv2.boundingRect(cont)

    area = cv2.contourArea(cont)
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.04 * peri, True)
    if x == prev_x and y == prev_y and 30 < area and 15 < w and h > 35 and len(approx) > 3:
        contours_dict[(x, y, w, h)] = cont
    prev_x = x
    prev_y = y
    
    # else:
    #     print area, w, h
    #     dd = cv2.drawContours(blank_background, [cont], -1, (255,255,255), thickness=2)
    #     cv2.imshow( 'blank',dd)
    #     cv2.waitKey()

blank_background = np.zeros_like(edged)
contours_filtered = sorted(contours_dict.values(), key=cv2.boundingRect)

img_contours = cv2.drawContours(blank_background, contours_filtered, -1, (255,255,255), thickness=2)
cv2.imshow( 'img_contours',img_contours)
cv2.waitKey()

def is_overlapping_horizontally(box1, box2):
    x1, _, w1, _ = box1
    x2, _, _, _ = box2
    if x1 > x2:
        return is_overlapping_horizontally(box2, box1)
    return (x2 - x1) < w1

def merge(box1, box2):
    assert is_overlapping_horizontally(box1, box2)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x = min(x1, x2)
    w = max(x1 + w1, x2 + w2) - x
    y = min(y1, y2)
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)

def windows(contours):
    """return List[Tuple[x: Int, y: Int, w: Int, h: Int]]"""
    boxes = []
    for cont in contours:
        box = cv2.boundingRect(cont)
        if not boxes:
            boxes.append(box)
        else:
            if is_overlapping_horizontally(boxes[-1], box):
                last_box = boxes.pop()
                merged_box = merge(box, last_box)
                boxes.append(merged_box)
            else:
                boxes.append(box)
    return boxes

boxes = windows(contours_filtered)

# ------------
# Boxes
img = image.copy()
for box in boxes:
    x, y, w, h = box
    img = cv2.rectangle(img, (x-1, y-1), (x + w + 1, y + h + 1), (0, 255, 0), 2)
    cv2.imshow( 'gray',img)
    cv2.waitKey()
# ------------

rects = [cv2.boundingRect(ctr) for ctr in contours_filtered]
# for rect in rects:
for box in boxes:
    # Draw the rectangles
    # cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    cv2.rectangle(img, (x, y), (x + w , y + h ), (0, 255, 0), 2)
    # Make the rectangular region around the digit
    # leng = int(box[3] * 1.6)
    leng = int(box[3] )

    pt1 = int(box[1] + box[3] // 2 - leng // 2)
    pt2 = int(box[0] + box[2] // 2 - leng // 2)
    roi = img_contours[pt1:pt1+leng, pt2:pt2+leng]
    roi = img_contours[pt1:pt1+leng, pt2:pt2+leng]
    # roi = contours_filtered[pt1:pt1+leng, pt2:pt2+leng]
    # roi = box[pt1:pt1+leng, pt2:pt2+leng]

    # cv2.imshow("roi", roi)
    # cv2.waitKey()  
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # cv2.imshow("roi", roi)
    # cv2.waitKey()    
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(image, str(int(nbr[0])), (box[0], box[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", image)
cv2.waitKey()
