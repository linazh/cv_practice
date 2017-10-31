#include <cv.h>
#include <highgui.h>
#include <math.h>

using namespace cv;

int main() {
	
	IplImage *src = cvLoadImage("image.jpg"), *blursrc;
	IplImage *dst = cvCloneImage(src), *dst2 = cvCloneImage(src);
	IplImage *dstblur = cvCloneImage(src), *dst2blur = cvCloneImage(src);
	Mat blur;

	// *** Gradient with Sobel Operator ***

	int xorder = 2, yorder = 0; //order of derivate from 0 to 2; only one(!) dimension can equal 0;
	
	medianBlur(Mat(src, true), blur, 5); //ksize is to be odd and > 1; ex. 3, 5, 7,...

	cvSobel(src, dst, xorder, yorder, 3);

	cvNamedWindow("Source", WINDOW_AUTOSIZE);
	cvNamedWindow("Sobel", WINDOW_AUTOSIZE);

	cvShowImage("Source", src);
	cvShowImage("Sobel", dst);

	waitKey(0);

	// *** Sobel with Blured Source ***
	
	cvSobel(&(blur.operator IplImage()), dstblur, xorder, yorder, 3);

	namedWindow("Source Blured", WINDOW_AUTOSIZE);
	imshow("Source Blured", blur);

	waitKey(0);

	cvNamedWindow("Sobel Blured", WINDOW_AUTOSIZE);
	cvShowImage("Sobel Blured", dstblur);

	waitKey(0);

	// *** Gradient with Laplace operator ***

	cvLaplace(src, dst2, 3);

	cvNamedWindow("Laplace", WINDOW_AUTOSIZE);
	cvShowImage("Laplace", dst2);

	waitKey(0);

	// *** Laplace with Blured Source

	cvLaplace(&(blur.operator IplImage()), dst2blur, 3);

	cvNamedWindow("Laplace Blured", WINDOW_AUTOSIZE);
	cvShowImage("Laplace Blured", dst2blur);

	waitKey(0);

	// *** Save Images ***

	cvSaveImage("bluredSource.jpg", &(blur.operator IplImage()));
	cvSaveImage("sobel.jpg", dst);
	cvSaveImage("sobelBlured.jpg", dstblur);
	cvSaveImage("laplace.jpg", dst2);
	cvSaveImage("laplaceBlured.jpg", dst2blur);

	return 0;
}