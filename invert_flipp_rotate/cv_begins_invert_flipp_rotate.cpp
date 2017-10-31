#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>

using namespace cv;

int main() {

	//*** INVERT ***
	
	IplImage* orImage = cvLoadImage("img.jpg");
	IplImage* invIm = cvCloneImage(orImage);
	
	//display the original image
	cvNamedWindow("MyWindow");
	cvShowImage("MyWindow", orImage);

	//invert and display the inverted image
	cvNot(orImage, invIm);
	cvNamedWindow("Inverted");
	cvShowImage("Inverted", invIm);

	cvWaitKey(0);

	//cleaning up
	cvDestroyWindow("MyWindow");
	cvDestroyWindow("Inverted");


	//*** ANOTHER WAY TO INVERT ***
	
	Mat img = imread("img.jpg");
	namedWindow("MyImage");
	imshow("MyImage", img);
	
	//matrix with zeros
	Mat resinvert = Mat::zeros(img.size(), img.type());
	
	//matrix of elems = 255
	Mat subMat = Mat(img.size(), img.type(), Scalar(1, 1, 1)) * 255;

	//subtract img matrix from subMat - to get inverted pixels
	subtract(subMat, img, resinvert);

	namedWindow("Inverted v2");
	imshow("Inverted v2", resinvert);

	waitKey(0);

	destroyWindow("Inverted v2");
	imwrite("inverted.jpg", resinvert);


	//*** FLIP ***

	Mat resflip;

	flip(img, resflip, -1);
	namedWindow("Flipped");
	imshow("Flipped", resflip);

	waitKey(0);

	imwrite("flipped.jpg", resflip);
	destroyWindow("Flipped");

	
	//*** ROTATE ***

	Mat resrotate;

	Point2d pt(img.cols / 2.0, img.rows / 2.0);
	Mat matRot = getRotationMatrix2D(pt, 30, 2.0/3);

	warpAffine(img, resrotate, matRot, img.size());

	namedWindow("Rotated30");
	imshow("Rotated30", resrotate);

	waitKey(0);

	imwrite("rotated.jpg", resrotate);
	destroyWindow("Rotated30");

	return 0;
}