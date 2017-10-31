#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

Mat src, srcGray;

// constants for functions
int thresh = 125;
int maxThresh = 250;
int maxCorners = 90;
int maxTrackbar = 200;

RNG rng(12345); // random number generator

string src_window = "Source image";
string cornerHarris_window = "Harris Corner Detector";
string cornerShiTomasi_window = "Shi-Tomasi Corner Detector";

// Function headers
void cornerHarris_apply(int, void*);
void goodFeaturesToTrack_apply(int, void*);


// function main
int main() {

	src = imread("im.jpg", 1);
	cvtColor(src, srcGray, CV_BGR2GRAY);

	// create windows and trackbars for 2 detectors
	namedWindow(src_window);
	imshow(src_window, src);
	
	namedWindow(cornerHarris_window);
	createTrackbar("Threshold: ", cornerHarris_window, &thresh, maxThresh, cornerHarris_apply);
	
	cornerHarris_apply(0, 0);

	namedWindow(cornerShiTomasi_window);
	createTrackbar("Max corners:", cornerShiTomasi_window, &maxCorners, maxTrackbar, goodFeaturesToTrack_apply);
	
	goodFeaturesToTrack_apply(0, 0);

	waitKey(0);
	
	return(0);
}


// function for Harris Corner Detector
void cornerHarris_apply(int, void*) {

	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);

	// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	// Detecting corners
	cornerHarris(srcGray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	// Drawing circles around corners
	int rad = 5;
	for (int y = 0; y < dst_norm.rows; y++) {
		for (int x = 0; x < dst_norm.cols; x++) {
			if ((int)dst_norm.at<float>(y, x) > thresh) {
				circle(dst_norm_scaled, Point(x, y), rad, Scalar(0), 2);
			}
		}
	}
	// Showing the result
	namedWindow(cornerHarris_window);
	imshow(cornerHarris_window, dst_norm_scaled);
}


// function for Shi-Tomasi Corner Detector
void goodFeaturesToTrack_apply(int, void*) {
	if (maxCorners < 1) { maxCorners = 1; }

	Mat copy = src.clone();

	// Detector Parameters
	vector<Point2f> corners;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;

	// Detecting corners
	goodFeaturesToTrack(srcGray, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);

	// Drawing circles around corners
	int rad = 5;
	for (int i = 0; i < corners.size(); i++) {
		circle(copy, corners[i], rad, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 2);
	}

	// Showing the result
	namedWindow(cornerShiTomasi_window);
	imshow(cornerShiTomasi_window, copy);
}