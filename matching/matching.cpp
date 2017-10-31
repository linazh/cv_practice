#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int main() {
	
	Mat img1 = imread("img.jpg", 0);
	Mat img2 = imread("rotated.jpg", 0);

	if (img1.empty() || img2.empty())
	{
		cout << "Can't read one of the images\n";
		return -1;
	}

	// detecting keypoints
	
	SiftFeatureDetector detector(400);
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(img1, keypoints1);
	detector.detect(img2, keypoints2);

	// computing descriptors
	
	SiftDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(img1, keypoints1, descriptors1);
	extractor.compute(img2, keypoints2, descriptors2);

	// matching descriptors
	
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches_vector;
	matcher.match(descriptors1, descriptors2, matches_vector);

	//cout << descriptors1.size() << endl;

	/*
	for (int i = 0; i < matches_vector.size(); ++i)
	{
		cout << matches_vector[i].queryIdx << "  " << matches_vector[i].trainIdx << endl;
	}
	*/

	// drawing the results
	
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, matches_vector, img_matches);
	
	namedWindow("Matches result");
	imshow("Matches result", img_matches);

	waitKey(0);

	return 0;
}