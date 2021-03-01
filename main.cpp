#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;

int main()
{
	cv::Mat imgL, imgR;

	imgL = cv::imread("im2.png", cv::IMREAD_GRAYSCALE);
	imgR = cv::imread("im6.png", cv::IMREAD_GRAYSCALE);

	// Define keypoints vector
	std::vector<cv::KeyPoint> keypointsL, keypointsR;

	// Define feature detector
	cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SURF::create();

	// Keypoint detection
	ptrFeature2D->detect(imgL, keypointsL);
	ptrFeature2D->detect(imgR, keypointsR);

	// Compute descriptors
	cv::Mat descriptorsL, descriptorsR;
	ptrFeature2D->compute(imgL, keypointsL, descriptorsL);
	ptrFeature2D->compute(imgR, keypointsR, descriptorsR);

	// Match the two image descriptors
	cv::BFMatcher matcher(cv::NORM_L2);
	std::vector<cv::DMatch> outputMatches;
	matcher.match(descriptorsL, descriptorsR, outputMatches);

	// //-- Draw matches
	// Mat imgMatches;
	// drawMatches(imgL, keypointsL, imgR, keypointsR, outputMatches, imgMatches);
	// //-- Show detected matches
	// namedWindow("Matches", WINDOW_NORMAL);
	// imshow("Matches", imgMatches);
	// waitKey();

	// Convert keypoints into Point2f
	std::vector<cv::Point2f> pointsL, pointsR;
	for (auto it = outputMatches.begin(); it != outputMatches.end(); ++it)
	{
		// Get the position of left keypoints
		pointsL.push_back(keypointsL[it->queryIdx].pt);
		// Get the position of right keypoints
		pointsR.push_back(keypointsR[it->trainIdx].pt);
	}

	std::vector<uchar> inliers(pointsL.size(), 0);
	cv::Mat fundamental = cv::findFundamentalMat(
		pointsL, pointsR, // matching points
		inliers,		  // match status (inlier or outlier)
		cv::FM_RANSAC,	  // RANSAC method
		1.0,			  // distance to epipolar line
		0.98);			  // confidence probability


	// Compute homographic rectification
	cv::Mat hL, hR;
	cv::stereoRectifyUncalibrated(pointsL, pointsR, fundamental, imgL.size(), hL, hR);

	// Rectify the images through warping
	cv::Mat rectifiedL;
	cv::warpPerspective(imgL, rectifiedL, hL, imgL.size());
	cv::Mat rectifiedR;
	cv::warpPerspective(imgR, rectifiedR, hR, imgR.size());

	// Compute disparity
	cv::Mat disparity;
	cv::Ptr<cv::StereoMatcher> pStereo = cv::StereoSGBM::create(0, 32, 1);
	pStereo->compute(rectifiedL, rectifiedR, disparity);

	cv::imwrite("disparitySURF.jpg", disparity);

}
