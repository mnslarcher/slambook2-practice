#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

/*
 * This program demonstrates how to estimate camera motion using 2D-2D feature
 * matching
 */

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches);

void pose_estimation_2d2d(vector<KeyPoint> keypoints_1,
                          vector<KeyPoint> keypoints_2, vector<DMatch> matches,
                          Mat &R, Mat &t);

// Pixel coordinates to camera normalized coordinates
Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "Usage: pose_estimation_2d2d img1 img2" << endl;
    return 1;
  }
  // Read images
  Mat img_1 = imread(argv[1], IMREAD_COLOR);
  Mat img_2 = imread(argv[2], IMREAD_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "In total, we get " << matches.size() << " set of feature points"
       << endl;

  // Estimation of motion between two images
  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  //  Verify that E = t^ R * scale
  Mat t_x = (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
             t.at<double>(2, 0), 0, -t.at<double>(0, 0), -t.at<double>(1, 0),
             t.at<double>(0, 0), 0);

  cout << "t^ R = " << endl << t_x * R << endl;

  // Check epipolar constraints
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  for (DMatch m : matches) {
    Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
    Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
    Mat d = y2.t() * t_x * R * y1;
    cout << "Epipolar constraint = " << d << endl;
  }
  return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches) {
  // Initialization
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create("BruteForce-Hamming");

  // Detect Oriented FAST
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  // Compute BRIEF descriptor
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  // Use Hamming distance to match the features
  vector<DMatch> all_matches;
  matcher->match(descriptors_1, descriptors_2, all_matches);

  // Matching point pair screening
  double min_dist = 10000, max_dist = 0;

  // Find the minimum and maximum distance between all matches, i.e., the
  // distance between the most similar and least similar sets of points
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = all_matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("Max dist: %f \n", max_dist);
  printf("Min dist: %f \n", min_dist);

  // Remove the bad matchig
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (all_matches[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(all_matches[i]);
    }
  }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                 (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void pose_estimation_2d2d(vector<KeyPoint> keypoints_1,
                          vector<KeyPoint> keypoints_2, vector<DMatch> matches,
                          Mat &R, Mat &t) {
  // Camera internals, TUM Freiburg2
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  // Convert matching points to the form of vector<Point2f>
  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int)matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  // Calculate the fundamental matrix
  Mat fundamental_matrix;
  fundamental_matrix = findFundamentalMat(points1, points2, FM_8POINT);
  cout << "Fundamental matrix is " << endl << fundamental_matrix << endl;

  // Calculate the essential matrix
  // Camera optical center, TUM dataset calibration value
  Point2d principal_point(325.1, 249.7);
  // Camera focal length, TUM dataset calibration value
  double focal_length = 521;
  Mat essential_matrix;
  essential_matrix =
      findEssentialMat(points1, points2, focal_length, principal_point);
  cout << "essential_matrix is " << endl << essential_matrix << endl;

  // Calculate the homography matrix
  // But in this case, the scene is not flat and the homography matrix is
  // not very meaningful
  Mat homography_matrix;
  homography_matrix = findHomography(points1, points2, RANSAC, 3);

  // Recover rotation and translation information from the essential matrix.
  recoverPose(essential_matrix, points1, points2, R, t, focal_length,
              principal_point);
  cout << "R is " << endl << R << endl;
  cout << "t is " << endl << t << endl;
}