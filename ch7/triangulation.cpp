#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches);

void pose_estimation_2d2d(const vector<KeyPoint> &keypoints_1,
                          const vector<KeyPoint> &keypoints_2,
                          const vector<DMatch> &matches, Mat &R, Mat &t);

void triangulation(const vector<KeyPoint> &keypoints_1,
                   const vector<KeyPoint> &keypoints_2,
                   const vector<DMatch> &matches, const Mat &R, const Mat &t,
                   vector<Point3d> &points);

// For drawing
// Inline function is a function that is expanded in line when it is called.
// The expansion happen at compile time.
inline Scalar get_color(float depth) {
  float up_th = 50, low_th = 10, th_range = up_th - low_th;
  if (depth > up_th)
    depth = up_th;
  else if (depth < low_th)
    depth = low_th;
  return Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

// Pixel coordinates to camera-normalized coordinates
Point2f pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "Usage: triangulation img1 img2" << endl;
    return 1;
  }

  // Read image
  Mat img_1 = imread(argv[1], IMREAD_COLOR);
  Mat img_2 = imread(argv[2], IMREAD_COLOR);

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "In total, we found" << matches.size() << " matches" << endl;

  // Estimation of motion between two images
  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  // Triangulation
  vector<Point3d> points;
  triangulation(keypoints_1, keypoints_2, matches, R, t, points);

  // Verify the reprojection relationship between triangulation points and
  // feature points
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  Mat img1_plot = img_1.clone();
  Mat img2_plot = img_2.clone();
  for (int i = 0; i < matches.size(); i++) {
    // First figure
    float depth1 = points[i].z;
    cout << "Depth: " << depth1 << endl;
    Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
    circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1),
           2);

    // Second figure
    Mat pt2_trans =
        R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
    float depth2 = pt2_trans.at<double>(2, 0);
    circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2),
           2);
  }
  imshow("img 1", img1_plot);
  imshow("img_2", img2_plot);
  waitKey();

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
  // Detect Oriented FAST corner locations
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  // Calculate BRIEF descriptors based on corner point positions
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  // Match BRIEF descriptors in both images using Hamming distance
  vector<DMatch> match;
  matcher->match(descriptors_1, descriptors_2, match);

  // Matching point pair filtering
  double min_dist = 10000, max_dist = 0;

  // Find the minimum and maximum distances between all matches, that is, the
  // distance between the most similar and least similar sets of points
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  printf("Max dist: %f\n", max_dist);
  printf("Min dist: %f\n", min_dist);

  // When the distance between descriptors is greater than twice the minimum
  // distance, the match is considered wrong. But sometimes the minimum
  // distance can be very small, so set an empirical value of 30 as the lower
  // limit.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

void pose_estimation_2d2d(const vector<KeyPoint> &keypoints_1,
                          const vector<KeyPoint> &keypoints_2,
                          const vector<DMatch> &matches, Mat &R, Mat &t) {
  // Camera internal parameters, TUM Freiburg2
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  // Convert matching points to the form of vector<Point2f>
  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int)matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  // Calculate the essential matrix
  Point2d principal_point(
      325.1, 249.7);       // Primary camera, TUM dataset calibration value
  int focal_length = 521;  // Camera focal length, TUM dataset calibration value
  Mat essential_matrix;
  essential_matrix =
      findEssentialMat(points1, points2, focal_length, principal_point);

  // Recover rotation and translation information from the essence matrix
  recoverPose(essential_matrix, points1, points2, R, t, focal_length,
              principal_point);
}

void triangulation(const vector<KeyPoint> &keypoints_1,
                   const vector<KeyPoint> &keypoints_2,
                   const vector<DMatch> &matches, const Mat &R, const Mat &t,
                   vector<Point3d> &points) {
  Mat T1 = (Mat_<float>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  Mat T2 = (Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1),
            R.at<double>(0, 2), t.at<double>(0, 0), R.at<double>(1, 0),
            R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
            t.at<double>(2, 0));

  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point2f> pts_1, pts_2;
  for (DMatch m : matches) {
    // Convert pixel coordinates to camera coordinates
    pts_1.push_back(pixel2cam(keypoints_1[m.queryIdx].pt, K));
    pts_2.push_back(pixel2cam(keypoints_2[m.trainIdx].pt, K));
  }

  Mat pts_4d;
  triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

  // Convert to non-homogeneous coordinates
  for (int i = 0; i < pts_4d.cols; i++) {
    Mat x = pts_4d.col(i);   // Every col is a point
    x /= x.at<float>(3, 0);  // Normalization
    Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
    points.push_back(p);
  }
}

Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                 (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}
