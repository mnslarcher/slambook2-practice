#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }
  // Read images
  Mat img_1 = imread(argv[1], IMREAD_COLOR);
  Mat img_2 = imread(argv[2], IMREAD_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);

  // Initialization
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create("BruteForce-Hamming");

  // Detect Oriented FAST
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  // Compute BRIEF descriptor
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  Mat outimg1;
  // Scalar::all(-1) returns a scalar with all elements set to -1

  // DrawMatchesFlags::DEFAULT
  // Output image matrix will be created (Mat::create), i.e. existing memory of
  // output image may be reused. Two source image, matches and single keypoints
  // will be drawn. For each keypoint only the center point will be drawn
  // (without the circle around keypoint with keypoint size and orientation).
  drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1),
                DrawMatchesFlags::DEFAULT);
  imshow("ORB features", outimg1);

  // Use Hamming distance to match the features
  vector<DMatch> matches;
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t1 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  // Sort and remove the outliers
  // Min and max distance
  auto min_max = minmax_element(
      matches.begin(), matches.end(),
      [](const DMatch &m1, DMatch &m2) { return m1.distance < m2.distance; });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("Max dist: %f \n", max_dist);
  printf("Min dist: %f \n", min_dist);

  // Remove the bad matchig
  std::vector<DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  // Draw the results
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches,
              img_goodmatch);
  imshow("all matches", img_match);
  imshow("good matches", img_goodmatch);
  waitKey(0);

  return 0;
}