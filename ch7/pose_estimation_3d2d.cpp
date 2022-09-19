#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sophus/se3.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches);

Point2f pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
    VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
    VecVector3d;

void bundleAdjustmentG2O(const VecVector3d &points_3d,
                         const VecVector2d &points_2d, const Mat &K,
                         Sophus::SE3d &pose);

// BA by Gauss-Newton
void bundleAdjustmentGaussNewton(const VecVector3d &points_3d,
                                 const VecVector2d &points_2d, const Mat &K,
                                 Sophus::SE3d &pose);

int main(int argc, char **argv) {
  if (argc != 5) {
    cout << "Usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
    return 1;
  }

  // Read images
  Mat img_1 = imread(argv[1], IMREAD_COLOR);
  Mat img_2 = imread(argv[2], IMREAD_COLOR);
  assert(img_1.data && img_2.data &&
         "Can not load images!");  // Trick to print a msg
  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "In total, we found " << matches.size() << " set of matching points"
       << endl;

  // Create 3D points
  Mat d1 = imread(argv[3], IMREAD_UNCHANGED);  // Depth map for 16-bit unsigned
                                               // numbers, single channel image
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3d> pts_3d;
  vector<Point2f> pts_2d;
  for (DMatch m : matches) {
    ushort d = d1.ptr<unsigned short>(
        int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0)  // bad depth
      continue;

    float dd = d / 5000.0;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);
  }

  cout << "3D-2D pairs: " << pts_3d.size() << endl;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  Mat r, t;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t,
           false);  // Call OpenCV's PnP solver, optionally EPNP, DLS, etc.
  Mat R;
  Rodrigues(r, R);  // r is in the form of a rotation vector, which is
                    // converted to a matrix using the Rodrigues formula

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "Solve PnP in OpenCV cost time: " << time_used.count() << " seconds."
       << endl;

  cout << "R:" << endl << R << endl;
  cout << "t:" << endl << t << endl;

  VecVector3d pts_3d_eigen;
  VecVector2d pts_2d_eigen;
  for (size_t i = 0; i < pts_3d.size(); ++i) {
    pts_3d_eigen.push_back(
        Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
  }

  cout << "Calling bundle adjustment by Gauss-Newton" << endl;
  Sophus::SE3d pose_gn;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "Solve PnP by Gauss-Newton cost time: " << time_used.count()
       << " seconds." << endl;

  cout << "Calling bundle adjustment by g2o" << endl;
  Sophus::SE3d pose_g2o;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "Solve PnP by g2o cost time: " << time_used.count() << " seconds."
       << endl;

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

Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                 (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void bundleAdjustmentGaussNewton(const VecVector3d &points_3d,
                                 const VecVector2d &points_2d, const Mat &K,
                                 Sophus::SE3d &pose) {
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  const int iterations = 10;
  double cost = 0, lastCost = 0;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  for (int iter = 0; iter < iterations; iter++) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b = Vector6d::Zero();

    cost = 0;
    // Compute cost
    for (int i = 0; i < points_3d.size(); i++) {
      Eigen::Vector3d pc = pose * points_3d[i];
      double inv_z = 1.0 / pc[2];
      double inv_z2 = inv_z * inv_z;
      Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
      Eigen::Vector2d e = points_2d[i] - proj;
      cost += e.squaredNorm();
      Eigen::Matrix<double, 2, 6> J;
      J << -fx * inv_z, 0, fx * pc[0] * inv_z2, fx * pc[0] * pc[1] * inv_z2,
          -fx - fx * pc[0] * pc[0] * inv_z2, fx * pc[1] * inv_z, 0, -fy * inv_z,
          fy * pc[1] * inv_z2, fy + fy * pc[1] * pc[1] * inv_z2,
          -fy * pc[0] * pc[1] * inv_z2, -fy * pc[0] * inv_z;

      H += J.transpose() * J;
      b += -J.transpose() * e;
    }

    Vector6d dx;
    dx = H.ldlt().solve(b);
    if (isnan(dx[0])) {
      cout << "Result is nan!" << endl;
      break;
    }

    if (iter > 0 && cost >= lastCost) {
      // Cost increase, update is not good
      cout << "Cost: " << cost << ", last cost: " << lastCost << endl;
      break;
    }

    // Update your estimation
    pose = Sophus::SE3d::exp(dx) * pose;
    lastCost = cost;

    cout << "Iteration " << iter << " cost = " << setprecision(12) << cost
         << endl;
    if (dx.norm() < 1e-6) {
      // Converge
      break;
    }
  }
  cout << "Pose by G-N:\n" << pose.matrix() << endl;
}

// Vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  virtual void setToOriginImpl() override { _estimate = Sophus::SE3d(); }

  // Left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4],
        update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }
  virtual bool read(istream &in) override {}
  virtual bool write(ostream &out) const override {}
};

class EdgeProjection
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K)
      : _pos3d(pos), _K(K) {}

  virtual void computeError() override {
    const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();  // The first 2 coeffs
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2,
        -fx - fx * X * X / Z2, fx * Y / Z, 0, -fy / Z, fy * Y / (Z * Z),
        fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  }

  virtual bool read(istream &in) override {}
  virtual bool write(ostream &out) const override {}

 private:
  Eigen::Vector3d _pos3d;
  Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(const VecVector3d &points_3d,
                         const VecVector2d &points_2d, const Mat &K,
                         Sophus::SE3d &pose) {
  // Build graph optimization, set g2o first
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>
      BlockSolverType;  // Pose is 6, landmark is 3

  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
      LinearSolverType;  // Linear solver type

  // Gradient descent method, can choose from GN, LM, DogLeg
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

  g2o::SparseOptimizer optimizer;  // Set up the solver
  optimizer.setAlgorithm(solver);  // Graph model
  optimizer.setVerbose(true);      // Turn on debug output

  // Vertex
  VertexPose *vertex_pose = new VertexPose();  // Camera vertex_pose
  vertex_pose->setId(0);
vertex_pose-
  // K

  // Edges
}