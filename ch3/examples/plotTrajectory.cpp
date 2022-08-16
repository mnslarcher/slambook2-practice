#include <pangolin/pangolin.h>
#include <unistd.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

string trajectory_file = "./examples/trajectory.txt";

void DrawTrajectory(vector<Isometry3d, aligned_allocator<Isometry3d>>);

int main(int argc, char **argv) {
  vector<Isometry3d, aligned_allocator<Isometry3d>>
      poses;  // use a custom allocator
  ifstream fin(trajectory_file);
  if (!fin) {
    cout << "Cannot find trajectory file at " << trajectory_file << endl;
    return 1;
  }
  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
    Twr.pretranslate(Vector3d(tx, ty, tz));
    poses.push_back(Twr);
  }
  cout << "Read total " << poses.size() << " pose entries" << endl;

  DrawTrajectory(poses);
  return 0;
}

void DrawTrajectory(vector<Isometry3d, aligned_allocator<Isometry3d>> poses) {
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  // Once enabled, OpenGL automatically stores fragments their z-values in the
  // depth buffer if they passed the depth test and discards fragments if they
  // failed the depth test accordingly
  glEnable(GL_DEPTH_TEST);
  // While discarding fragments is great and all, it doesn't give us the
  // flexibility to render semi-transparent images; we either render the
  // fragment or completely discard it. To render images with different levels
  // of transparency we have to enable blending. Like most of OpenGL's
  // functionality we can enable blending by enabling GL_BLEND:
  glEnable(GL_BLEND);
  // To get the blending result of our little two square example, we want to
  // take the alpha of the source color vector for the source factor and 1−alpha
  // of the same color vector for the destination factor. This translates to
  // glBlendFunc as follows:
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0.0, -0.1, -1.8, 0.0, 0.0, 0.0, 0.0, -1.0,
                                0.0));
  // d_cam is a reference
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                              .SetHandler(new pangolin::Handler3D(s_cam));
  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glLineWidth(2);
    for (size_t i = 0; i < poses.size(); i++) {
      Vector3d Ow = poses[i].translation();
      Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
      Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
      Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Xw[0], Xw[1], Xw[2]);
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Yw[0], Yw[1], Yw[2]);
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Zw[0], Zw[1], Zw[2]);
      glEnd();
    }
    for (size_t i = 0; i < poses.size(); i++) {
      glColor3f(0.0, 0.0, 0.0);
      glBegin(GL_LINES);
      auto p1 = poses[i], p2 = poses[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);
  }
}