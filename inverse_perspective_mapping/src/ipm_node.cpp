#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class IPMNode : public rclcpp::Node
{
public:
  IPMNode()
  : Node("ipm_node")
  {
    // Declare parameters
    this->declare_parameter<double>("camera_pos_x", 0.0);
    this->declare_parameter<double>("camera_pos_y", 0.0);
    this->declare_parameter<double>("camera_pos_z", 80.0);
    this->declare_parameter<double>("fov_h", 78.0);
    this->declare_parameter<double>("fov_v", 42.5);
    this->declare_parameter<int>("src_width", 1280);
    this->declare_parameter<int>("src_height", 720);
    this->declare_parameter<int>("dst_width", 700);
    this->declare_parameter<int>("dst_height", 700);
    this->declare_parameter<double>("camera_tilt_angle", 0.0);

    // Get parameters
    this->get_parameter("camera_pos_x", camera_pos_x_);
    this->get_parameter("camera_pos_y", camera_pos_y_);
    this->get_parameter("camera_pos_z", camera_pos_z_);
    this->get_parameter("fov_h", fov_h_);
    this->get_parameter("fov_v", fov_v_);
    this->get_parameter("src_width", src_width_);
    this->get_parameter("src_height", src_height_);
    this->get_parameter("dst_width", dst_width_);
    this->get_parameter("dst_height", dst_height_);
    this->get_parameter("camera_tilt_angle", camera_tilt_angle_);

    // Setup subscriber and publisher
    image_sub_ = image_transport::create_subscription(
      this, "/v4l/camera/image_raw",
      std::bind(&IPMNode::imageCallback, this, std::placeholders::_1),
      "raw");

    image_pub_ = image_transport::create_publisher(this, "ipm/image", rclcpp::QoS(2).get_rmw_qos_profile());

    // Setup IPM transformation matrices
    setupIPM();
  }

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
  {
    try {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat src_img = cv_ptr->image;

      cv::Mat dst_img;
      cv::warpPerspective(src_img, dst_img, H_, cv::Size(dst_width_, dst_height_));

      cv_bridge::CvImage out_msg;
      out_msg.header = msg->header;
      out_msg.encoding = sensor_msgs::image_encodings::BGR8;
      out_msg.image = dst_img;

      image_pub_.publish(out_msg.toImageMsg());
    }
    catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
  }

  void setupIPM()
  {
    double w = src_width_;
    double h = src_height_;

    // Source points
    std::vector<cv::Point2f> src_pts;
    src_pts.push_back(cv::Point2f(w * 0.2, h * 0.2));  // top-left
    src_pts.push_back(cv::Point2f(w * 0.8, h * 0.2));  // top-right
    src_pts.push_back(cv::Point2f(w, h));              // bottom-right
    src_pts.push_back(cv::Point2f(0, h));              // bottom-left

    // Destination points
    std::vector<cv::Point2f> dst_pts;
    dst_pts.push_back(cv::Point2f(0, 0));               // top-left
    dst_pts.push_back(cv::Point2f(dst_width_, 0));      // top-right
    dst_pts.push_back(cv::Point2f(dst_width_, dst_height_)); // bottom-right
    dst_pts.push_back(cv::Point2f(0, dst_height_));     // bottom-left

    H_ = cv::getPerspectiveTransform(src_pts, dst_pts);
  }

  double camera_pos_x_, camera_pos_y_, camera_pos_z_;
  double fov_h_, fov_v_;
  int src_width_, src_height_, dst_width_, dst_height_;
  double camera_tilt_angle_;
  cv::Mat H_;

  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IPMNode>());
  rclcpp::shutdown();
  return 0;
}

