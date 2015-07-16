#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

static const std::string ORDINARY_WINDOW = "original window";
static const std::string EDGE_WINDOW = "edge window";
static const std::string MERGE_WINDOW= "merge window";

bool compare_function(const Vec2f& a, const Vec2f& b){ return a[0] < b[0];}

class ImageConverter
{

    Mat cameraMatrix = (Mat_<float>(3,3) <<
            3.6074152472743589e+02, 0., 3.6586328234052519e+02,
            0., 3.6063001844318876e+02, 2.3570199656779755e+02,
            0., 0., 1
            );

    Mat distCoeffs = (Mat_<float>(5,1) <<
            -2.7368371924753454e-01, 7.8500838559257463e-02,
            -3.8624488476715640e-04, 1.0467601355110387e-04,
            -1.0033752790579050e-02
            );

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;

    unsigned line_left = 0x100;
    unsigned line_middle = 0x010;
    unsigned line_right = 0x001;
    unsigned line_current_lr = 0x000;
    unsigned line_current_ud = 0x000;

    int line_reset_counter_lr = 0;
    int line_reset_counter_ud = 0;

    int left_to_right = 0;
    int up_to_down = 0;

    public:
    ImageConverter()
        : it_(nh_)
    {
        // Subscrive to input video feed and publish output video feed
        image_sub_ = it_.subscribe("/camera/image", 1,
                &ImageConverter::imageCb, this);
        image_pub_ = it_.advertise("/counter/output_video", 1);

        cv::namedWindow(ORDINARY_WINDOW);
        //cv::namedWindow(EDGE_WINDOW);
        cv::namedWindow(MERGE_WINDOW);
    }

    ~ImageConverter()
    {
        cv::destroyWindow(ORDINARY_WINDOW);
        cv::destroyWindow(EDGE_WINDOW);
    }


    void imageCb(const sensor_msgs::ImageConstPtr& msg)
    {
        //read frame from cv_bridge
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        Mat undistorted;
        //distort image
        undistort(cv_ptr->image, undistorted, cameraMatrix, distCoeffs);
        //resize
        resize(undistorted,undistorted,Size(480,320));
        //binary
        threshold(undistorted, undistorted, 200, 255, THRESH_BINARY);

        Mat drawing = Mat::zeros( undistorted.size(), CV_8UC1 );

        //find image skeleton to filter lines
        Mat skel(undistorted.size(), CV_8UC1, Scalar(0));
        Mat temp(undistorted.size(), CV_8UC1);
        Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

        bool done;
        do
        {
            cv::morphologyEx(undistorted, temp, cv::MORPH_OPEN, element);
            cv::bitwise_not(temp, temp);
            cv::bitwise_and(undistorted, temp, temp);
            cv::bitwise_or(skel, temp, skel);
            cv::erode(undistorted, undistorted, element);

            double max;
            cv::minMaxLoc(undistorted, 0, &max);
            done = (max == 0);
        } while (!done);

        //Houghline transform
        vector<Vec2f> lines;
        vector<Vec2f> lines_;
        HoughLines(skel, lines, 3, CV_PI/180, 150, 0, 0 );

        //std::sort(lines.begin(), lines.end(), compare_function);

        for (int i = 0; i < lines.size(); i ++) {
            //convert negative rho into positive
            if(lines[i][0] < 0) {
                lines[i][0] = -lines[i][0];
                lines[i][1] = lines[i][1]-M_PI;
            }
            //convert negative theta into positive
            if(abs(lines[i][1] +M_PI/2)<0.2) {
                lines[i][0] = 320-lines[i][0];
                lines[i][1] = lines[i][1]+M_PI;
            }
        }

        //merge distorted lines
        while (!lines.empty()) {
            float rho_sum = lines[0][0];
            float theta_sum = lines[0][1];
            int counter = 1;
            vector <int> index;

            //find the same line as lines[0]
            for (int i = 1; i < lines.size(); i++ ){
                //std::cout << i <<" " << lines[i][0] <<","<< lines[i][1] << std::endl;
                if ((abs(lines[i][0] - rho_sum/counter) < 30) && (abs(theta_sum/counter - lines[i][1])<0.2)) {
                    rho_sum += lines[i][0];
                    theta_sum += lines[i][1];
                    counter ++;
                    index.push_back(i);
                }
            }

            //remove merged lines
            std::sort(index.begin(),index.end(),std::greater<int>());

            for (int i = 0; i < index.size(); i ++) {
                lines.erase(lines.begin()+index[i]);
                //std::cout << "erased: " << index[i] << endl;
            }
            lines.erase(lines.begin());

            //store result
            Vec2f result(rho_sum/counter, theta_sum/counter);
            lines_.push_back(result);

        }


        //draw lines
        for( int i = 0; i < lines_.size(); i++ )
        {
            float rho = lines_[i][0], theta = lines_[i][1];
            std::cout << i <<" " << rho <<","<<  theta << std::endl;
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            line( drawing, pt1, pt2, Scalar::all(255), 1, CV_AA);
            circle( drawing, Point(240,160), 5, Scalar::all(255),-1 );
        }

        /*
         * count blocks
         *
         * ----->
         * |(u,v)
         * v
         *
         * The method works well for perfect horizontal/vertical lines
         *----------------------------------------------
         *| left region | middle region | right region |
         *----------------------------------------------
         * left <-> middle <-> right == 0x111
         *
         * The noise comes from rho/theta, for horizontal lines, rho varies terribly for theta > 0.1, which means the middle of line may not be the camera vertical center
         *
         * Two possible way to reduce noise:
         * 1. Turning parameters wider
         * 2. Remove the middle region
         * 3. Fuse the IMU data(the yaw Angle) to offset the theta of line data
         *
         */
        for( int i = 0; i < lines_.size(); i++ ){
            float rho = lines_[i][0], theta = lines_[i][1];
            if(abs(theta)<0.2){
                if ((rho-240)>-15 && (rho-240)<-5) {
                    line_reset_counter_lr = 0;
                    line_current_lr |= line_left;
                    if(line_current_lr == 0x111) {
                        //the left_right here is for camera, which is mirrored direction of the line passing it
                        left_to_right ++;
                        line_current_lr = line_left;
                    }
                }
                else if((rho-240)>-5 && (rho-240)<5){
                    line_reset_counter_lr = 0;
                    line_current_lr |= line_middle;
                }
                else if((rho-240)>5 && (rho-240)<15){
                    line_reset_counter_lr = 0;
                    line_current_lr |= line_right;
                    if(line_current_lr == 0x111){
                        left_to_right --;
                        line_current_lr = line_right;
                    }
                }
                else {
                    line_reset_counter_lr ++;
                }
            }

            else if((abs(theta-M_PI/2)<0.2) || abs(theta+M_PI/2)<0.2) {
                if((rho-160)>-15 && (rho-160)<-5){
                    line_reset_counter_ud = 0;
                    line_current_ud |= line_left;
                    if(line_current_ud == 0x111){
                        up_to_down ++;
                        line_current_ud = line_left;
                    }
                }
                else if((rho-160)>-5 && (rho-160)<5){
                    line_reset_counter_ud = 0;
                    line_current_ud |= line_middle;
                }
                else if((rho-160)>5 && (rho-160)<15){
                    line_reset_counter_ud = 0;
                    line_current_ud |= line_right;
                    if(line_current_ud == 0x111){
                        up_to_down --;
                        line_current_ud = line_right;
                    }
                }
                else {
                    line_reset_counter_ud ++;
                }
            }
        }

        //reset if leave region more than 10 frames
        if (line_reset_counter_lr == 10)
            line_current_lr = 0x000;

        if (line_reset_counter_ud == 10)
            line_current_ud = 0x000;

        cout << "(" << left_to_right <<"," << up_to_down<<")" << endl;
        // Update GUI Window
        imshow(ORDINARY_WINDOW, skel);
        imshow(MERGE_WINDOW, drawing);
        waitKey(3);

        // Output modified video stream
        image_pub_.publish(cv_ptr->toImageMsg());
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_converter");
    ImageConverter ic;
    ros::spin();
    return 0;
}
