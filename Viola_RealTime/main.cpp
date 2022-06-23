#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "class.h"

using namespace std;
using namespace cv;

/// For the Raspberry Pi 32-bit Bullseye OS

std::string gstreamer_pipeline(int capture_width, int capture_height, int framerate, int display_width, int display_height) {
    return
            " libcamerasrc ! video/x-raw, "
            " width=(int)" + std::to_string(capture_width) + ","
            " height=(int)" + std::to_string(capture_height) + ","
            " framerate=(fraction)" + std::to_string(framerate) +"/1 !"
            " videoconvert ! videoscale !"
            " video/x-raw,"
            " width=(int)" + std::to_string(display_width) + ","
            " height=(int)" + std::to_string(display_height) + " ! appsink";
}

int main()
{
    //pipeline parameters
    int capture_width = 640; //1280 ;
    int capture_height = 480; //720 ;
    int framerate = 15 ;
    int display_width = 640; //1280 ;
    int display_height = 480; //720 ;

    //reset frame average
    std::string pipeline = gstreamer_pipeline(capture_width, capture_height, framerate,
                                              display_width, display_height);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n\n\n";

    /* Cascade classifiers inizialization; a series of weak lerners put into a cascade
    /  one after the other, in this way they become a good strong classifier */

    CascadeClassifier face_cascade;
    CascadeClassifier body_cascade;

    /* Load the classifiers from a "data" provided by openCV, so already trained
    /  It is basetoolpoollikujmnhhytgbvfredcxswqazasdsdfdgffhgjhjklipiuyutreewd on Haar features that use the integral image for evaluation and are really fast to compute */
    String face_cascade_name = "/home/francesco/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
    String body_cascade_name = "/home/francesco/opencv/data/haarcascades/haarcascade_fullbody.xml";
    face_cascade.load(face_cascade_name);
    body_cascade.load(body_cascade_name);

    /* Start the video capture with the gstreamer Pipeline */
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
        std::cout<<"Failed to open camera."<<std::endl;
        return (-1);
    }

    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    cv::Mat frame;

    std::cout << "Hit ESC to exit" << "\n" ;
    while(true)
    {
    	if (!cap.read(frame)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }

        /*Get the acquired frame and elaborate it with a custom class*/
        detectAndDisplay(frame, face_cascade, body_cascade);

        char esc = cv::waitKey(5);
        if(esc == 27) break;
    }
    cap.release(); //Stop the camera capture
    cv::destroyAllWindows();

    return 0;
}


