/* FOR THE RASPBERRY PI 32-bit BULLSEYE OS
/  This code achieves a real time person detection via the HOG detector
/  provided by OpenCV on the target device Raspberry Pi with OS mentioned above
/  In order to detect the camera connected to the Raspberry the gstreamer_pipeline is used */

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "class.h"

using namespace std;
using namespace cv;

/*Pipeline parameters*/
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
    int capture_width = 640;
    int capture_height = 480;
    int framerate = 15 ;
    int display_width = 640;
    int display_height = 480;
    float scale = 1.0;

    //reset frame average
    std::string pipeline = gstreamer_pipeline(capture_width, capture_height, framerate,
                                              display_width, display_height);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n\n\n";

    /* Cascade classifier inizialization; a series of weak lerners put into a cascade
    /  one after the other, in this way they become a good strong classifier */

    CascadeClassifier body_cascade;

    /* Load the classifiers from a "data" provided by openCV, so already trained
    /  It is based on Haar features that use the integral image for evaluation and are really fast to compute */
    String body_cascade_name = "/home/francesco/opencv/data/haarcascades/haarcascade_fullbody.xml";
    body_cascade.load(body_cascade_name);

    /* Start the video capture with the gstreamer Pipeline */
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) 
    {
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
        /* For efficiency reasons and to reduce computations,
        /  convert the frame into GrayScale and resize it significantely to reduce the number of pixels hence features
        /  equalize hist to also reduce the number of features and mantain only the significant ones */
        Mat frame_gray;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        resize(frame_gray, frame_gray, Size(), scale, scale, INTER_LINEAR);
        equalizeHist(frame_gray,frame_gray);
        
        std::vector<Rect> bodies; // Rect is a Template class for 2D rectangles, it contains top left corner(x and y), width and height
        
        auto start = getTickCount();
        body_cascade.detectMultiScale(frame_gray, bodies); // Detect objects of different sizes in the input image, the rectangle should contain the detected object
        auto end = getTickCount();
        
        auto totalTime = (end - start)/ getTickFrequency();
        auto fps = 1/totalTime;
        
        //Draw the deteced bodie on the frame
        for(vector<Rect>::iterator i = bodies.begin(); i != bodies.end(); ++i)
        {
            Rect &r = *i;
            rectangle(frame, r.tl(), r.br(), Scalar(0,255,0), 2);
        }

        putText(frame, to_string(fps) + " fps", Point(40, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(250, 0, 150), 2); //Draw also the fps

        resize(frame, frame, Size(), 3, 3, INTER_LINEAR); //Resizing for better user visualization

        char esc = cv::waitKey(5);
        if(esc == 27) break;
    }
    cap.release(); //Stop the camera capture
    cv::destroyAllWindows();

    return 0;
}


