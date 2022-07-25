#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include "classes.h"
/// For the Raspberry Pi 64-bit Bullseye OS
using namespace std;
using namespace cv;
using namespace dnn;


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
    //----CUSTOM YOLO INITIALIZATION----//
    Net yolo = readNetFromTensorflow("/home/pastifra/Desktop/Workspace/CameraCV/Model/yolo_big_dw.pb");
    //----CAMERA PIPELINE INITIALIZATION----//
    int capture_width = 1280;
    int capture_height = 720;
    int framerate = 15;
    int display_width = 1280;
    int display_height = 720;
    Mat frame;
    string pipeline = gstreamer_pipeline(capture_width, capture_height, framerate,
                                              display_width, display_height);
    cout << "Using pipeline: \n\t" << pipeline << "\n\n\n";

    VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened())
    {
        cout<<"Failed to open camera."<<std::endl;
        return (-1);
    }

    //----DETECTION LOOP----//
    cout << "Hit ESC to exit" << "\n" ;
    while(true) //Exit the loop only if user presses ESC
    {
    	if (!cap.read(frame))
    	{
            cout<<"Capture read error"<<std::endl;
            break;
        }
        auto start = getTickCount();
        vector<Rect> propBboxes;

        vector<Mat> netOutput = preProcessAndInfer(yolo, frame);
        propBboxes = decodeNetOut(netOutput[0]);

        for(int i = 0; i < propBboxes.size(); i++)
        {
            Rect Bbox = propBboxes[0];
            rectangle(frame, Bbox, Scalar(0,0,255), 2, LINE_8);
        }

        auto end = getTickCount();
        auto totTime = (end - start)/getTickFrequency();

        auto fps = 1/totTime;

        putText(frame, to_string(fps) + " FPS", Point(70,70), FONT_HERSHEY_PLAIN, 1.5, Scalar(255,0,255), 2);

        namedWindow("Camera", WINDOW_AUTOSIZE);
        imshow("Camera", frame);

        char esc = cv::waitKey(5);
        if(esc == 27) break;
    }

    cap.release();
    destroyAllWindows() ;
    return 0;
}


