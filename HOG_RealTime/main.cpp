#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

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
    int capture_width = 640;
    int capture_height = 480;
    int framerate = 15 ;
    int display_width = 640;
    int display_height = 480;

    float scale = 0.4;

    //reset frame average
    std::string pipeline = gstreamer_pipeline(capture_width, capture_height, framerate,
                                              display_width, display_height);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n\n\n";


    HOGDescriptor hog; //initialization of hog descriptor
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());


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
    	if (!cap.read(frame))
    	{
            std::cout<<"Capture read error"<<std::endl;
            break;
        }
        resize(frame, frame, Size(), scale, scale, INTER_LINEAR);
        vector<Rect> bodies;
        auto start = getTickCount();
        hog.detectMultiScale(frame, bodies, 0, Size(8,8), Size(), 1.05, 2, false);
        auto end = getTickCount();

        auto totalTime = (end - start)/ getTickFrequency();

        auto fps = 1/totalTime;
        //show frame
        for(vector<Rect>::iterator i = bodies.begin(); i != bodies.end(); ++i)
        {
            Rect &r = *i;
            rectangle(frame, r.tl(), r.br(), Scalar(0,255,0), 2);
        }

        putText(frame, to_string(fps) + " fps", Point(40, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(250, 0, 150), 2);

        resize(frame, frame, Size(), 3, 3, INTER_LINEAR);
        cv::imshow("Camera",frame);

        char esc = cv::waitKey(5);
        if(esc == 27)
        {
            //imwrite("/home/francesco/Desktop/OpenCV/HOG/Results/ped_05 .jpg", frame);
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}


