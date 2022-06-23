#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;


int main(int, char**)
{
    double scale = 1.0;
    Mat frame = imread(""); //insert here path of the image
    resize(frame, frame, Size(), scale, scale, INTER_LINEAR); //optional resizing of the image to obtain better performances

    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    /*Initialization of the HOG descriptor with the default People Detector provided by OpenCV*/
    HOGDescriptor hog; 
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    vector<Rect> bodies;

    /*Actual detection over multiple scales, start and end are needed to count the inference time*/
    auto start = getTickCount();
    hog.detectMultiScale(frame, bodies, 0, Size(8,8), Size(), 1.05, 2, false);//true for Daimler
    auto end = getTickCount();
    
    auto totalTime = (end - start)/ getTickFrequency(); //Total inference time
    
    /*For loop to draw on the image all the detected bodies*/
    for(vector<Rect>::iterator i = bodies.begin(); i != bodies.end(); ++i)
    {   Rect &r = *i;
        rectangle(frame, r.tl(), r.br(), Scalar(0,255,0), 2);
    }

    
    putText(frame, to_string(totalTime) + " s", Point(30, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(250, 0, 150), 2); //Draw also the total time

    
    cv::imshow("Camera",frame);

    waitKey(0);
    return 0;
}
