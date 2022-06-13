#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;


int main(int, char**)
{
    double scale = 0.5;
    Mat frame = imread(""); //insert here path of the image
    resize(frame, frame, Size(), scale, scale, INTER_LINEAR);

    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    HOGDescriptor hog; //(Size(48,96), Size(16,16), Size(8,8), Size(8,8), 9); <- for daimler //initialization of hog descriptor
    //DEF = 64, DAIM = 48
    //DEF = 128, DAIM = 96
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    vector<Rect> bodies;


    auto start = getTickCount();

    hog.detectMultiScale(frame, bodies, 0, Size(8,8), Size(), 1.05, 2, false);//true for Daimler

    vector<Rect> toFile;
    auto end = getTickCount();
    for(vector<Rect>::iterator i = bodies.begin(); i != bodies.end(); ++i)
    {
        Rect &r = *i;
        rectangle(frame, r.tl(), r.br(), Scalar(0,255,0), 2);
        toFile.push_back(r);
    }

    auto totalTime = (end - start)/ getTickFrequency();
    putText(frame, to_string(totalTime) + " s", Point(30, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(250, 0, 150), 2);

    /*fstream file;
    file.open("/home/francesco/Desktop/Iou HOG/hog.txt",ios_base::out);

    for(size_t i = 0; i < toFile.size(); i++)
    {
        file<<toFile[i]<<endl;
    }

    file.close();*/
    cv::imshow("Camera",frame);

    waitKey(0);
    return 0;
}
