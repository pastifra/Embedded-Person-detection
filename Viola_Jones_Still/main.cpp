#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;


int main(int, char**)
{
   
    double scale = 1.0;
    Mat frame = imread("/home/francesco/Desktop/OpenCV/Test_Dataset/img9.jpg");
    resize(frame, frame, Size(), scale, scale, INTER_LINEAR);
    

    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);


    /* Cascade classifiers inizialization; a series of weak lerners put into a cascade
    /  one after the other, in this way they become a good strong classifier */

    CascadeClassifier face_cascade;
    CascadeClassifier body_cascade;

    /* Load the classifiers from a "data" provided by openCV, so already trained
    /  It is basetoolpoollikujmnhhytgbvfredcxswqazasdsdfdgffhgjhjklipiuyutreewd on Haar features that use the integral image for evaluation and are really fast to compute */
    String body_cascade_name = "/home/francesco/opencv/data/haarcascades/haarcascade_fullbody.xml";
    body_cascade.load(body_cascade_name);



    /* For efficiency reasons and to reduce computations,
    /  convert the frame into GrayScale and resize it significantely to reduce the number of pixels hence features
    /  equalize hist to also reduce the number of features and mantain only the significant ones */
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray,frame_gray);

    std::vector<Rect> bodies; // Rect is a Template class for 2D rectangles, it contains top left corner(x and y), width and height
    auto start = getTickCount();
    body_cascade.detectMultiScale(frame_gray, bodies); // Detect objects of different sizes in the input image, the rectangle should contain the detected object
    auto end = getTickCount();
    //std::vector<Rect> bodies; // Detect also for bodies
    //body_cascade.detectMultiScale(frame_gray, bodies);

    vector<Rect> toFile;
    for(vector<Rect>::iterator i = bodies.begin(); i != bodies.end(); ++i)
    {
        Rect &r = *i;
        rectangle(frame, r.tl(), r.br(), Scalar(0,255,0), 2);
        toFile.push_back(r);
    }

    auto totalTime = (end - start)/ getTickFrequency();
    putText(frame, to_string(totalTime) + " s", Point(30, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(250, 0, 150), 2);

    cv::imshow("Camera",frame);

    waitKey(0);
    return 0;
}
