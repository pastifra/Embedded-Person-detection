#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "class.h"

using namespace cv;
using namespace std;

void detectAndDisplay(cv::Mat frame, cv::CascadeClassifier face_cascade, cv::CascadeClassifier body_cascade)
{
    /* For efficiency reasons and to reduce computations,
    /  convert the frame into GrayScale and resize it significantely to reduce the number of pixels hence features
    /  equalize hist to also reduce the number of features and mantain only the significant ones */
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    resize(frame_gray, frame_gray, Size(), 0.5, 0.5, INTER_LINEAR);
    equalizeHist(frame_gray,frame_gray);

    std::vector<Rect> faces; // Rect is a Template class for 2D rectangles, it contains top left corner(x and y), width and height
    face_cascade.detectMultiScale(frame_gray, faces); // Detect objects of different sizes in the input image, the rectangle should contain the detected object

    //std::vector<Rect> bodies; // Detect also for bodies
    //body_cascade.detectMultiScale(frame_gray, bodies);

    /* If some faces are found, draw a circle around the center of the first detected face */
    if(faces.size() > 0)
    {
        Point center(faces[0].x + faces[0].width/2, faces[0].y + faces[0].height/2);
        circle(frame_gray, center, faces[0].width/2, Scalar(255), 8, LINE_AA, 0);

    }

    /* If some bodies are found draw a circle around the first detected body center */
    /*if(bodies.size() > 0)
    {
        Point center(bodies[0].x + bodies[0].width/2, bodies[0].y + bodies[0].height/2);
        circle(frame_gray, center, bodies[0].width/2, Scalar(255), 7, LINE_AA, 0);
        imwrite("bodies.jpg", frame_gray);
    }*/


    resize(frame_gray, frame_gray, Size(), 1.6, 1.6, INTER_LINEAR); //Scale up the image to better visualization on the monitor

    imshow("Capture", frame_gray);
}
