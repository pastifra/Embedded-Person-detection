#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;

int main(int, char**)
{
    //Load the source image
    String input;
    cout << "Please enter the path of the first image: ";
    cin >> input;
    Mat frame = imread(input);
    if (img1.empty())
    {
        printf("Error opening the first image");
        return 0;
    }
   
    //Get scale from user input
    double scale;
    cout << "Please enter the scale at which you'd like to resize the image: (Enter 1.0 if you do not wish to resize it)"
    cin >> scale;
   
    resize(frame, frame, Size(), scale, scale, INTER_LINEAR);
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    /* Cascade classifiers inizialization; a series of weak lerners put into a cascade
    /  one after the other, in this way they become a good strong classifier */
    CascadeClassifier body_cascade;

    /* Load the classifiers from a "data" provided by openCV, so already trained
    /  It is basetoolpoollikujmnhhytgbvfredcxswqazasdsdfdgffhgjhjklipiuyutreewd on Haar features that use the integral image for evaluation and are really fast to compute */
    String body_cascade_name = ".../Viola_Jones_Still/Data/haarcascade_fullbody.xml";
    body_cascade.load(body_cascade_name);

    /* For efficiency reasons and to reduce computations,
    /  convert the frame into GrayScale and resize it significantely to reduce the number of pixels hence features
    /  equalize hist to also reduce the number of features and mantain only the significant ones */
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray,frame_gray);

    std::vector<Rect> bodies; // Rect is a Template class for 2D rectangles, it contains top left corner(x and y), width and height
    auto start = getTickCount();
    body_cascade.detectMultiScale(frame_gray, bodies); // Detect objects of different sizes in the input image, each rectangle should contain the detected object
    auto end = getTickCount();

    /* Draw the Rectangles (bounding boxes) containing the detected bodies */
    for(vector<Rect>::iterator i = bodies.begin(); i != bodies.end(); ++i)
    {
        Rect &r = *i;
        rectangle(frame, r.tl(), r.br(), Scalar(0,255,0), 2);
    }
   
    auto totalTime = (end - start)/ getTickFrequency(); // Calculation of the inference time to detect bodies
   
    putText(frame, to_string(totalTime) + " s", Point(30, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(250, 0, 150), 2);

    cv::imshow("Camera",frame);

    waitKey(0);
    return 0;
}
