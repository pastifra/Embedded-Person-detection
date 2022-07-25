#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <vector>
#include "classes.h"

using namespace std;
using namespace cv;
using namespace dnn;

std::vector<cv::Mat> preProcessAndInfer(cv::dnn::Net yolo, cv::Mat frame)
{
    Mat inImg;
    cvtColor(frame, inImg, COLOR_BGR2RGB);

    resize(inImg,inImg,Size(448,448));

    inImg.convertTo(inImg, CV_32F);
    inImg = inImg/255.0;

    Mat inNet = blobFromImage(inImg);

    yolo.setInput(inNet);
    Mat out = yolo.forward(); //1x5x7x7
    vector<Mat> outs; //1*(7x7x5)
    imagesFromBlob(out,outs);

    return outs;
}
