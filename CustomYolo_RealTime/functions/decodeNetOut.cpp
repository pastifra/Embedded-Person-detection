#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "classes.h"

using namespace std;
using namespace cv;
using namespace dnn;

std::vector<cv::Rect> decodeNetOut(cv::Mat output)
{
    typedef Vec<float, 5> Vec5f;
    int cell_size = 64;

    double scale_w = 448.0/1280.0;
    double scale_h = 448.0/720.0;
    vector<Rect> propBoxes;
    for(int i = 0; i < 7; i++)
    {
        for(int j = 0; j < 7; j++)
        {
            if(output.at<Vec5f>(i,j)[0] > 0.8)
            {
                float x = output.at<Vec5f>(i,j)[1];
                float y = output.at<Vec5f>(i,j)[2];
                float w_cell = output.at<Vec5f>(i,j)[3];
                float h_cell = output.at<Vec5f>(i,j)[4];
                double ox = x*cell_size + cell_size*j;
                double oy = y*cell_size + cell_size*i;
                double w = w_cell*1280;
                double h = h_cell*720;
                double lx = (ox/scale_w) - w/2;
                double ly = (oy/scale_h) - h/2;
                propBoxes.push_back(Rect(static_cast<int>(lx), static_cast<int>(ly), static_cast<int>(w), static_cast<int>(h)));
            }
        }
    }
    return propBoxes;
}
