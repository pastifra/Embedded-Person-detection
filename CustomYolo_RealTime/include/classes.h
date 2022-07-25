#ifndef DEC
#define DEC

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

std::vector<cv::Rect> decodeNetOut(cv::Mat output);
std::vector<cv::Mat> preProcessAndInfer(cv::dnn::Net yolo, cv::Mat frame);

#endif
