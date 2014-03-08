#ifndef HIGHESTLIKELIHOOD_H
#define HIGHESTLIKELIHOOD_H

#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/ml/ml.hpp"
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace std;

class HighestLikelihood
{
    public:
        HighestLikelihood();

        void armConnectDetection(Mat img, vector<RotatedRect> hand);
        void skinScore(vector<RotatedRect> hand);
        vector<RotatedRect> removeTobigHandBox(Mat In, vector<RotatedRect> hand, double eps);

        vector<int> getScore();

    private:
        vector<int> score;
        Rect face;
        Mat binairySkin;

        Mat skinSegmentation(Mat In);
        void faceDetection(Mat In);
        void rotate(cv::Mat& src, double angle, cv::Mat& dst);
};

#endif // HIGHESTLIKELIHOOD_H
