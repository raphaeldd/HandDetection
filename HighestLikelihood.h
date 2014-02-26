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

        vector<int> getScore();

    private:
        vector<int> score;

        Mat skinSegmentation(Mat In);
};

#endif // HIGHESTLIKELIHOOD_H
