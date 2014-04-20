#ifndef CLUSTER_H
#define CLUSTER_H

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/ml/ml.hpp"
#include <opencv2/gpu/gpu.hpp>

using namespace cv;

class Cluster
{
    public:
        Cluster(double eps = 0.1, int sigma = 30);
        bool operator()(RotatedRect a, RotatedRect b);

    private:

        double eps;
        int sigma;
};

#endif // CLUSTER_H
