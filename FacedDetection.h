#ifndef FACEDDETECTION_H
#define FACEDDETECTION_H

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/ml/ml.hpp"
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace std;

class FaceDetection
{
    public:
        FaceDetection(string faceModelFile);

        Rect detectFace(Mat In);

    private:
        CascadeClassifier face_cascade;

        Rect face;
};

#endif // FACEDDETECTION_H
