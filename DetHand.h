/*!
 * @brief Hand detection class
 *
 * @author Den Dooven Raphael
 *
 */

#ifndef DETHAND_H
#define DETHAND_H

#include "SimpleOpt.h"

#include "Intersector.h"
#include "Mixture.h"
#include "Scene.h"
#include "PersonDetection.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/ml/ml.hpp"
#include <opencv2/gpu/gpu.hpp>

#include <time.h>

#define PI 3.141592654

/*
 *  DET_TYPE 0: Detection per rotation
 *  DET_TYPE 1: Detection for all rotations at ones
 *  DET_TYPE 2: Detections for a few rotations at ones
 **/

#define DET_TYPE 2

using namespace cv;

class DetHand
{
    public:
        DetHand(String modelHand, double threshold);
        void runDetection(Mat image , int frameNumber);
        void drawResult(Mat& img, Rect& box, float angle, const Scalar& color);
        vector<Mat> getCutouts();
        vector<pair<Rect, int> > getRect();
        int getSize();


    private:
        vector<UniqueDetections> FinalUpperHandDetections;
        vector<UniqueDetections> FinalHandDetections;
        vector<UniqueDetections> FinalDetections;
        Mixture mixtureHand;
        String modelHand;

        vector<Mat> detections;
        vector<pair<Rect, int> > pos;

        double TH;

        void rotate(cv::Mat& src, double angle, cv::Mat& dst);
        void correction(Rect box, int angle, int correction);
};

#endif // DETHAND_H
