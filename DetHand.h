/*!
 * @class DetHand
 * Hand detection class. Detects hand on images.
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
using namespace std;


class DetHand
{
    public:
        /// @brief Constructor from classe DetHand.
        /// Preps the detector with the right hand model.
        /// @param ModelHand Filename of model
        /// @param threshold Threshold value of detector
        DetHand(String modelHand, String contextModel, double threshold);

        /// @brief Apply the detector on a image
        /// @param image The image for the detector
        /// @param framenumber Frame number of the video (not really importend)
        void runDetection(Mat image , int frameNumber);

        /// @brief Draws the found regions on a image
        /// Draws a rotated rectangle with rescpect to the center
        /// @param img Detination image
        /// @param box Rectangle information
        /// @param angle Rectangles rotation with rescpect to the center
        /// @param color color op the rectangle
        void drawResult(Mat& img, RotatedRect& box, const Scalar& color);

        /// @brief Get all the detection cutouts of the last detection
        /// @return All cutouts of type Mat put in a vector
        vector<Mat> getCutouts();

        /// @brief Get all the detection regions of the last detection with the respected rotation
        /// @return All regions of type Rect and rotation put in a vector
        vector<RotatedRect> getRect();

        /// @brief Get score of detections found
        /// @return Score of detections found
        vector<double> getScore();

        /// @brief Get number of detections found
        /// @return Number of detections found of type integer
        int getSize();


    private:
        vector<UniqueDetections> FinalUpperHandDetections;
        vector<UniqueDetections> FinalHandDetections;
        vector<UniqueDetections> FinalDetections;
        vector<double> score;
        Mixture mixtureHand, mixtureHandContext;
        String modelHand;

        vector<Mat> detections;
        vector<RotatedRect> pos;

        double TH;

        void rotate(cv::Mat& src, double angle, cv::Mat& dst);
        RotatedRect correction(Rect box, int angle, int correction, Point center, int orientCorr = 0);
        void similarRects(vector<RotatedRect>& rects);
};

#endif // DETHAND_H
