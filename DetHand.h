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

using namespace cv;
using namespace std;

#define ROWS 1
#define COLS 1
#define ROTDEGREES 10


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
        /// @param face Give location face for skin segmentation
        void runDetection(Mat image , int frameNumber, Rect face);

        /// @brief Draws the found regions on a image
        /// Draws a rotated rectangle with rescpect to the center
        /// @param img Detination image
        /// @param box Rectangle information
        /// @param angle Rectangles rotation with rescpect to the center
        /// @param color color op the rectangle
        void drawResult(Mat& img, RotatedRect& box, const Scalar& color, int thickness = 1);

        /// @brief Get all the detection cutouts of the last detection
        /// @return All cutouts of type Mat put in a vector
        vector<Mat> getCutouts();

        /// @brief Get the detection regions of the last detection with the respected rotation of the hand detector
        /// @return All regions of type Rect and rotation put in a vector
        vector<RotatedRect> getLocationHand();

        /// @brief Get score of hand detections found
        /// @return Score of detections found
        vector<double> getScoreHand();

        /// @brief Get the detection regions of the last detection with the respected rotation of the context detector
        /// @return All regions of type Rect and rotation put in a vector
        vector<RotatedRect> getLocationContext();

        /// @brief Get score of context detections found
        /// @return Score of detections found
        vector<double> getScoreContext();

        /// @brief Get the detection regions of the last detection with the respected rotation of the arm hand detector
        /// @return All regions of type Rect and rotation put in a vector
        vector<RotatedRect> getLocationArm();

        /// @brief Get score of arm detections found
        /// @return Score of detections found
        vector<double> getScoreArm();

        /// @brief Get Skin segmentation
        /// @return skin
        Mat getSkin();

        /// @brief Convert results in relative coordinates
        /// @param list converion list
        /// @param refPoint referents point 0, 0
        vector<RotatedRect> absToRel(vector<RotatedRect> list, Point refPoint);

        /// @brief Convert results in absolute coordinates
        /// @param list converion list
        /// @param refPoint referents point 0, 0
        vector<RotatedRect> relToAbs(vector<RotatedRect> list, Point refPoint);


    private:
        vector<UniqueDetections> FinalUpperHandDetections;
        vector<UniqueDetections> FinalHandDetections;
        vector<UniqueDetections> FinalDetections;

        Mixture mixtureHand, mixtureHandContext;
        String modelHand;

        vector<RotatedRect> posHand;
        vector<RotatedRect> posContext;
        vector<RotatedRect> posArm;

        vector<double> scoreHand;
        vector<double> scoreContext;
        vector<double> scoreArm;

        Mat skin;

        double TH;

        Mat skinSegmentation(Mat img);
        Mat skeletonisation( Mat src );

        void rotate(Mat& src, double angle, Mat& dst);
        RotatedRect correction(Rect box, int angle, int correction, Point center, int orientCorr = 0);

        double length(Vec4i p);
        Point centerLine(Vec4i p);
        double angleLine(Vec4i p);


};

#endif // DETHAND_H
