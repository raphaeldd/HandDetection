#ifndef HIGHESTLIKELIHOOD_H
#define HIGHESTLIKELIHOOD_H

#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/ml/ml.hpp"
#include <opencv2/gpu/gpu.hpp>

#include "DetHand.h"
#include "Cluster.h"

using namespace cv;
using namespace std;

class HighestLikelihood
{
    public:
        /// @brief Constructor
        HighestLikelihood();

        /// @brief Runs the correct methodes for eliminating unjust detections
        /// @param img Image where the detected hands were detected
        /// @param hands Rotated bounding boxes of detected hands
        /// @param handScores Scores from the detector
        /// @param TH threshold hands
        /// @param eps used for elimination to big bounding boxes
        void run(DetHand* hand, Rect face, Mat img, RotatedRect lefty, RotatedRect righty);

        /// @brief Gives the result after the running the run function
        /// @return higest likely detected hands
        vector<RotatedRect> getResults();

        RotatedRect getLefty();
        RotatedRect getRighty();

    private:
        vector<RotatedRect> results;
        RotatedRect lefty;
        double leftyScore;
        RotatedRect righty;
        double rightyScore;

        void clustering(vector<RotatedRect> &hand, vector<double> &score, double eps);
        void skinEliminator(vector<RotatedRect> &hand, vector<double> &score, double TH, Mat skin);
        void removeTobigHandBox(vector<RotatedRect>& hand, vector<double>& score, Rect face, double eps);
        void lowerScoreFaceHand(vector<RotatedRect>& hand, vector<double>& score, Rect face, double eps, int lower);
        void closestPoint(vector<RotatedRect>& hand, vector<double>& score, Point predictedPoint);
        void findRighty(vector<RotatedRect>& resultsRighty, vector<double>& scoreRighty, RotatedRect predictedPoint, double eps);
        void findLefty(vector<RotatedRect>& resultsLefty, vector<double>& scoreLefty, RotatedRect predictedPoint, double eps);

        RotatedRect toRelative(RotatedRect hand, Rect face);

        float dstCalc( Point pt1, Point pt2 );
};

#endif // HIGHESTLIKELIHOOD_H
