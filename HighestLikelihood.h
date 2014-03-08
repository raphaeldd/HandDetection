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
        /// @brief Constructor
        HighestLikelihood();

        /// @brief Runs the correct methodes for eliminating unjust detections
        /// @param img Image where the detected hands were detected
        /// @param hands Rotated bounding boxes of detected hands
        /// @param handScores Scores from the detector
        /// @param TH threshold hands
        /// @param eps used for elimination to big bounding boxes
        void run(Mat img, vector<RotatedRect> hands, vector<double> handScores, double TH = 0, double eps = 0.1);

        /// @brief Gives the result after the running the run function
        /// @return higest likely detected hands
        vector<RotatedRect> getResults();

        /// @brief Gives face rectangle
        /// @return Rect with face location and it's size
        Rect getFace();

    private:
        vector<RotatedRect> results;
        Rect face;

        Mat skinSegmentation(Mat In, Rect face);
        Rect faceDetection(Mat In);

        vector<int> armConnectDetection(Mat binairy, vector<RotatedRect> hand);
        vector<int> skinScore(vector<RotatedRect> hand, Mat skin);
        vector<RotatedRect> removeTobigHandBox(vector<RotatedRect> &hand, vector<double> &score, Rect face, double eps);
};

#endif // HIGHESTLIKELIHOOD_H
