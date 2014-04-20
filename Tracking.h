#ifndef TRACKING_H
#define TRACKING_H

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace cv;
using namespace std;

class Tracking
{
    public:
        /// @brief Constructor from classe Tracking.
        /// Creates 5 Kalman filters:
        ///     - Head location, used as referance
        ///     - Lefty hand (lefty) location
        ///     - Lefty hand rotation
        ///     - Right hand (righty) location
        ///     - Right hand rotation
        /// @param lefty first hand zone for left hand
        /// @param righty first hand zone for right hand
        Tracking(RotatedRect lefty, RotatedRect righty);

        void track(Rect face, pair<RotatedRect, RotatedRect> hands);

        Rect getFace();
        RotatedRect getLefty();
        RotatedRect getRighty();
        RotatedRect getPredictedLefty();
        RotatedRect getPredictedRighty();

    private:
        Rect face;
        RotatedRect lefty;
        RotatedRect righty;
        RotatedRect predictedLefty;
        RotatedRect predictedRighty;

        KalmanFilter* KFFace;
        KalmanFilter* KFLeftyLoc;
        KalmanFilter* KFRightyLoc;
        KalmanFilter* KFLefyRot;
        KalmanFilter* KFRightyRot;

        pair<float, float> convertToVector(float degrees);
        float convertToDegrees(pair<float, float> u);
};

#endif // TRACKING_H
