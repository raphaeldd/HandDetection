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
        Tracking();

        void track(Rect face, RotatedRect foundLefty, RotatedRect foundRighty);

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
        // Boolean variables for first pase detection
        bool f, l, r;

        KalmanFilter* KFFace;
        KalmanFilter* KFLeftyLoc;
        KalmanFilter* KFRightyLoc;
        KalmanFilter* KFLefyRot;
        KalmanFilter* KFRightyRot;

        // Test
        KalmanFilter* KFLefty;
        KalmanFilter* KFRighty;

        pair<float, float> convertToVector(float degrees);
        float convertToDegrees(pair<float, float> u);
        Point rectCenter(Rect box);
        pair<float, float> polarToCartesian(float d);
        float cartesianToPolar(float x, float y);
};

#endif // TRACKING_H
