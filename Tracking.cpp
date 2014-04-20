#include "Tracking.h"

Tracking::Tracking(RotatedRect lefty, RotatedRect righty) {
    // 5 Kalman filters
    //      1. Head location
    //      2. Lefty location
    //      3. Lefty rotation
    //      4. Righty location
    //      5. Righty Rotation

    this->face = Rect(0, 0, 0, 0);
    this->lefty = RotatedRect(Point(0, 0), Size(0, 0), 0);
    this->righty = RotatedRect(Point(0, 0), Size(0, 0), 0);
    this->predictedLefty = RotatedRect(Point(0, 0), Size(0, 0), 0);
    this->predictedRighty = RotatedRect(Point(0, 0), Size(0, 0), 0);

    // Face initialisation
    this->KFFace = new KalmanFilter(4, 2, 0);
    this->KFFace->transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    this->KFFace->measurementMatrix = *(Mat_<float>(2, 4) << 1,0, 1,0, 0,1, 0,1 );

    // Lefty location initialisation
    this->KFLeftyLoc = new KalmanFilter(4, 2, 0);
    this->KFLeftyLoc->transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    this->KFLeftyLoc->measurementMatrix = *(Mat_<float>(2, 4) << 1,0, 1,0, 0,1, 0,1 );

    // Righty location initialisation
    this->KFRightyLoc = new KalmanFilter(4, 2, 0);
    this->KFRightyLoc->transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    this->KFRightyLoc->measurementMatrix = *(Mat_<float>(2, 4) << 1,0, 1,0, 0,1, 0,1 );


    // Calculating Kalman filter on angles:
    //      1. Convert degrees to unit vector
    //          - x = sin( d * PI_CV/180 )
    //          - y = cos( d * PI_CV/180 )
    //      2. Put in Kalman filter
    //      3. Calculate Kalman prediction
    //      4. Convert back to degrees
    //          - d = atan2(y, x) * 180/PI_CV
    // Lefty rotation initialisation
    this->KFLefyRot = new KalmanFilter(4, 2, 0);
    this->KFLefyRot->transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    this->KFLefyRot->measurementMatrix = *(Mat_<float>(2, 4) << 1,0, 1,0, 0,1, 0,1 );

    // Righty rotation initialisation
    this->KFRightyRot = new KalmanFilter(4, 2, 0);
    this->KFRightyRot->transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    this->KFRightyRot->measurementMatrix = *(Mat_<float>(2, 4) << 1,0, 1,0, 0,1, 0,1 );
}


void Tracking::track(Rect face, pair<RotatedRect, RotatedRect> hands) {
    // Get new face absolute position if posible
    // Get to 2 hands if posible ( if not give empty (0, 0) "hands")

    // Convert hand absolute location to relative location from face center
    // Find lefty and righty

    // Initialisize when first face is found else give NaN predictions and exit function
    // Initialisize when first lefty is found else give NaN predictions
    // Initialisize when first righty is found else give NaN predictions

    // if one or both arent found use prediction previus pase
    // else correct Kalman filter


    // Predict next pase
}

Rect Tracking::getFace() {
    return Rect();
}
RotatedRect Tracking::getLefty() {
    return RotatedRect();
}

RotatedRect Tracking::getRighty() {
    return RotatedRect();
}

RotatedRect Tracking::getPredictedLefty() {
    return RotatedRect();
}

RotatedRect Tracking::getPredictedRighty() {
    return RotatedRect();
}

pair<float, float> Tracking::convertToVector(float degrees) {
    return pair<float, float>(sin( degrees * CV_PI/180 ), cos( degrees * CV_PI/180 ));
}

float Tracking::convertToDegrees(pair<float, float> u) {
    return atan2( u.first, u.second ) * 180/CV_PI;
}
