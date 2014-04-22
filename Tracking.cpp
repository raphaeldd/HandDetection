#include "Tracking.h"

Tracking::Tracking() {
    // 5 Kalman filters
    //      1. Head location
    //      2. Lefty location
    //      3. Lefty rotation
    //      4. Righty location
    //      5. Righty Rotation

    // Reset all variables
    this->face = Rect(0, 0, 0, 0);
    this->lefty = RotatedRect(Point(0, 0), Size(0, 0), 0);
    this->righty = RotatedRect(Point(0, 0), Size(0, 0), 0);
    this->predictedLefty = RotatedRect(Point(0, 0), Size(0, 0), 0);
    this->predictedRighty = RotatedRect(Point(0, 0), Size(0, 0), 0);
    f = l = r = false;

    // Face initialisation
    this->KFFace = new KalmanFilter(4, 2, 0);
    // this->KFFace->transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    // TEST: Slowing predition down after time
    this->KFFace->transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,.9,0,  0,0,0,.9);
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


    // TEST
    this->KFLefty = new KalmanFilter(8, 4, 0);
    this->KFLefty->transitionMatrix = *(Mat_<float>(8, 8) << \
                                        1, 0, 1, 0, 0, 0, 0, 0, \
                                        0, 1, 0, 1, 0, 0, 0, 0, \
                                        0, 0,.9, 0, 0, 0, 0, 0, \
                                        0, 0, 0,.9, 0, 0, 0, 0, \
                                        0, 0, 0, 0, 1, 0, 1, 0, \
                                        0, 0, 0, 0, 0, 1, 0, 1, \
                                        0, 0, 0, 0, 0, 0, 1, 0, \
                                        0, 0, 0, 0, 0, 0, 0, 1);
    this->KFLefty->measurementMatrix = *(Mat_<float>(4, 8) << \
                                         1, 0, 0, 0, \
                                         0, 1, 0, 0, \
                                         0, 0, 0, 0, \
                                         0, 0, 0, 0, \
                                         0, 0, 1, 0, \
                                         0, 0, 0, 1, \
                                         0, 0, 0, 0, \
                                         0, 0, 0, 0 );

    this->KFRighty = new KalmanFilter(8, 4, 0);
    this->KFRighty->transitionMatrix = *(Mat_<float>(8, 8) << \
                                        1, 0, 1, 0, 0, 0, 0, 0, \
                                        0, 1, 0, 1, 0, 0, 0, 0, \
                                        0, 0,.9, 0, 0, 0, 0, 0, \
                                        0, 0, 0,.9, 0, 0, 0, 0, \
                                        0, 0, 0, 0, 1, 0, 1, 0, \
                                        0, 0, 0, 0, 0, 1, 0, 1, \
                                        0, 0, 0, 0, 0, 0, 1, 0, \
                                        0, 0, 0, 0, 0, 0, 0, 1);
    this->KFRighty->measurementMatrix = *(Mat_<float>(4, 8) << \
                                         1, 0, 0, 0, \
                                         0, 1, 0, 0, \
                                         0, 0, 0, 0, \
                                         0, 0, 0, 0, \
                                         0, 0, 1, 0, \
                                         0, 0, 0, 1, \
                                         0, 0, 0, 0, \
                                         0, 0, 0, 0 );
}


void Tracking::track(Rect face, RotatedRect foundLefty, RotatedRect foundRighty) {
    // Get new face absolute position if posible
    // Get to 2 hands if posible ( if not give empty (0, 0) "hands")

    // Initialisize when first face is found else give NaN predictions and exit function
    // Face found
    if ( f == false && face.area() > 0 ) {
        cout << "-------> First face found start Kalman filter for face." << endl;
        f = true; // Set Kalman tracking active
        // init first location
        KFFace->statePre.at<float>(0) = this->rectCenter(face).x; // Position x
        KFFace->statePre.at<float>(1) = this->rectCenter(face).y; // Position y
        KFFace->statePre.at<float>(2) = 0; // Velocity x
        KFFace->statePre.at<float>(3) = 0; // Velocity y

        setIdentity(KFFace->measurementMatrix);
        // Porcess Noise = Q if large => Tracks large changes in the data more closely
        setIdentity(KFFace->processNoiseCov, Scalar::all(1e-1));
        // Measurment Noise = R if high => Kalman considers measurments as not very accurate
        //if smaller R it will follow the measurements more closely.
        setIdentity(KFFace->measurementNoiseCov, Scalar::all(1e-10));
        setIdentity(KFFace->errorCovPost, Scalar::all(1));
        this->face = face;

        // Get Kalman filter going
        Mat_<float> faceMeaserement(2, 1);
        faceMeaserement(0) = this->rectCenter(face).x;
        faceMeaserement(1) = this->rectCenter(face).y;
        for ( int i = 0; i < 5; i++) {
            KFFace->predict();
            KFFace->statePre.copyTo(KFFace->statePost);
            KFFace->errorCovPre.copyTo(KFFace->errorCovPost);
            KFFace->correct(faceMeaserement);
        }
    }
    if ( f == true ) {
        // predict face location
        cout << "-------> Face predication." << endl;
        Mat facePrediction = KFFace->predict();
        KFFace->statePre.copyTo(KFFace->statePost);
        KFFace->errorCovPre.copyTo(KFFace->errorCovPost);
        this->face.x = facePrediction.at<float>(0) - this->face.width/2;
        this->face.y = facePrediction.at<float>(1) - this->face.height/2;

        if ( face.area() > 0 ) {
            // Correct
            cout << "-------> Face correction." << endl;
            Mat_<float> faceMeaserement(2, 1);
            faceMeaserement(0) = this->rectCenter(face).x;
            faceMeaserement(1) = this->rectCenter(face).y;

            KFFace->correct(faceMeaserement);
            this->face.size() = face.size();
        }

        // Left hand tracking
        if ( l == false && foundLefty.size.area() > 0 ) {
            // init left hand
            cout << "-------> First hand found start Kalman filter for face." << endl;
            l = true; // Set Kalman tracking active
            // init first location
            KFLefty->statePre.at<float>(0) = foundLefty.center.x;                   // Posistion left hand on the x-axis
            KFLefty->statePre.at<float>(1) = foundLefty.center.y;                   // Posistion left hand on the y-axis
            KFLefty->statePre.at<float>(2) = 0;                                     // Velocity of the left hand in the x direction
            KFLefty->statePre.at<float>(3) = 0;                                     // Velocity of the left hand in the y direction
            pair<float, float> cartLefty = this->polarToCartesian(foundLefty.angle);
            KFLefty->statePre.at<float>(4) = cartLefty.first;                       // Angle projection on the x-axis
            KFLefty->statePre.at<float>(5) = cartLefty.second;                      // Angle projection on the y-axis
            KFLefty->statePre.at<float>(6) = 0;                                     // Angle projection on the x-axis
            KFLefty->statePre.at<float>(7) = 0;                                     // Angle projection on the y-axis

            setIdentity(KFLefty->measurementMatrix);
            // Porcess Noise = Q if large => Tracks large changes in the data more closely
            setIdentity(KFLefty->processNoiseCov, Scalar::all(1e-1));
            // Measurment Noise = R if high => Kalman considers measurments as not very accurate
            //if smaller R it will follow the measurements more closely.
            setIdentity(KFLefty->measurementNoiseCov, Scalar::all(1e-10));
            setIdentity(KFLefty->errorCovPost, Scalar::all(1));

            // Get Kalman filter going
            Mat_<float> LeftyMeaserement(4, 1);
            LeftyMeaserement(0) = foundLefty.center.x;
            LeftyMeaserement(1) = foundLefty.center.y;
            LeftyMeaserement(2) = cartLefty.first;
            LeftyMeaserement(3) = cartLefty.second;
            for ( int i = 0; i < 5; i++) {
                KFLefty->predict();
                KFLefty->statePre.copyTo(KFLefty->statePost);
                KFLefty->errorCovPre.copyTo(KFLefty->errorCovPost);
                KFLefty->correct(LeftyMeaserement);
            }
            this->lefty = foundLefty;
        }
        if ( l == true ) {
            // predict Lefty location
            cout << "-------> Lefty predication." << endl;
            Mat LeftyPrediction = KFLefty->predict();
            KFLefty->statePre.copyTo(KFLefty->statePost);
            KFLefty->errorCovPre.copyTo(KFLefty->errorCovPost);

            this->lefty.center.x = LeftyPrediction.at<float>(0);
            this->lefty.center.y = LeftyPrediction.at<float>(1);
            this->lefty.angle = this->cartesianToPolar(LeftyPrediction.at<float>(3), LeftyPrediction.at<float>(2));

            if ( foundLefty.size.area() > 0 ) {
                // Correct
                cout << "-------> Lefty correction." << endl;
                Mat_<float> LeftyMeaserement(4, 1);
                LeftyMeaserement(0) = foundLefty.center.x;
                LeftyMeaserement(1) = foundLefty.center.y;
                pair<float, float> cartLefty = this->polarToCartesian(foundLefty.angle);
                LeftyMeaserement(2) = cartLefty.first;
                LeftyMeaserement(3) = cartLefty.second;

                KFLefty->correct(LeftyMeaserement);
                this->lefty.size = foundLefty.size;
                this->predictedLefty.size = foundLefty.size;
            }

            cout << "-------> Lefty predication next frame." << endl;
            LeftyPrediction = KFLefty->predict();
            KFLefty->statePre.copyTo(KFLefty->statePost);
            KFLefty->errorCovPre.copyTo(KFLefty->errorCovPost);


            this->predictedLefty.center.x = LeftyPrediction.at<float>(0);
            this->predictedLefty.center.y = LeftyPrediction.at<float>(1);
            this->predictedLefty.angle = this->cartesianToPolar(LeftyPrediction.at<float>(3), LeftyPrediction.at<float>(2));
        }


        // Right hand tracking
        if ( r == false && foundRighty.size.area() > 0 ) {
            // init left hand
            cout << "-------> First hand found start Kalman filter for face." << endl;
            r = true; // Set Kalman tracking active
            // init first location
            KFRighty->statePre.at<float>(0) = foundRighty.center.x;                   // Posistion left hand on the x-axis
            KFRighty->statePre.at<float>(1) = foundRighty.center.y;                   // Posistion left hand on the y-axis
            KFRighty->statePre.at<float>(2) = 0;                                     // Velocity of the left hand in the x direction
            KFRighty->statePre.at<float>(3) = 0;                                     // Velocity of the left hand in the y direction
            pair<float, float> cartRighty = this->polarToCartesian(foundRighty.angle);
            KFRighty->statePre.at<float>(4) = cartRighty.first;                       // Angle projection on the x-axis
            KFRighty->statePre.at<float>(5) = cartRighty.second;                      // Angle projection on the y-axis
            KFRighty->statePre.at<float>(6) = 0;                                     // Angle projection on the x-axis
            KFRighty->statePre.at<float>(7) = 0;                                     // Angle projection on the y-axis

            setIdentity(KFRighty->measurementMatrix);
            // Porcess Noise = Q if large => Tracks large changes in the data more closely
            setIdentity(KFRighty->processNoiseCov, Scalar::all(1e-1));
            // Measurment Noise = R if high => Kalman considers measurments as not very accurate
            //if smaller R it will follow the measurements more closely.
            setIdentity(KFRighty->measurementNoiseCov, Scalar::all(1e-10));
            setIdentity(KFRighty->errorCovPost, Scalar::all(1));

            // Get Kalman filter going
            Mat_<float> RightyMeaserement(4, 1);
            RightyMeaserement(0) = foundRighty.center.x;
            RightyMeaserement(1) = foundRighty.center.y;
            RightyMeaserement(2) = cartRighty.first;
            RightyMeaserement(3) = cartRighty.second;
            for ( int i = 0; i < 5; i++) {
                KFRighty->predict();
                KFRighty->statePre.copyTo(KFLefty->statePost);
                KFRighty->errorCovPre.copyTo(KFLefty->errorCovPost);
                KFRighty->correct(RightyMeaserement);
            }
            this->righty = foundRighty;
        }
        if ( r == true ) {
            // predict Righty location
            cout << "-------> Righty predication." << endl;
            Mat RightyPrediction = KFRighty->predict();
            KFRighty->statePre.copyTo(KFRighty->statePost);
            KFRighty->errorCovPre.copyTo(KFRighty->errorCovPost);

            this->righty.center.x = RightyPrediction.at<float>(0);
            this->righty.center.y = RightyPrediction.at<float>(1);
            this->righty.angle = this->cartesianToPolar(RightyPrediction.at<float>(3), RightyPrediction.at<float>(2));

            if ( foundRighty.size.area() > 0 ) {
                // Correct
                cout << "-------> Righty correction." << endl;
                Mat_<float> RightyMeaserement(4, 1);
                RightyMeaserement(0) = foundRighty.center.x;
                RightyMeaserement(1) = foundRighty.center.y;
                pair<float, float> cartRighty = this->polarToCartesian(foundRighty.angle);
                RightyMeaserement(2) = cartRighty.first;
                RightyMeaserement(3) = cartRighty.second;

                KFRighty->correct(RightyMeaserement);
                this->righty.size = foundRighty.size;
                this->predictedRighty.size = foundRighty.size;
            }

            cout << "-------> Righty predication next frame." << endl;
            RightyPrediction = KFRighty->predict();
            KFRighty->statePre.copyTo(KFRighty->statePost);
            KFRighty->errorCovPre.copyTo(KFRighty->errorCovPost);


            this->predictedRighty.center.x = RightyPrediction.at<float>(0);
            this->predictedRighty.center.y = RightyPrediction.at<float>(1);
            this->predictedRighty.angle = this->cartesianToPolar(RightyPrediction.at<float>(3), RightyPrediction.at<float>(2));
        }

    }
}

Rect Tracking::getFace() {
    return this->face;
}
RotatedRect Tracking::getLefty() {
    return this->lefty;
}

RotatedRect Tracking::getRighty() {
    return this->righty;
}

RotatedRect Tracking::getPredictedLefty() {
    return this->predictedLefty;
}

RotatedRect Tracking::getPredictedRighty() {
    return this->predictedRighty;
}

pair<float, float> Tracking::convertToVector(float degrees) {
    return pair<float, float>(sin( degrees * CV_PI/180 ), cos( degrees * CV_PI/180 ));
}

float Tracking::convertToDegrees(pair<float, float> u) {
    return atan2( u.first, u.second ) * 180/CV_PI;
}

Point Tracking::rectCenter(Rect box) {
    return Point(box.x + box.width/2, box.y + box.height/2);
}

pair<float, float> Tracking::polarToCartesian(float d) {
    return pair<float, float>( cos( d * CV_PI/180 ), sin( d * CV_PI/180 ) );
}

float Tracking::cartesianToPolar(float x, float y) {
    return atan2(x, y) * 180/CV_PI;
}
