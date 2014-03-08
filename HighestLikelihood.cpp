#include "HighestLikelihood.h"

HighestLikelihood::HighestLikelihood()
{
}

void HighestLikelihood::run(Mat img, vector<RotatedRect> hands, vector<double> handScores, double TH, double eps) {
    vector<int> ScoreSkin;
    vector<int> ScoreArms;

    // Get skin segmentation based on face color
    Rect face = this->faceDetection(img);
    if (face.area() != 0) {
        Mat binairySkin = this->skinSegmentation(img, face);

        // Crude eliminate wrong hands
        hands = this->removeTobigHandBox(hands, handScores, face, eps);

        // Calculating score from features
        ScoreSkin = this->skinScore(hands, binairySkin);
        ScoreArms = this->armConnectDetection(binairySkin, hands);
    } else {
        ScoreSkin.assign(hands.size(), 0);
        ScoreArms.assign(hands.size(), 0);
    }
    double max = TH, secMax = TH;
    this->results.assign(2, RotatedRect());
    for ( unsigned int i = 0; i < hands.size(); i++ ) {

        // find 2 best scores
        double score = (double)ScoreSkin[i] + (double)ScoreArms[i] + handScores[i];

        if ( score >= max) {
            max = score;
            this->results[1] = this->results[0];
            this->results[0] = hands[i];
        }
        if ( score >= secMax && score < max) {
            secMax = score;
            this->results[1] = hands[i];
        }

    }

}

vector<RotatedRect> HighestLikelihood::getResults() {
    return this->results;
}

Rect HighestLikelihood::getFace(){
    return this->face;
}


vector<int> HighestLikelihood::armConnectDetection(Mat skin, vector<RotatedRect> hand)
{
    // Skeletonize
    cv::threshold(skin, skin, 127, 255, cv::THRESH_BINARY);
    cv::Mat skel(skin.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    bool done;

    do {
        erode(skin, eroded, element);
        dilate(eroded, temp, element); // temp = open(img)
        subtract(skin, temp, temp);
        bitwise_or(skel, temp, skel);
        eroded.copyTo(skin);
        done = (countNonZero(skin) == 0);
    } while(!done);

    // Line detection
    vector<Vec4i> lines;
    HoughLinesP(skel, lines, 1, CV_PI / 180, 5, 20, 15);

    // Check if endpoints of detected lines are in founded hand detection areas
    vector<int> score(hand.size(), 0);

    //      Loop for each line point
    for(unsigned int l = 0; l < lines.size(); l++) {
        Vec4i line = lines[l];

        //      Loop for each detected hand
        for(unsigned int h = 0; h < hand.size(); h++) {
            Rect bb = hand[h].boundingRect();

            if(line[0] >= bb.tl().x && line[0] <= bb.br().x && line[1] >= bb.tl().y && line[1] <= bb.br().y) {
                score.at(h)++;
            }

            if(line[2] >= bb.tl().x && line[2] <= bb.br().x && line[3] >= bb.tl().y && line[3] <= bb.br().y) {
                score.at(h)++;
            }
        }
    }
    return score;
}

vector<int> HighestLikelihood::skinScore(vector<RotatedRect> hand, Mat skin) {
    vector<int> score(hand.size() , 0);

    for (int i = 0; i < hand.size(); i++) {
        Mat M, rotated, cropped;

        float angle = hand.at(i).angle;
        Size hand_size = hand.at(i).size;

        if ( hand.at(i).angle < -45 ) {
            angle += 90;
            swap(hand_size.width, hand_size.height);
        }

        M = getRotationMatrix2D(hand.at(i).center, angle, 1.0);
        warpAffine(skin, rotated, M, skin.size(), INTER_CUBIC);
        getRectSubPix(rotated, hand_size, hand.at(i).center, cropped);

        score[i] += (float)countNonZero(cropped)/(float)(cropped.rows * cropped.cols) * 5;
    }

    return score;
}

vector<RotatedRect> HighestLikelihood::removeTobigHandBox(vector<RotatedRect>& hand, vector<double>& score, Rect face, double eps) {
    for ( int i = 0; i < hand.size(); i++ ) {
        if ( ( hand.at(i).size.height * hand.at(i).size.width ) > ( ( face.size().height * face.size().width ) * ( 1 + eps ) ) ) {
            hand.erase(hand.begin() + i);
            score.erase(score.begin() + i);
        }
    }
    return hand;
}


Mat HighestLikelihood::skinSegmentation(Mat In, Rect face)
{
    Mat skin, faceCut;
    // Image to HSV
    cvtColor(In, skin, CV_BGR2HSV);
    // Get face region
    GaussianBlur(skin(face), faceCut, Size(7, 7), 1, 1);
    // Get face color in the Hue color space
    vector<Mat> channel;
    split(faceCut, channel);
    int centerH = 0;

    for(int i = 10; i < channel[0].rows - 10; i++) {
        for(int j = 10; j < channel[0].cols - 10; j++) {
            centerH = (centerH + channel[0].at<uchar>(i, j)) / 2;
        }
    }

    // Segmentation of the skin in Input image
    inRange(skin, Scalar(0, 70, 60), Scalar(centerH * 1.2, 255, 255), skin);
    Mat element = getStructuringElement(MORPH_ERODE, Size(7, 7), Point(3, 3));
    morphologyEx(skin, skin, 2, element);
    element = getStructuringElement(MORPH_DILATE, Size(7, 7), Point(3, 3));
    dilate(skin, skin, element);

    Mat blobs = skin.clone();
    vector<vector<Point> > contours;
    findContours(blobs, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    Mat bigSkin(skin.rows, skin.cols, skin.type(), Scalar(0, 0, 0));

    for(unsigned  int i = 0; i < contours.size(); i++) {
        if ( contourArea( contours[i] ) >= face.area()/4 )
            drawContours(bigSkin, contours, i, Scalar(255, 255, 255), CV_FILLED);
    }

    return bigSkin.clone();
}

Rect HighestLikelihood::faceDetection(Mat In) {
    // Face detection
    CascadeClassifier face_cascade;
    vector<Rect> faces;

    if(!face_cascade.load("haarcascade_frontalface_alt.xml")) {
        cout << "Error loading haarcascade xml file." << endl;

        return Rect();
    }

    face_cascade.detectMultiScale(In, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    if ( faces.size() > 0) {
        this->face = faces[0];
        return faces[0];
    } else {
        cout << "No face found" << endl;
        return Rect();
    }
}

