#include "HighestLikelihood.h"

HighestLikelihood::HighestLikelihood()
{
}

void HighestLikelihood::armConnectDetection(Mat img, vector<RotatedRect> hand)
{
    // Skin segmentation
    Mat skin = this->skinSegmentation(img);
    // Find 4 biggest blobs
    Mat blobs = skin.clone();
    vector<vector<Point> > contours;
    findContours(blobs, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    vector<pair<double, int> > sizeContours;
    Mat bigSkin(skin.rows, skin.cols, skin.type(), Scalar(0, 0, 0));

    for(unsigned int i = 0; i < contours.size(); i++) {
        sizeContours.push_back(make_pair(contourArea(contours[i]), i));
    }

    sort(sizeContours.begin(), sizeContours.end());

    for(unsigned  int i = sizeContours.size() - 1; i > sizeContours.size() - 5; i--) {
        drawContours(bigSkin, contours, sizeContours[i].second, Scalar(255, 255, 255), CV_FILLED);
    }

    imshow("4 bigest skin areas", bigSkin);
    // Skeletonize
    cv::threshold(bigSkin, bigSkin, 127, 255, cv::THRESH_BINARY);
    cv::Mat skel(bigSkin.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    bool done;

    do {
        cv::erode(bigSkin, eroded, element);
        cv::dilate(eroded, temp, element); // temp = open(img)
        cv::subtract(bigSkin, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(bigSkin);
        done = (cv::countNonZero(bigSkin) == 0);
    } while(!done);

    imshow("Skin seg. + Skeleton", skel);
    // Line detection
    vector<Vec4i> lines;
    HoughLinesP(skel, lines, 1, CV_PI / 180, 5, 20, 15);
    Mat arms = img.clone();

    for(unsigned int i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(arms, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 128, 0), 1, CV_AA);
    }

    // Check if endpoints of detected lines are in founded hand detection areas
    this->score.assign(hand.size(), 0);

    //      Loop for each line point
    for(unsigned int l = 0; l < lines.size(); l++) {
        Vec4i line = lines[l];

        //      Loop for each detected hand
        for(unsigned int h = 0; h < hand.size(); h++) {
            Rect bb = hand[h].boundingRect();

            if(line[0] >= bb.tl().x && line[0] <= bb.br().x && line[1] >= bb.tl().y && line[1] <= bb.br().y) {
                this->score.at(h)++;
            }

            if(line[2] >= bb.tl().x && line[2] <= bb.br().x && line[3] >= bb.tl().y && line[3] <= bb.br().y) {
                this->score.at(h)++;
            }
        }
    }

    imshow("Arms", arms);
}

vector<int> HighestLikelihood::getScore()
{
    return this->score;
}





Mat HighestLikelihood::skinSegmentation(Mat In)
{
    // Face detection
    CascadeClassifier face_cascade;
    vector<Rect> faces;
    Mat skin, faceCut;

    if(!face_cascade.load("haarcascade_frontalface_alt.xml")) {
        cout << "Error loading haarcascade xml file." << endl;
    }

    face_cascade.detectMultiScale(In, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    if(faces.size() > 0) {
        // Image to HSV
        cvtColor(In, skin, CV_BGR2HSV);
        // Get face region
        GaussianBlur(skin(faces.at(0)), faceCut, Size(7, 7), 1, 1);
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
    }

    return skin;
}
