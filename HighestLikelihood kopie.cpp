#include "HighestLikelihood.h"

HighestLikelihood::HighestLikelihood()
{
}

void HighestLikelihood::armConnectDetection(Mat img, vector<RotatedRect> hand)
{
    // Skin segmentation
    Mat skin = this->skinSegmentation(img);
    imshow("skin segmentation", skin);
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

void HighestLikelihood::skinScore(vector<RotatedRect> hand)
{
    for(int i = 0; i < hand.size(); i++) {
        Mat M, rotated, cropped;
        float angle = hand.at(i).angle;
        Size hand_size = hand.at(i).size;

        if(hand.at(i).angle < -45) {
            angle += 90;
            swap(hand_size.width, hand_size.height);
        }

        M = getRotationMatrix2D(hand.at(i).center, angle, 1.0);
        warpAffine(this->binairySkin, rotated, M, this->binairySkin.size(), INTER_CUBIC);
        getRectSubPix(rotated, hand_size, hand.at(i).center, cropped);
        this->score.at(i) += (float)countNonZero(cropped) / (float)(cropped.rows * cropped.cols) * 5;
    }
}

vector<int> HighestLikelihood::getScore()
{
    return this->score;
}

vector<RotatedRect> HighestLikelihood::removeTobigHandBox(Mat In, vector<RotatedRect> hand, double eps)
{
    if(face.area() == 0)
        faceDetection(In);

    cout << "   Start elimination process with " << hand.size() << endl;

    for(int i = 0; i < hand.size(); i++) {
        if((hand.at(i).size.height * hand.at(i).size.width) > ((this->face.size().height * this->face.size().width) * (1 + eps))) {
            hand.erase(hand.begin() + i);
        }
    }

    cout << "   Ended elimination process with " << hand.size() << endl << endl;
    return hand;
}


Mat HighestLikelihood::skinSegmentation(Mat In)
{
    Mat skin, faceCut;

    if(face.area() == 0)
        faceDetection(In);

    // Image to HSV
    cvtColor(In, skin, CV_BGR2HSV);
    // Get face region
    GaussianBlur(skin(this->face), faceCut, Size(7, 7), 1, 1);
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
        if(contourArea(contours[i]) >= face.area() / 4)
            drawContours(bigSkin, contours, i, Scalar(255, 255, 255), CV_FILLED);
    }

    this->binairySkin = bigSkin.clone();
    return bigSkin;
}

void HighestLikelihood::faceDetection(Mat In)
{
    // Face detection
    CascadeClassifier face_cascade;
    vector<Rect> faces;

    if(!face_cascade.load("haarcascade_frontalface_alt.xml")) {
        cout << "Error loading haarcascade xml file." << endl;
        return;
    }

    face_cascade.detectMultiScale(In, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    if(faces.size() > 0) {
        this->face = faces[0];
        Mat faceImg = In.clone();
        rectangle(faceImg, this->face, Scalar(250, 14, 140));
        imshow("Face", faceImg);
    } else {
        cout << "No face found" << endl;
        return;
    }
}

void HighestLikelihood::rotate(cv::Mat& src, double angle, cv::Mat& dst)
{
    cv::Point2f pt(src.cols / 2., src.rows / 2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
}
