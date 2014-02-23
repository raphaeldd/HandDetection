#include "HighestLikelihood.h"

HighestLikelihood::HighestLikelihood()
{
}

void HighestLikelihood::armConnectDetection(Mat img, vector<RotatedRect> hand) {
    // Skin segmentation
    Mat skin = this->skinSegmentation(img);
    imwrite("Skin.png", skin);
    imwrite("Original.png", img);


    // Skeletonize
    cv::threshold(skin, skin, 127, 255, cv::THRESH_BINARY);
    cv::Mat skel(skin.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;

    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    bool done;
    do
    {
        cv::erode(skin, eroded, element);
        cv::dilate(eroded, temp, element); // temp = open(img)
        cv::subtract(skin, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(skin);

        done = (cv::countNonZero(skin) == 0);
    } while (!done);

    imshow("Skin seg. + Skeleton", skel);
    imwrite("SkeletonizedSkin.png", skel);



    // Line detection
    vector<Vec4i> lines;
    HoughLinesP(skel, lines, 1, CV_PI/180, 5, 20, 15);
    Mat dst = img.clone();
    for (int i = 0; i < lines.size(); i++ ) {
        Vec4i l = lines[i];
        line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
    }

    imshow("Arms", dst);
    imwrite("Arms.png", dst);
}

Mat HighestLikelihood::skinSegmentation(Mat In) {
    // Face detection
    CascadeClassifier face_cascade;
    vector<Rect> faces;
    Mat skin, faceCut;
    if ( !face_cascade.load("haarcascade_frontalface_alt.xml") ) {
        cout << "Error loading haarcascade xml file." << endl;
    }
    face_cascade.detectMultiScale(In, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    if ( faces.size() > 0 ) {
        // Image to HSV
        cvtColor(In, skin, CV_BGR2HSV);

        // Get face region
        GaussianBlur(skin(faces.at(0)), faceCut, Size(7, 7), 1, 1);


        // Get face color in the Hue color space
        vector<Mat> channel;
        split( faceCut, channel );
        int centerH = 0;
        for ( int i = 10; i < channel[0].rows-10; i++ ) {
            for ( int j = 10; j < channel[0].cols-10; j++ ) {
                centerH = ( centerH + channel[0].at<uchar>(i, j) ) / 2;
            }
        }

        // Segmentation of the skin in Input image
        inRange(skin, Scalar(0, 48, 80), Scalar(centerH+2, 255, 255), skin);
        int sizeMask = 2;
        Mat element = getStructuringElement( MORPH_ERODE, Size( 2*sizeMask+1, 2*sizeMask+1 ), Point( sizeMask, sizeMask ) );
        morphologyEx( skin, skin, 2, element);
    }
    return skin;
}
