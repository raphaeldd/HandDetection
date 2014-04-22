#include "DetHand.h"

DetHand::DetHand(String modelHand, String contextModel, double threshold)
{
    // initialization: open mixtureHand + initialize patchwork class + calculate filter transform
    int res = Init(mixtureHand, modelHand);

    if(res == -1) {
        exit(1);
    }

    res = Init(mixtureHandContext, contextModel);

    if(res == -1) {
        exit(1);
    }

    this->TH = threshold;  // -0.5
}

void DetHand::runDetection(Mat image , int frameNumber, Rect face)
{
    int correction, orientCor;
    Mat blackImage;
    Rect roi;

    if(image.cols >= image.rows) {
        orientCor = 0;
        correction = (image.cols - image.rows) / 2;
        blackImage.create(image.cols, image.cols, image.type());
        roi = Rect(Point(0, (image.cols - image.rows) / 2), Size(image.cols, image.rows));
    } else {
        orientCor = 1;
        correction = (image.rows - image.cols) / 2;
        blackImage.create(image.rows, image.rows, image.type());
        roi = Rect(Point((image.rows - image.cols) / 2, 0), Size(image.cols, image.rows));
    }

    image.copyTo(blackImage(roi));
    int iteration = 0;
    this->posHand.clear();
    this->posContext.clear();
    this->posArm.clear();
    this->scoreHand.clear();
    this->scoreContext.clear();
    this->scoreArm.clear();

    while(iteration < 360) {
        Mat rotImage(blackImage.rows * ROWS, blackImage.cols * COLS, blackImage.type());
        Mat detImage;

        // ---------- Rotated images and merge them in one image ----------
        for(int r = iteration; r < (ROWS * COLS * ROTDEGREES) + iteration; r = r + ROTDEGREES) {
            Mat tmpImage;
            int y = floor((double)(r - iteration) / (ROTDEGREES * COLS)) ;
            int x = (r - iteration - y * (ROTDEGREES * ROWS)) / ROTDEGREES ;
            Rect roi(Point(x * blackImage.cols, y * blackImage.rows), Size(blackImage.rows, blackImage.cols));
            rotate(blackImage, r, tmpImage);
            tmpImage.copyTo(rotImage(roi));
        }

        // ---------- Hand detection ----------
        rotImage.copyTo(detImage);
        FinalDetections.clear();
        FinalHandDetections.clear();
        FinalUpperHandDetections.clear();
        PersonDetection((frameNumber * 1000) + iteration, mixtureHand, FinalDetections, FinalUpperHandDetections, FinalHandDetections, detImage, TH);

        for(unsigned int k = 0; k < FinalHandDetections.size(); k++) {
            int rot = floor((double)FinalHandDetections[k].detect.x / (double)blackImage.cols) * ROTDEGREES + iteration;
            Rect tmpRect(FinalHandDetections[k].detect.x % blackImage.cols, FinalHandDetections[k].detect.y, FinalHandDetections[k].detect.width, FinalHandDetections[k].detect.height);
            this->posHand.push_back(this->correction(tmpRect, rot, correction, Point(blackImage.cols / 2, blackImage.rows / 2), orientCor));
            this->scoreHand.push_back(FinalHandDetections.at(k).BestScore - TH);
        }

        // ---------- Context detection ----------
        rotImage.copyTo(detImage);
        FinalDetections.clear();
        FinalHandDetections.clear();
        FinalUpperHandDetections.clear();
        PersonDetection((frameNumber * 1000) + iteration, mixtureHandContext, FinalDetections, FinalUpperHandDetections, FinalHandDetections, detImage, TH);

        for(unsigned int k = 0; k < FinalHandDetections.size(); k++) {
            int rot = floor((double)FinalHandDetections[k].detect.x / (double)blackImage.cols) * ROTDEGREES + iteration;
            Rect tmpRect(FinalHandDetections[k].detect.x % blackImage.cols, FinalHandDetections[k].detect.y, FinalHandDetections[k].detect.width, FinalHandDetections[k].detect.height);
            this->posContext.push_back(this->correction(tmpRect, rot, correction, Point(blackImage.cols / 2, blackImage.rows / 2), orientCor));
            this->scoreContext.push_back(FinalHandDetections.at(k).BestScore - TH);
        }

        iteration = (ROWS * COLS * ROTDEGREES) + iteration;
    }

    // ---------- Arm hand detection ----------
    // If there is no face this detection will not be executed

    // Skin segmentation
    this->skin = this->skinSegmentation(image);
    //imshow("Skin", skin);
    if ( face.area() ) {
        // Sketelizing
        Mat skel = this->skeletonisation(skin.clone());
        imshow("Skelet", skel);
        // Hough line detector
        vector<Vec4i> lines;
        HoughLinesP(skel, lines, 1, CV_PI / 180, 5, face.height, face.height * .5);
        // Detect arm or hamd >> create bounding boxes
        for (int i = 0; i < lines.size(); i++) {
            line(skel, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), 255, 3);
            if(this->length(lines.at(i)) > face.height) {
                // ends op arms extra hands
                this->posArm.push_back(RotatedRect(Point(lines[i][0], lines[i][1]), Size(face.height / 2, face.height), this->angleLine(lines.at(i))));
                this->posArm.push_back(RotatedRect(Point(lines[i][2], lines[i][3]), Size(face.height / 2, face.height), this->angleLine(lines.at(i))));
                this->scoreArm.push_back(.1);
                this->scoreArm.push_back(.1);
            } else {
                // full line as hand
                this->posArm.push_back(RotatedRect(this->centerLine(lines.at(i)), Size(face.height / 2, face.height), this->angleLine(lines.at(i))));
                this->scoreArm.push_back(.1);
            }
        }
        imshow("Skelet", skel);
        imshow("Skin", skin);
        // Score detected regions
    }
}

Mat DetHand::skinSegmentation( Mat src) {
    // allocate the result matrix
    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

    Mat src_ycrcb, src_hsv;
    // OpenCV scales the YCrCb components, so that they
    // cover the whole value range of [0,255], so there's
    // no need to scale the values:
    cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
    // OpenCV scales the Hue Channel to [0,180] for
    // 8bit images, so make sure we are operating on
    // the full spectrum from [0,360] by using floating
    // point precision:
    src.convertTo(src_hsv, CV_32FC3);
    cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
    // Now scale the values between [0,255]:
    normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {

            Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
            int B = pix_bgr.val[0];
            int G = pix_bgr.val[1];
            int R = pix_bgr.val[2];
            // apply rgb rule
            bool e1 = (R>95) && (G>40) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
            bool e2 = (R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);
            bool a = (e1||e2);

            Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
            //int Y = pix_ycrcb.val[0];
            int Cr = pix_ycrcb.val[1];
            int Cb = pix_ycrcb.val[2];
            // apply ycrcb rule
            bool e3 = Cr <= 1.5862*Cb+20;
            bool e4 = Cr >= 0.3448*Cb+76.2069;
            bool e5 = Cr >= -4.5652*Cb+234.5652;
            bool e6 = Cr <= -1.15*Cb+301.75;
            bool e7 = Cr <= -2.2857*Cb+432.85;
            bool b = e3 && e4 && e5 && e6 && e7;

            Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
            float H = pix_hsv.val[0];
            //float S = pix_hsv.val[1];
            //float V = pix_hsv.val[2];
            // apply hsv rule
            bool c = (H<25) || (H > 230);

            if((a&&b&&c))
                dst.ptr(i)[j] = 255;
        }
    }
    return dst;
}

void DetHand::rotate(cv::Mat& src, double angle, cv::Mat& dst)
{
    cv::Point2f pt(src.cols / 2., src.rows / 2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
}

void DetHand::drawResult(Mat& img, RotatedRect &box, const Scalar& color, int thickness)
{
    Point2f corners[4];
    box.points(corners);

    for(int i = 0; i < 4; i++) {
        line(img, corners[i], corners[(i + 1) % 4], color, thickness);
    }

    Point dirLine(-sin(box.angle * CV_PI / 180) * 10 + box.center.x, cos(box.angle * CV_PI / 180) * 10 + box.center.y);
    line(img, box.center, dirLine, color);
    circle(img, box.center, 3, color);
}

RotatedRect DetHand::correction(Rect box, int angle, int correction, Point center, int orientCorr)
{
    float rat = angle * CV_PI / 180;
    float cx = box.x + box.width / 2;
    float cy = box.y + box.height / 2;
    float x, y;

    if(orientCorr == 0) {
        x = cos(rat) * (cx - center.x) - sin(rat) * (cy - center.x) + center.x;
        y = sin(rat) * (cx - center.y) + cos(rat) * (cy - center.y) + center.y - correction;
    } else {
        x = cos(rat) * (cx - center.x) - sin(rat) * (cy - center.x) + center.x - correction;
        y = sin(rat) * (cx - center.y) + cos(rat) * (cy - center.y) + center.y;
    }

    RotatedRect rot(Point2f(x, y), box.size(), angle);
    return rot;
}


vector<RotatedRect> DetHand::getLocationHand()
{
    return this->posHand;
}

vector<double> DetHand::getScoreHand()
{
    return this->scoreHand;
}

vector<RotatedRect> DetHand::getLocationContext()
{
    return this->posContext;
}

vector<double> DetHand::getScoreContext()
{
    return this->scoreContext;
}

vector<RotatedRect> DetHand::getLocationArm()
{
    return this->posArm;
}

vector<double> DetHand::getScoreArm()
{
    return this->scoreArm;
}

Mat DetHand::getSkin() {
    return this->skin;
}

double DetHand::length(Vec4i p) {
    return sqrt(pow(double(p[0] - p[2]) , 2) + pow(double(p[1] - p[3]) , 2));
}

Point DetHand::centerLine(Vec4i p)
{
    return Point(std::min(p[0], p[2]) + std::abs(p[0] - p[2]) / 2, std::min(p[1], p[3]) + std::abs(p[1] - p[3]) / 2);
}

double DetHand::angleLine(Vec4i p)
{
    return 180 + atan2(p[3] - p[1], p[2] - p[0]) * 180 / CV_PI + 90;
}

Mat DetHand::skeletonisation( Mat src ) {
    cv::threshold(src, src, 127, 255, cv::THRESH_BINARY);
    cv::Mat skel(src.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;

    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    bool done;

    do {
        erode(src, eroded, element);
        dilate(eroded, temp, element); // temp = open(img)
        subtract(src, temp, temp);
        bitwise_or(skel, temp, skel);
        eroded.copyTo(src);
        done = (countNonZero(src) == 0);
    } while(!done);
    return skel;
}

vector<RotatedRect> DetHand::absToRel(vector<RotatedRect> list, Point refPoint) {
    //cout << " ------ abs to rel ------ " << endl;
    for ( int i = 0; i < list.size(); i++ ) {
        //cout << "   " << i << ": " << list.at(i).center << "   ->  ";
        list.at(i).center.x = list.at(i).center.x - refPoint.x;
        list.at(i).center.y = list.at(i).center.y - refPoint.y;
        //cout << list.at(i).center << "   ref:  " << refPoint << endl;
    }
    return list;
}

vector<RotatedRect> DetHand::relToAbs(vector<RotatedRect> list, Point refPoint) {
    //cout << " ------ rel to abs ------ " << endl;
    for ( int i = 0; i < list.size(); i++ ) {
        //cout << "   " << i << ": " << list.at(i).center << "   ->  ";
        list.at(i).center.x = list.at(i).center.x + refPoint.x;
        list.at(i).center.y = list.at(i).center.y + refPoint.y;
        //cout << list.at(i).center << "   ref:  " << refPoint << endl;
    }
    return list;
}
