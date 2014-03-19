#include "DetHand.h"


struct RotRectLike {
        RotRectLike(double eps = 0.1, int sigma = 30) {
            this->eps = eps;
            this->sigma = sigma;
        }
        bool operator()(RotatedRect a, RotatedRect b) {
            //return  abs(((int)a.angle + 180 - (int)b.angle) % 360 - 180) <= sigma  &&
            return        sqrt( pow(a.center.x - b.center.x, 2) + pow(a.center.y - b.center.y, 2) ) <= ( a.size.area() * pow(eps, 2) );
        }
        double eps;
        int sigma;
};

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

void DetHand::runDetection(Mat image , int frameNumber)
{
    int correction, orientCor;
    Mat blackImage;
    Rect roi;

    if ( image.cols >= image.rows ) {
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
    this->detections.clear();
    this->pos.clear();

    int rows = 1, cols = 2, iteration = 0;
    this->FinalDetections.clear();
    this->FinalHandDetections.clear();
    this->FinalUpperHandDetections.clear();

    while(iteration < 360) {
        Mat detImage(blackImage.rows * rows, blackImage.cols * cols, blackImage.type());
        Mat cutImage;

        for(int r = iteration; r < (rows * cols * 10) + iteration; r = r + 10) {
            Mat tmpImage;
            int y = floor((double)(r - iteration) / (10 * cols)) ;
            int x = (r - iteration - y * (10 * rows)) / 10 ;
            Rect roi(Point(x * blackImage.cols, y * blackImage.rows), Size(blackImage.rows, blackImage.cols));
            rotate(blackImage, r, tmpImage);
            tmpImage.copyTo(detImage(roi));
        }

        detImage.copyTo(cutImage);
        PersonDetection((frameNumber * 1000) + iteration, mixtureHand, FinalDetections, FinalUpperHandDetections, FinalHandDetections, detImage, TH);
        PersonDetection((frameNumber * 1000) + iteration, mixtureHandContext, FinalDetections, FinalUpperHandDetections, FinalHandDetections, detImage, TH);

        // Put detections in detection vectors
        for(unsigned int k = this->pos.size(); k < FinalHandDetections.size(); k++) {
            int rot = floor((double)FinalHandDetections[k].detect.x / (double)blackImage.cols) * 10 + iteration;
            Rect tmpRect(FinalHandDetections[k].detect.x % blackImage.cols, FinalHandDetections[k].detect.y, FinalHandDetections[k].detect.width, FinalHandDetections[k].detect.height);
            this->pos.push_back(this->correction(tmpRect, rot, correction, Point(blackImage.cols / 2, blackImage.rows / 2), orientCor));
            this->score.push_back(FinalHandDetections.at(k).BestScore);
        }

        iteration = (rows * cols * 10) + iteration;
    }

    // Merge overlapping detections
    Mat tmpRes = image.clone();

//        for(unsigned int i = 0; i < this->pos.size(); i++)
//            drawResult(tmpRes, this->pos[i], Scalar(0, 0, 255));

    similarRects(this->pos);

    //    for(unsigned int i = 0; i < this->pos.size(); i++)
    //        drawResult(tmpRes, this->pos[i], Scalar(0, 255, 0));

    //    imshow("Before and after", tmpRes);
    //    waitKey(-1);
    // TODO: redo cutouts with merged bounding boxes
    //          watch out they are rotated
    cout << "       Found detections:           " << this->pos.size() << endl;
}

vector<Mat> DetHand::getCutouts()
{
    return this->detections;
}

vector<RotatedRect> DetHand::getRect()
{
    return this->pos;
}

vector<double> DetHand::getScore(){
    return this->score;
}

int DetHand::getSize()
{
    return this->pos.size();
}

void DetHand::rotate(cv::Mat& src, double angle, cv::Mat& dst)
{
    cv::Point2f pt(src.cols / 2., src.rows / 2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
}

void DetHand::drawResult(Mat& img, RotatedRect &box, const Scalar& color)
{
    Point2f corners[4];
    box.points(corners);

    for(int i = 0; i < 4; i++) {
        line(img, corners[i], corners[(i + 1) % 4], color);
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
    if (orientCorr == 0 ) {
        x = cos(rat) * (cx - center.x) - sin(rat) * (cy - center.x) + center.x;
        y = sin(rat) * (cx - center.y) + cos(rat) * (cy - center.y) + center.y - correction;
    } else {
        x = cos(rat) * (cx - center.x) - sin(rat) * (cy - center.x) + center.x - correction;
        y = sin(rat) * (cx - center.y) + cos(rat) * (cy - center.y) + center.y;
    }
    RotatedRect rot(Point2f(x, y), box.size(), angle);
    return rot;
}

void DetHand::similarRects(vector<RotatedRect>& rects)
{
    vector<int> label;
    RotRectLike rlike(0.05, 20);
    cout << "Rects found:   " << rects.size() << endl;
    int labelSize = partition(rects, label , rlike);
    cout << "Cluster found: " << labelSize << endl;


    vector<RotatedRect> newRects(labelSize, RotatedRect(Point2f(0, 0), Size2f(0, 0), 0));
    vector<int> Sum(labelSize, 0);
    vector<pair<double, double> > angle(labelSize, make_pair(0, 0));

    for(unsigned int i = 0; i < rects.size(); i++) {
        int n = label[i];
        angle[n].first += cos(rects[i].angle * CV_PI / 180);
        angle[n].second += sin(rects[i].angle * CV_PI / 180);
        newRects[n].center.x += rects[i].center.x;
        newRects[n].center.y += rects[i].center.y;
        newRects[n].size.height += rects[i].size.height;
        newRects[n].size.width += rects[i].size.width;
        Sum[n]++;
    }

    for(int n = 0; n < labelSize; n++) {
        newRects[n].angle = atan2(angle[n].second, angle[n].first) * 180 / CV_PI;
        newRects[n].center.x = newRects[n].center.x / Sum[n];
        newRects[n].center.y = newRects[n].center.y / Sum[n];
        newRects[n].size.height = newRects[n].size.height / Sum[n];
        newRects[n].size.width = newRects[n].size.width / Sum[n];
    }

    rects.clear();
    rects.swap(newRects);
}




