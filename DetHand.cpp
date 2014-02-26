#include "DetHand.h"


struct RotRectLike {
    RotRectLike(double eps = 0.5, double theta = 0.5) {
        this->eps = eps;
        this->theta = theta;
    }
    bool operator()(RotatedRect a, RotatedRect b) {
        pair<double, double> ori1, ori2;
        int dx = eps * b.size.width;
        int dy = eps * b.size.height;
        ori1 = make_pair(cos(a.angle), sin(a.angle));
        ori2 = make_pair(cos(b.angle), sin(b.angle));
        return a.center.x >= (b.center.x - dx) && a.center.y >= (b.center.y - dy) &&
               (a.center.x + a.size.width) <= (b.center.x + b.size.width + dx) &&
               (a.center.y + a.size.height) <= (b.center.y + b.size.height + dy) &&
               sqrt(pow(ori1.second + ori2.second, 2) + pow(ori1.first + ori2.first, 2)) >= 2 - theta ;
    }
    double eps;
    double theta;
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
    int correction = (image.cols - image.rows) / 2;
    Mat blackImage(image.cols, image.cols, image.type(), double(0));
    Rect roi(Point(0, (image.cols - image.rows) / 2), Size(image.cols, image.rows));
    image.copyTo(blackImage(roi));
    clock_t T1, T2;
    this->detections.clear();
    this->pos.clear();
    T1 = clock();
#if  ( DET_TYPE == 0 )

    for(int r = 0; r < 360; r = r + 10) {
        Mat detImage;
        rotate(blackImage, r, detImage);
        this->FinalDetections.clear();
        this->FinalHandDetections.clear();
        this->FinalUpperHandDetections.clear();
        PersonDetection((frameNumber * 1000) + r, mixtureHand, FinalDetections, FinalUpperHandDetections, FinalHandDetections, detImage, TH);
        // Put detections in detection vectors
        rotate(blackImage, r, detImage);

        for(unsigned int k = 0; k < FinalHandDetections.size(); k++) {
            // TODO: Pas posities voor Absloute coordinaten tov origineel upper body cutout
            this->pos.push_back(make_pair(FinalHandDetections[k].detect , r));
            this->detections.push_back(detImage(this->pos.back().first).clone());
        }
    }

#elif ( DET_TYPE == 1 )
    int rows = 6, cols = 6;
    Mat detImage(blackImage.rows * rows, blackImage.cols * cols, blackImage.type());
    Mat cutImage;

    for(int r = 0; r < 360; r = r + 10) {
        Mat tmpImage;
        int y = floor((double)r / (10 * cols)) ;
        int x = (r - y * (10 * rows)) / 10 ;
        Rect roi(Point(x * blackImage.cols, y * blackImage.rows), Size(blackImage.rows, blackImage.cols));
        rotate(blackImage, r, tmpImage);
        tmpImage.copyTo(detImage(roi));
    }

    detImage.copyTo(cutImage);
    // Write result away
    imwrite("Patchwork.png", cutImage);
    this->FinalDetections.clear();
    this->FinalHandDetections.clear();
    this->FinalUpperHandDetections.clear();
    PersonDetection((frameNumber * 1000), mixtureHand, FinalDetections, FinalUpperHandDetections, FinalHandDetections, detImage, TH);

    // Put detections in detection vectors

    for(unsigned int k = 0; k < FinalHandDetections.size(); k++) {
        // TODO: Pas posities voor Absloute coordinaten tov origineel upper body cutout
        int rot = floor((double)FinalHandDetections[k].detect.x / (double)blackImage.cols) * 10;
        this->pos.push_back(make_pair(Rect(FinalHandDetections[k].detect.x % blackImage.cols, FinalHandDetections[k].detect.y, FinalHandDetections[k].detect.width, FinalHandDetections[k].detect.height), rot));
        this->detections.push_back(cutImage(FinalHandDetections[k].detect).clone());
    }

#else
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
            this->pos.push_back(this->correction(tmpRect, rot, correction, Point(blackImage.cols / 2, blackImage.rows / 2)));
            //this->detections.push_back(cutImage(FinalHandDetections[k].detect).clone());
        }

        iteration = (rows * cols * 10) + iteration;
    }

#endif
    // Merge overlapping detections
    Mat tmpRes = image.clone();

    for(int i = 0; i < this->pos.size(); i++)
        drawResult(tmpRes, this->pos[i], Scalar(0, 0, 255));

    similarRects(this->pos);

    for(int i = 0; i < this->pos.size(); i++)
        drawResult(tmpRes, this->pos[i], Scalar(0, 255, 0));

    imshow("Before and after", tmpRes);
    // TODO: redo cutouts with merged bounding boxes
    //          watch out they are rotated
    T2 = clock();
    float diff = ((float)T2 - (float)T1) / CLOCKS_PER_SEC;
    cout << "       Runtime rotation detection: " << setprecision(4) << diff << endl;
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

int DetHand::getSize()
{
    return this->detections.size();
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

    Point dirLine(-sin(box.angle * CV_PI / 180) * 10 + box.center.x, cos(box.angle * PI / 180) * 10 + box.center.y);
    line(img, box.center, dirLine, color);
    circle(img, box.center, 3, color);
}

RotatedRect DetHand::correction(Rect box, int angle, int correction, Point center)
{
    float rat = angle * CV_PI / 180;
    float cx = box.x + box.width / 2;
    float cy = box.y + box.height / 2;
    float x = cos(rat) * (cx - center.x) - sin(rat) * (cy - center.x) + center.x;
    float y = sin(rat) * (cx - center.y) + cos(rat) * (cy - center.y) + center.y  - correction;
    RotatedRect rot(Point2f(x, y), box.size(), angle);
    return rot;
}

void DetHand::similarRects(vector<RotatedRect>& rects)
{
    vector<int> label;
    RotRectLike rlike(0.5, 0.5);
    int labelSize = partition(rects, label , rlike);
    vector<RotatedRect> newRects(labelSize, RotatedRect(Point2f(0, 0), Size2f(0, 0), 0));
    vector<int> Sum(labelSize, 0);
    vector<pair<double, double> > angle(labelSize, make_pair(0, 0));

    for(unsigned int i = 0; i < rects.size(); i++) {
        int n = label[i];
        angle[n].first += cos(rects[i].angle * CV_PI / 180);
        angle[n].second += sin(rects[i].angle * CV_PI / 180);
        //newRects[n].angle = newRects[n].angle + rects[i].angle;
        newRects[n].center.x += rects[i].center.x;
        newRects[n].center.y += rects[i].center.y;
        newRects[n].size.height += rects[i].size.height;
        newRects[n].size.width += rects[i].size.width;
        Sum[n]++;
        cout << "Cluster " << n << ": " << endl;
        cout << "   Angle: " << rects[i].angle << " => " << cos(rects[i].angle * CV_PI / 180) << ", " << sin(rects[i].angle * CV_PI / 180) << endl << endl;
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




