#include "DetHand.h"

DetHand::DetHand(String modelHand, double threshold)
{
    // initialization: open mixtureHand + initialize patchwork class + calculate filter transform
    int res = Init(mixtureHand, modelHand);
    if ( res == -1 ) {
        exit(1);
    }
    this->TH = threshold;  // -0.5
}

void DetHand::runDetection( Mat image , int frameNumber) {
    int correction = ( image.cols - image.rows ) / 2;

    Mat blackImage(image.cols, image.cols, image.type(), double(0));
    Rect roi(Point(0, ( image.cols - image.rows )/2), Size(image.cols, image.rows));
    image.copyTo( blackImage( roi ) );

    clock_t T1, T2;

    this->detections.clear();
    this->pos.clear();


    T1 = clock();
#if  ( DET_TYPE == 0 )
    for ( int r = 0; r < 360; r = r + 10 ) {
        Mat detImage;
        rotate(blackImage, r, detImage);

        this->FinalDetections.clear();
        this->FinalHandDetections.clear();
        this->FinalUpperHandDetections.clear();

        PersonDetection((frameNumber * 1000) + r, mixtureHand, FinalDetections, FinalUpperHandDetections, FinalHandDetections, detImage, TH);

        // Put detections in detection vectors
        rotate(blackImage, r, detImage);
        for ( unsigned int k = 0; k < FinalHandDetections.size(); k++ )
        {
            // TODO: Pas posities voor Absloute coordinaten tov origineel upper body cutout
            this->pos.push_back( make_pair( FinalHandDetections[k].detect , r) );
            this->detections.push_back(detImage(this->pos.back().first).clone());
        }
    }
#elif ( DET_TYPE == 1 )
    int rows = 6, cols = 6;

    Mat detImage(blackImage.rows * rows, blackImage.cols * cols, blackImage.type());
    Mat cutImage;

    for ( int r = 0; r < 360; r = r + 10 ) {
        Mat tmpImage;
        int y = floor( (double)r / (10 * cols) ) ;
        int x = ( r - y * (10 * rows) ) / 10 ;
        Rect roi( Point( x * blackImage.cols, y * blackImage.rows ), Size( blackImage.rows, blackImage.cols ) );
        rotate( blackImage, r, tmpImage );
        tmpImage.copyTo( detImage( roi ) );

    }


    detImage.copyTo(cutImage);

    // Write result away
    imwrite("Patchwork.png", cutImage);


    this->FinalDetections.clear();
    this->FinalHandDetections.clear();
    this->FinalUpperHandDetections.clear();

    PersonDetection((frameNumber * 1000), mixtureHand, FinalDetections, FinalUpperHandDetections, FinalHandDetections, detImage, TH);

    // Put detections in detection vectors

    for ( unsigned int k = 0; k < FinalHandDetections.size(); k++ )
    {
        // TODO: Pas posities voor Absloute coordinaten tov origineel upper body cutout
        int rot = floor( (double)FinalHandDetections[k].detect.x / (double)blackImage.cols ) * 10;
        this->pos.push_back( make_pair( Rect ( FinalHandDetections[k].detect.x % blackImage.cols, FinalHandDetections[k].detect.y, FinalHandDetections[k].detect.width, FinalHandDetections[k].detect.height ), rot ) );
        this->detections.push_back(cutImage(FinalHandDetections[k].detect).clone());
    }

#else
    int rows = 1, cols = 2, iteration = 0;

    this->FinalDetections.clear();
    this->FinalHandDetections.clear();
    this->FinalUpperHandDetections.clear();

    while ( iteration < 360 ) {
        Mat detImage(blackImage.rows * rows, blackImage.cols * cols, blackImage.type());
        Mat cutImage;

        for ( int r = iteration; r < ( rows * cols * 10 ) + iteration; r = r + 10 ) {
            Mat tmpImage;
            int y = floor( (double)( r - iteration ) / (10 * cols) ) ;
            int x = ( r - iteration - y * (10 * rows) ) / 10 ;
            Rect roi( Point( x * blackImage.cols, y * blackImage.rows ), Size( blackImage.rows, blackImage.cols ) );
            rotate( blackImage, r, tmpImage );
            tmpImage.copyTo( detImage( roi ) );
        }
        detImage.copyTo(cutImage);



        PersonDetection((frameNumber * 1000) + iteration, mixtureHand, FinalDetections, FinalUpperHandDetections, FinalHandDetections, detImage, TH);

        // Put detections in detection vectors

        for ( unsigned int k = this->pos.size(); k < FinalHandDetections.size(); k++ )
        {
            int rot = floor( (double)FinalHandDetections[k].detect.x / (double)blackImage.cols ) * 10 + iteration;
            Rect tmpRect( FinalHandDetections[k].detect.x % blackImage.cols, FinalHandDetections[k].detect.y, FinalHandDetections[k].detect.width, FinalHandDetections[k].detect.height );
            this->pos.push_back( this->correction(tmpRect, rot, correction, Point( blackImage.cols/2, blackImage.rows/2 ) ) );
            this->detections.push_back(cutImage(FinalHandDetections[k].detect).clone());
        }

        iteration = ( rows * cols * 10 ) + iteration;
    }
#endif

    T2 = clock();
    float diff = ( (float)T2 - (float)T1 ) / CLOCKS_PER_SEC;
    cout << "       Runtime rotation detection: " << setprecision(4) << diff << endl;
    cout << "       Found detections:           " << this->getRect().size() << endl;
    cout << "       Found detections:           " << this->getCutouts().size() << endl;

//    vector<int> label;
//    cout << "Total classes found: " << partition(this->pos, label, similarRotRects(1, 1) ) << endl;
//    for (unsigned int i = 0; i < this->pos.size(); i++) {
//        cout << i << ".         " << this->pos.at(i).center.x << ", " << this->pos.at(i).center.x << "; angle: " << this->pos.at(i).angle << "; label: " << label.at(i) << endl;
//    }
}

vector<Mat> DetHand::getCutouts() {
    return this->detections;
}

vector<RotatedRect> DetHand::getRect() {
    return this->pos;
}

int DetHand::getSize() {
    return this->detections.size();
}

void DetHand::rotate(cv::Mat& src, double angle, cv::Mat& dst)
{
    cv::Point2f pt(src.cols/2., src.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
}

void DetHand::drawResult(Mat& img, RotatedRect &box, const Scalar& color)
{
    Point2f corners[4];
    box.points(corners);
    for (int i = 0; i < 4; i++)
    {
        line( img, corners[i], corners[(i+1)%4], color );
    }
    Point dirLine( sin(box.angle* CV_PI/180) * 10 + box.center.x, cos(box.angle* PI/180) * 10 + box.center.y );
    line( img, box.center, dirLine, color );
    circle(img, box.center, 3, color);
}

RotatedRect DetHand::correction(Rect box, int angle, int correction, Point center)
{
    float rat = angle * CV_PI/180;
    float cx = box.x + box.width/2;
    float cy = box.y + box.height/2;
    float x = cos(rat) * (cx - center.x) - sin(rat) * (cy - center.x) + center.x;
    float y = sin(rat) * (cx - center.y) + cos(rat) * (cy - center.y) + center.y  - correction;
    RotatedRect rot(Point2f(x, y), box.size(), angle);
    return rot;
}




