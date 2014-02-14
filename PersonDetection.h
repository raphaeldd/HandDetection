#ifndef PERSONDETECTION_H
#define PERSONDETECTION_H

#include "SimpleOpt.h"
#include "Intersector.h"
#include "Mixture.h"
#include "Scene.h"
#include "PersonDetection.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/ml/ml.hpp"
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace gpu;
using namespace FFLD;
using namespace std;

//----------------------------------------------------------------//
// Structures
//----------------------------------------------------------------//
struct UniqueDetections
{
    Point Center;
    cv::Rect detect;
    double BestScore;
};

struct TSDetection
{
    Point Center;
    cv::Rect Rectangle;
    int weight;
    int update;
    int lastFrameNr;
    int firstFrameNr;
    int FrameNr;
    int estimate;
    double DetectionScore;
};


struct Detection : public FFLD::Rectangle
{
    HOGPyramid::Scalar score;
    int l;
    int x;
    int y;

    Detection() : score(0), l(0), x(0), y(0)
    {
    }

    Detection(HOGPyramid::Scalar score, int l, int x, int y, FFLD::Rectangle bndbox) :
    FFLD::Rectangle(bndbox), score(score), l(l), x(x), y(y)
    {
    }

    bool operator<(const Detection & detection) const
    {
        return score > detection.score;
    }
};

//----------------------------------------------------------------//
// Functions
//----------------------------------------------------------------//

void detect(const Mixture & mixture, int width, int height, const HOGPyramid & pyramid,
            double threshold, double overlap, const string image,
            const string & images, vector<Detection> &detections, vector<Detection> &Facedetections, vector<Detection> &Bodydetections, Mat CVimage);

int PersonDetection(int framenr,Mixture & mixture,vector<UniqueDetections> & FinalDetections, vector<UniqueDetections> & FinalUpperBodyDetections,vector<UniqueDetections> & FinalTorsoDetections, Mat CVimage, double TH);

int Init(Mixture &mixture, string model);

void NonMaxSuppr(vector<Detection> input,vector<UniqueDetections> & output, int nmsFactor );


#endif // PERSONDETECTION_H
