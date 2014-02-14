/*!
 * @brief Body detection class
 *
 * @author Den Dooven Raphael
 *
 */

#ifndef DETBODY_H
#define DETBODY_H

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



class DetBody
{
    public:
        DetBody(String modelBody, double threshold);
        void runDetection(Mat image , int frameNumber);
        vector<Mat> getCutouts();
        vector<Rect> getRect();
        int getSize();

    private:
        vector<UniqueDetections> FinalUpperBodyDetections;
        vector<UniqueDetections> FinalTorsoDetections;
        vector<UniqueDetections> FinalDetections;
        Mixture mixtureBody;
        String modelBody;

        vector<Mat> detections;
        vector<Rect> pos;

        double TH;

};

#endif // DETBODY_H
