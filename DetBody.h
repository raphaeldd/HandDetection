/*!
 * @class DetBody
 * Detects upperbodys in images
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
        /// @brief Constructor from classe DetBody
        /// Preps the detector with the right model
        /// @param ModelBody Filename of model
        /// @param threshold Threshold value of detector
        DetBody(String modelBody, double threshold);

        /// @brief Apply the detector on a image
        /// @param image The image for the detector
        /// @param framenumber Frame number of the video (not really importend)
        void runDetection(Mat image , int frameNumber);

        /// @brief Get all the detection cutouts of the last detection
        /// @return All cutouts of type Mat put in a vector
        vector<Mat> getCutouts();

        /// @brief Get all the detection regions of the last detection
        /// @return All regions of type Rect put in a vector
        vector<Rect> getRect();

        /// @brief Get number of detections found
        /// @return Number of detections found of type integer
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
