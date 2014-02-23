/// @class SaveDetections Saves the detections.
/// Puts al the detections in a handy XML file. So that when rerunning the program no unnessesary detections ar avoided
///
/// @author Den Dooven Raphael

#ifndef SAVEDETECTIONS_H
#define SAVEDETECTIONS_H

#include <iostream>
#include <QCoreApplication>
#include <QFile>
#include <QXmlStreamWriter>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/ml/ml.hpp"
#include <opencv2/gpu/gpu.hpp>

using namespace cv;

class SaveDetections
{
    public:
        /// @brief Constructor of class SaveDetections
        /// Opens the xml file Detections.xml in local directory
        /// @param filenameAvi gives new entery for specific AVI file
        SaveDetections(QString filenameAvi );

        /// @brief Adds new element frame to XML
        /// @param framNr Puts the frame number in the XML file
        void newFrame (int frameNr);

        /// @brief adds new element for a upper body detection
        /// @param box Position information of the detection
        void newUpperBody (Rect box);

        /// @brief adds new element for a hand detection
        /// @param box Position information of the detection
        /// @param angle rotaion of the detection
        void newHand (RotatedRect box);

        /// @brief adds a runtime element to the XML
        /// @param time Runtime
        void runtime (float time);

        /// @brief Destructor of the class
        /// Closes the file and write the data to the XML file
        ~SaveDetections ();

    private:
        QFile* file;
        QXmlStreamWriter* xmlWriter;
        int debtCounter;
};

#endif // SAVEDETECTIONS_H
