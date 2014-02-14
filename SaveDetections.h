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
        SaveDetections(QString filenameAvi );
        void newFrame (int frameNr);
        void newUpperBody (Rect box);
        void newHand (Rect box, int angle);
        void runtime (float time);
        ~SaveDetections ();

    private:
        QFile* file;
        QXmlStreamWriter* xmlWriter;
        int debtCounter;
};

#endif // SAVEDETECTIONS_H
