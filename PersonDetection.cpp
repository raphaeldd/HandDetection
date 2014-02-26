//--------------------------------------------------------------------------------------------------
// Implementation of the paper "Exact Acceleration of Linear Object Detectors", 12th European
// Conference on Computer Vision, 2012.
//
// Copyright (c) 2012 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
//
// This file is part of FFLD (the Fast Fourier Linear Detector)
//
// FFLD is free software: you can redistribute it and/or modify it under the terms of the GNU
// General Public License version 3 as published by the Free Software Foundation.
//
// FFLD is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU General Public License along with FFLD. If not, see
// <http://www.gnu.org/licenses/>.
//
// This file was adapted by Stijn De Beugher 2013.
//--------------------------------------------------------------------------------------------------
#include "PersonDetection.h"
#include "Parameters.h"

#define RatioTest
#define MaxCols 320
#define MaxRows 240


/************************************************************************
This function is a creates a JPEGImage element
Input = OpenCV MAT image
Return value image in JPEGImage format
************************************************************************/
void detect(const Mixture & mixture, int width, int height, const HOGPyramid & pyramid,
            double threshold, double overlap, const string image,
            const string & images, vector<Detection> & detections, vector<Detection> & Facedetections, vector<Detection> & Bodydetections, Mat CVimage)
{
    Mat PartsImage = CVimage.clone();
    // Compute the scores
    vector<HOGPyramid::Matrix> scores;
    vector<Mixture::Indices> argmaxes;
    vector<vector<vector<Model::Positions> > > positions;

    if(!images.empty())
        mixture.convolve(pyramid, scores, argmaxes, &positions);
    else
        mixture.convolve(pyramid, scores, argmaxes);

    // Cache the size of the models
    vector<pair<int, int> > sizes(mixture.models().size());

    for(int i = 0; i < sizes.size(); ++i)
        sizes[i] = mixture.models()[i].rootSize();

    // For each scale
    for(int i = pyramid.interval(); i < scores.size(); ++i) {
        // Scale = 8 / 2^(1 - i / interval)
        const double scale = pow(2.0, static_cast<double>(i) / pyramid.interval() + 2.0);
        const int rows = scores[i].rows();
        const int cols = scores[i].cols();

        for(int y = 0; y < rows; ++y) {
            for(int x = 0; x < cols; ++x) {
                const float score = scores[i](y, x);

                if(score > threshold) {
                    if(((y == 0) || (x == 0) || (score > scores[i](y - 1, x - 1))) &&
                            ((y == 0) || (score > scores[i](y - 1, x))) &&
                            ((y == 0) || (x == cols - 1) || (score > scores[i](y - 1, x + 1))) &&
                            ((x == 0) || (score > scores[i](y, x - 1))) &&
                            ((x == cols - 1) || (score > scores[i](y, x + 1))) &&
                            ((y == rows - 1) || (x == 0) || (score > scores[i](y + 1, x - 1))) &&
                            ((y == rows - 1) || (score > scores[i](y + 1, x))) &&
                            ((y == rows - 1) || (x == cols - 1) || (score > scores[i](y + 1, x + 1)))) {
                        FFLD::Rectangle bndbox((x - pyramid.padx()) * scale + 0.5,
                                               (y - pyramid.pady()) * scale + 0.5,
                                               sizes[argmaxes[i](y, x)].second * scale + 0.5,
                                               sizes[argmaxes[i](y, x)].first * scale + 0.5);
                        // Truncate the object
                        bndbox.setX(max(bndbox.x(), 0));
                        bndbox.setY(max(bndbox.y(), 0));
                        bndbox.setWidth(min(bndbox.width(), width - bndbox.x()));
                        bndbox.setHeight(min(bndbox.height(), height - bndbox.y()));

                        if(!bndbox.empty())
                            detections.push_back(Detection(score, i, x, y, bndbox));
                    }
                }
            }
        }
    }

    // Non maxima suppression
    sort(detections.begin(), detections.end());

    for(int i = 1; i < detections.size(); ++i)
        detections.resize(remove_if(detections.begin() + i, detections.end(),
                                    Intersector(detections[i - 1], overlap, true)) -
                          detections.begin());

    // Print the detection
    const size_t lastDot = image.find_last_of('.');
    string id = image.substr(0, lastDot);
    const size_t lastSlash = id.find_last_of("/\\");

    if(lastSlash != string::npos)
        id = id.substr(lastSlash + 1);

    // Show parts
#ifdef ShowParts

    for(int j = 0; j < detections.size(); ++j) {
        int randomVal_R = rand() & 255;
        int randomVal_G = rand() & 255;
        int randomVal_B = rand() & 255;
        // The position of the root one octave below
        const int argmax = argmaxes[detections[j].l](detections[j].y, detections[j].x);
        const int x2 = detections[j].x * 2 - pyramid.padx();
        const int y2 = detections[j].y * 2 - pyramid.pady();
        const int l = detections[j].l - pyramid.interval();
        // Scale = 8 / 2^(1 - j / interval)
        const double scale = pow(2.0, static_cast<double>(l) / pyramid.interval() + 2.0);

        for(int k = 0; k < positions[argmax].size(); ++k) {
            const FFLD::Rectangle bndbox((positions[argmax][k][l](y2, x2)(0) - pyramid.padx()) *
                                         scale + 0.5,
                                         (positions[argmax][k][l](y2, x2)(1) - pyramid.pady()) *
                                         scale + 0.5,
                                         mixture.models()[argmax].partSize().second * scale + 0.5,
                                         mixture.models()[argmax].partSize().second * scale + 0.5);
            Point P1, P2;
            P1.y = bndbox.top();
            P1.x = bndbox.left();
            P2.y = bndbox.bottom();
            P2.x = bndbox.right();
            Rect PartRectangle(P1, P2);
            rectangle(PartsImage, PartRectangle, Scalar(randomVal_R, randomVal_G, randomVal_B));
            imshow("Parts", PartsImage);
            waitKey();
        }
    }

#endif
    //////////////////////////////////////////////////////////////
#ifdef RatioTest
    cout << "ratiotes = on!" << endl;

    for(int i = 0; i < detections.size(); ++i) {
        double Height = (double)detections[i].bottom() - detections[i].top();
        double Width = (double)detections[i].right() - detections[i].left();
#ifdef debug
        cout << "detection " << i << " = (" << detections[i].left() << "," << detections[i].top() << ") | ( " << detections[i].right() << "," << detections[i].bottom() << ") --> score = " << detections[i].score << endl;
        cout << "Width = " << Width;
        cout << " Height = " << Height;
        cout << " ratio = " << Width / Height << endl;
#endif

        if(Width <= Height) {
            // store as body detection
            Bodydetections.push_back(detections[i]);
        } else {
            // store as face detection
            Facedetections.push_back(detections[i]);
        }
    }

#endif
}

/************************************************************************
This function calls the actual ffld person detection routine
input:  current frame nr, mixture(model), vector for all detections, vector for upperbody detections
        vector for torso detections, Mat image, detection threshold
************************************************************************/
int PersonDetection(int framenr, Mixture & mixture, vector<UniqueDetections> & FinalDetections, vector<UniqueDetections> & FinalUpperBodyDetections, vector<UniqueDetections> & FinalTorsoDetections, Mat CVimage, double TH)
{
    char FileName[200];
    string images("EnableParts");  // enable parts
    int padding = 10;           // Amount of zero padding in HOG cells
    int interval = 10;          // Number of levels per octave in HOG pyramid
    double overlap = 0.55;      // Minimum overlap in nms
    //******************************************************//
    JPEGImage image(CVimage);

    if(!image.empty()) {
        cout << endl << "calcultation of image " << framenr << endl;
        HOGPyramid pyramid(image, padding, padding, interval);
        vector<Detection> detections;
        vector<Detection> UpperBodydetections;
        vector<Detection> Torsodetections;
        detect(mixture, image.width(), image.height(), pyramid, TH, overlap,
               FileName, images, detections, UpperBodydetections, Torsodetections, CVimage);
#ifndef RatioTest
        NonMaxSuppr(detections, FinalDetections);
#endif
#ifdef RatioTest
        NonMaxSuppr(UpperBodydetections, FinalUpperBodyDetections, 50);
        NonMaxSuppr(Torsodetections, FinalTorsoDetections, 50);
#endif
        // clear vectors
        detections.clear();
        UpperBodydetections.clear();
        Torsodetections.clear();
    }
}


/************************************************************************
This is this initialization function
This functions opens the model and transforms it into a mixture structure
input: pointer to mixture structure, path to model
output: -1 if something was wrong
************************************************************************/
int Init(Mixture & mixture, string model)
{
    int maxRows = MaxRows;
    int maxCols = MaxCols;
    //******************************************************//
    // Try to open the mixture
    ifstream in(model.c_str(), ios::binary);

    if(!in.is_open()) {
        cerr << "\nInvalid model file " << model << endl;
        return -1;
    }

    in >> mixture;

    if(mixture.empty()) {
        cerr << "\nInvalid model file " << model << endl;
        return -1;
    }

    if(!Patchwork::Init((maxRows + 15) & ~15, (maxCols + 15) & ~15)) {
        cerr << "\nCould not initialize the Patchwork class" << endl;
        return -1;
    }

    //******************************************************//
    // Filter transform
    mixture.cacheFilters();
    //******************************************************//
}

/************************************************************************
This function is an extra non maxima suppression for the ffld_based detector
intput is a vector of dections, a vector of "unique detections" and a overlap factor
Detections are clustered together if there is enough overlap between their bounding boxes
clustered detections are written into the "unique detections" vector
************************************************************************/
void NonMaxSuppr(vector<Detection> input, vector<UniqueDetections> & output, int nmsFactor)
{
    for(int j = 0; j < input.size(); j++) {
        const FFLD::Rectangle & rect = input[j];
        Point P1(rect.left(), rect.top());
        Point P2(rect.right(), rect.bottom());
        Rect R1(P1, P2);
        int width = rect.right() - rect.left();
        int height = rect.top() - rect.bottom();
        Point Center;
        Center.x = rect.left() + width / 2;
        Center.y = rect.bottom() + height / 2;
        int assigned = 0;

        for(int DetSize = 0; DetSize < output.size(); DetSize++) {
            double res = norm(Center - output[DetSize].Center);

            if(res < nmsFactor && res > 0) {
                assigned = 1;
                output[DetSize].Center = Center; // update center

                if(input[j].score > output[DetSize].BestScore) {
                    output[DetSize].BestScore = input[j].score;
                    output[DetSize].detect = R1;
                    break;
                }
            }
        }

        if(assigned == 0) {
            UniqueDetections UniqueDet;
            UniqueDet.BestScore = input[j].score;
            UniqueDet.detect = R1;
            output.push_back(UniqueDet);
        }
    }
}
