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
//--------------------------------------------------------------------------------------------------

#include "SimpleOpt.h"

#include "Intersector.h"
#include "Mixture.h"
#include "Scene.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#ifndef _WIN32
#include <sys/time.h>

timeval Start, Stop;

inline void start()
{
    gettimeofday(&Start, 0);
}

inline int stop()
{
    gettimeofday(&Stop, 0);
    timeval duration;
    timersub(&Stop, &Start, &duration);
    return duration.tv_sec * 1000 + (duration.tv_usec + 500) / 1000;
}
#else
#include <time.h>
#include <windows.h>

ULARGE_INTEGER Start, Stop;

inline void start()
{
    GetSystemTimeAsFileTime((FILETIME *)&Start);
}

inline int stop()
{
    GetSystemTimeAsFileTime((FILETIME *)&Stop);
    Stop.QuadPart -= Start.QuadPart;
    return (Stop.QuadPart + 5000) / 10000;
}
#endif

using namespace FFLD;
using namespace std;

struct Detection : public FFLD::Rectangle {
    HOGPyramid::Scalar score;
    int l;
    int x;
    int y;

    Detection() : score(0), l(0), x(0), y(0) {
    }

    Detection(HOGPyramid::Scalar score, int l, int x, int y, FFLD::Rectangle bndbox) :
        FFLD::Rectangle(bndbox), score(score), l(l), x(x), y(y) {
    }

    bool operator<(const Detection & detection) const {
        return score > detection.score;
    }
};

void draw(JPEGImage & image, const FFLD::Rectangle & rect, uint8_t r, uint8_t g, uint8_t b,
          int linewidth)
{
    if(image.empty() || rect.empty() || (image.depth() < 3))
        return;

    const int width = image.width();
    const int height = image.height();
    const int depth = image.depth();
    uint8_t * bits = image.bits();
    // Draw 2 horizontal lines
    const int top = min(max(rect.top(), 1), height - linewidth - 1);
    const int bottom = min(max(rect.bottom(), 1), height - linewidth - 1);

    for(int x = max(rect.left() - 1, 0); x <= min(rect.right() + linewidth, width - 1); ++x) {
        if((x != max(rect.left() - 1, 0)) && (x != min(rect.right() + linewidth, width - 1))) {
            for(int i = 0; i < linewidth; ++i) {
                bits[((top + i) * width + x) * depth    ] = r;
                bits[((top + i) * width + x) * depth + 1] = g;
                bits[((top + i) * width + x) * depth + 2] = b;
                bits[((bottom + i) * width + x) * depth    ] = r;
                bits[((bottom + i) * width + x) * depth + 1] = g;
                bits[((bottom + i) * width + x) * depth + 2] = b;
            }
        }

        // Draw a white line below and above the line
        if((bits[((top - 1) * width + x) * depth    ] != 255) &&
                (bits[((top - 1) * width + x) * depth + 1] != 255) &&
                (bits[((top - 1) * width + x) * depth + 2] != 255))
            for(int i = 0; i < 3; ++i)
                bits[((top - 1) * width + x) * depth + i] = 255;

        if((bits[((top + linewidth) * width + x) * depth    ] != 255) &&
                (bits[((top + linewidth) * width + x) * depth + 1] != 255) &&
                (bits[((top + linewidth) * width + x) * depth + 2] != 255))
            for(int i = 0; i < 3; ++i)
                bits[((top + linewidth) * width + x) * depth + i] = 255;

        if((bits[((bottom - 1) * width + x) * depth    ] != 255) &&
                (bits[((bottom - 1) * width + x) * depth + 1] != 255) &&
                (bits[((bottom - 1) * width + x) * depth + 2] != 255))
            for(int i = 0; i < 3; ++i)
                bits[((bottom - 1) * width + x) * depth + i] = 255;

        if((bits[((bottom + linewidth) * width + x) * depth    ] != 255) &&
                (bits[((bottom + linewidth) * width + x) * depth + 1] != 255) &&
                (bits[((bottom + linewidth) * width + x) * depth + 2] != 255))
            for(int i = 0; i < 3; ++i)
                bits[((bottom + linewidth) * width + x) * depth + i] = 255;
    }

    // Draw 2 vertical lines
    const int left = min(max(rect.left(), 1), width - linewidth - 1);
    const int right = min(max(rect.right(), 1), width - linewidth - 1);

    for(int y = max(rect.top() - 1, 0); y <= min(rect.bottom() + linewidth, height - 1); ++y) {
        if((y != max(rect.top() - 1, 0)) && (y != min(rect.bottom() + linewidth, height - 1))) {
            for(int i = 0; i < linewidth; ++i) {
                bits[(y * width + left + i) * depth    ] = r;
                bits[(y * width + left + i) * depth + 1] = g;
                bits[(y * width + left + i) * depth + 2] = b;
                bits[(y * width + right + i) * depth    ] = r;
                bits[(y * width + right + i) * depth + 1] = g;
                bits[(y * width + right + i) * depth + 2] = b;
            }
        }

        // Draw a white line left and right the line
        if((bits[(y * width + left - 1) * depth    ] != 255) &&
                (bits[(y * width + left - 1) * depth + 1] != 255) &&
                (bits[(y * width + left - 1) * depth + 2] != 255))
            for(int i = 0; i < 3; ++i)
                bits[(y * width + left - 1) * depth + i] = 255;

        if((bits[(y * width + left + linewidth) * depth    ] != 255) &&
                (bits[(y * width + left + linewidth) * depth + 1] != 255) &&
                (bits[(y * width + left + linewidth) * depth + 2] != 255))
            for(int i = 0; i < 3; ++i)
                bits[(y * width + left + linewidth) * depth + i] = 255;

        if((bits[(y * width + right - 1) * depth    ] != 255) &&
                (bits[(y * width + right - 1) * depth + 1] != 255) &&
                (bits[(y * width + right - 1) * depth + 2] != 255))
            for(int i = 0; i < 3; ++i)
                bits[(y * width + right - 1) * depth + i] = 255;

        if((bits[(y * width + right + linewidth) * depth    ] != 255) &&
                (bits[(y * width + right + linewidth) * depth + 1] != 255) &&
                (bits[(y * width + right + linewidth) * depth + 2] != 255))
            for(int i = 0; i < 3; ++i)
                bits[(y * width + right + linewidth) * depth + i] = 255;
    }
}

void detect(const Mixture & mixture, int width, int height, const HOGPyramid & pyramid,
            double threshold, double overlap, const string image, ostream & out,
            const string & images, vector<Detection> & detections)
{
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

    /**
    //////////////////////////////////////////////////////////////
    for (int i = 0; i < detections.size(); ++i)
    {
        cout << "detection " << i << " = (" << detections[i].left() << "," << detections[i].top() << ") | ( " << detections[i].right() << "," << detections[i].bottom() << ")";
        int Height = detections[i].bottom()-detections[i].top();
        int Width = detections[i].right()-detections[i].left();
        cout << "Width = "<< Width;
        cout << "Height = "<< Height;
        if(Width>Height)
        {
            detections.erase(detections.begin()+i);
            cout << "--> W > H" << "--> remove detection: " << i;
            i--;        // decrease counter since vector is one item smaller
        }
        cout << endl;
    }
    cout << endl;
    /////////////////////////////////////////////////////////////
    * **/

    if(out) {
        #pragma omp critical

        for(int i = 0; i < detections.size(); ++i)
            out << id << ' ' << detections[i].score << ' ' << (detections[i].left() + 1) << ' '
                << (detections[i].top() + 1) << ' ' << (detections[i].right() + 1) << ' '
                << (detections[i].bottom() + 1) << endl;
    }

    if(!images.empty()) {
        JPEGImage im(image);

        for(int j = 0; j < detections.size(); ++j) {
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
                draw(im, bndbox, 255, 255, 0, 2);
            }

            // Draw the root last
            draw(im, detections[j], 255, 0, 0, 2);
        }

        im.save(images + '/' + id + ".jpg");
    }
}

// Test a mixture model (compute a ROC curve)
int main(int argc, char * argv[])
{
    cout << "ffld adapted by SDB" << endl;
    // Default parameters
    string model("../torso3comp.txt");
    Object::Name name = Object::PERSON;
    int nbNegativeScenes = -1;  // max number of neg images to consider = all
    int padding = 12;           // Amount of zero padding in HOG cells
    int interval = 10;          // Number of levels per octave in HOG pyramid
    double threshold = -0.6; // Minimum detection TH
    double overlap = 0.6;       // Minimum overlap in nms
    string images("./Detections");
    string inputFrame("/home/sdb/Doctoraat/InSightOut/VisualWords/DataSets/PresentationData/FrameFolderJPG/image00001.txt");
    string results("result.txt");
    char FileName[200];
    // Try to open the mixture
    ifstream in(model.c_str(), ios::binary);

    if(!in.is_open()) {
        cerr << "\nInvalid model file " << model << endl;
        return -1;
    }

    Mixture mixture;
    in >> mixture;

    if(mixture.empty()) {
        cerr << "\nInvalid model file " << model << endl;
        return -1;
    }

    // The image/dataset
    const string file = inputFrame;
    const size_t lastDot = file.find_last_of('.');

    if((lastDot == string::npos) ||
            ((file.substr(lastDot) != ".jpg") && (file.substr(lastDot) != ".txt"))) {
        cerr << "\nInvalid file " << file << ", should be .jpg or .txt" << endl;
        return -1;
    }

    // Try to open the results
    ofstream out;

    if(!results.empty()) {
        out.open(results.c_str(), ios::binary);

        if(!out.is_open()) {
            cerr << "\nInvalid results file " << results << endl;
            return -1;
        }
    }

    // Most of the computations inside are already multi-threaded but the performance is higher
    // (~20% on my machine) if the threading is done at the level of the scenes rather than at a
    // lower level (pyramid levels/filters)
    // The performace measurements reported in the paper were done without this scene level
    // threading
    start();
    int maxRows = 240;
    int maxCols = 320;

    if(!Patchwork::Init((maxRows + 15) & ~15, (maxCols + 15) & ~15)) {
        cerr << "\nCould not initialize the Patchwork class" << endl;
        return -1;
    }

    cout << "Initialized FFTW in " << stop() << " ms" << endl;
    start();
    mixture.cacheFilters();
    cout << "Transformed the filters in " << stop() << " ms" << endl;
    start();
    int i;

    //#pragma omp parallel for private(i)
    for(i = 0; i < 1; i++) {
        cout << "in for lus structure " << i  << endl;
        sprintf(FileName, "%s%05d%s", "/home/sdb/Doctoraat/InSightOut/VisualWords/DataSets/PresentationData/FrameFolderJPG/image", i, ".jpg");
        //sprintf(FileName,"%s%05d%s", "/home/sdb/Doctoraat/InSightOut/VisualWords/DataSets/OpnamesMuseumM/4Steven/output_Person/imgCrop-",i,"jpg");
        JPEGImage image(FileName);

        if(image.empty()) {
            cerr << "\nInvalid image " << FileName << endl;
            //return -1;
        } else {
            cout << "calcultation of image " << i << endl;
            HOGPyramid pyramid(image, padding, padding, interval);
            // Compute the detections
            vector<Detection> detections;
            detect(mixture, image.width(), image.height(), pyramid, threshold, overlap,
                   FileName, out, images, detections);

            for(int j = 0; j < detections.size(); j++) {
                const FFLD::Rectangle & rect = detections[j];
                cout << "detection " << j << " = (" << rect.left() << "," << rect.top() << ") | ( " << rect.right() << "," << rect.bottom() << ")" << endl;
            }
        }
    }

    cout << "Computed the convolutions and distance transforms in " << stop() << " ms" << endl;
    return EXIT_SUCCESS;
}
