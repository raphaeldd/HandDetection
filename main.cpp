/*!
 * @brief Main programma.
 *
 * Deze main functie gaat verschillende classen uitvoeren en de verkregen data doorgeven aan de volgende classes.
 * Ook worden hier de verschillende input parameters verwerkt.
 *
 * \author Den Dooven Raphael
 */

#include <QCoreApplication>

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

#include <signal.h>
#include <time.h>

#include "DetBody.h"
#include "DetHand.h"
#include "SaveDetections.h"
#include "HighestLikelihood.h"


using namespace std;
using namespace cv;

SaveDetections* save;

void signal_callback_handler(int signum)
{
    cout << endl << endl << "Signal caught" << endl;
    save->~SaveDetections();
    exit(signum);
}

int main(int argc, char *argv[])
{
    int c = 0;
    int startFrame = 1;
    int skipFrames = 1;
    String inputFile;

    while((c = getopt(argc, argv, "hb:s:")) != -1) {
        switch(c) {
            case 'h':
                cout << "Help" << endl;
                return 0;
                break;

            case 'b':
                startFrame = atoi(optarg);
                break;

            case 's':
                skipFrames = atoi(optarg);
                break;

            case '?':
                return 1;

            default:
                abort();
        }
    }

    inputFile = argv[argc - 1];

    if(inputFile.substr(inputFile.length() - 4) != ".avi") {
        cout << "No valid input file is found" << endl;
        return 1;
    }

    cout << "start frame: " << startFrame << endl;
    cout << "skip frames: " << skipFrames << endl;
    cout << "input file: " << inputFile << endl;
    // START OFFICIAL PROGRAM
    // ----------------------------------------
    // Opening image file
    VideoCapture cap(inputFile);

    if(!cap.isOpened()) {
        cout << "Invalid input data" << endl;
        return -1;
    }

    cap.set(CV_CAP_PROP_POS_FRAMES, startFrame - 1);
    // Detections
    cout << "---> Initalisation upper body detection." << endl;
    DetBody body("torso3comp.txt", -0.5);
    cout << "---> Initalisation hand detection." << endl;
    DetHand hand("hand.txt", "hand-context.txt", -0.3);



    // Program loop
    cout << "---> Start program loop." << endl;
    save = new SaveDetections(inputFile.c_str());
    signal(SIGINT, signal_callback_handler);

    while(cap.get(CV_CAP_PROP_POS_AVI_RATIO)  != -1) {

        cout << "-----> Frame " << cap.get(CV_CAP_PROP_POS_FRAMES) << "/" << cap.get(CV_CAP_PROP_FRAME_COUNT) << "." << endl;
        Mat cameraImage, textImage;
        cap >> cameraImage;
        cameraImage.copyTo(textImage);
        stringstream ss;

        // Run body detection
        cout << "-----> Run upper body detection." << endl;
        save->newFrame(cap.get(CV_CAP_PROP_POS_FRAMES));
        body.runDetection(cameraImage, cap.get(CV_CAP_PROP_POS_FRAMES));
        cout << body.getSize() << " detections found" << endl;

        // Do hand detection per body detection
        for(int n = 0; n < body.getSize(); n++) {
            clock_t T = clock();
            save->newUpperBody(body.getRect()[n]);
            ss.str("");
            ss << "Body detection: " << n + 1;
            imshow(ss.str(), body.getCutouts()[n]);

            // Run hand detection
            cout << "-----> Run Hand detection." << endl;
            hand.runDetection(body.getCutouts()[n], cap.get(CV_CAP_PROP_POS_FRAMES));
            body.getCutouts()[n].copyTo(cameraImage);



            // Run Highest Likelihood eliminator
            cout << "------> Run higest likelihood eliminator." << endl;
            HighestLikelihood likeli;
            Mat tmp = cameraImage.clone();
            likeli.run(tmp, hand.getRect(), hand.getScore(), 5);


            cout << "------> View results." << endl;
            Mat result = cameraImage.clone();
            for ( int r = 0; r < likeli.getResults().size(); r++ ) {
                hand.drawResult(result, likeli.getResults()[r], Scalar(255, 30, 30));
                save->newHand(likeli.getResults()[r]);
            }
            imshow("Result", result);


            T = clock() - T;
            save->newFace(likeli.getFace());
            save->runtime((float)T / CLOCKS_PER_SEC);

        }
        waitKey(100);
        //waitKey(-1);
        destroyAllWindows();
        // Skip frames
        cap.set(CV_CAP_PROP_POS_FRAMES, cap.get(CV_CAP_PROP_POS_FRAMES) + skipFrames - 1);
    }

    save->~SaveDetections();
    return EXIT_SUCCESS;
}
