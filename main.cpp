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

void signal_callback_handler (int signum) {
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

    while ( ( c = getopt(argc, argv, "hb:s:") ) != -1 ) {
        switch (c) {
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

    inputFile = argv[argc-1];
    if ( inputFile.substr(inputFile.length()-4) != ".avi" ) {
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
    if (!cap.isOpened())
    {
        cout << "Invalid input data" << endl;
        return -1;
    }
    cap.set(CV_CAP_PROP_POS_FRAMES, startFrame - 1 );

    // Detections
    cout << "---> Initalisation upper body detection." << endl;
    DetBody body("torso3comp.txt", -0.5);
    cout << "---> Initalisation hand detection." << endl;
    DetHand hand("hand.txt", -0.3);cout;

    // Program loop
    cout << "---> Start program loop." << endl;
    save = new SaveDetections(inputFile.c_str());
    signal(SIGINT, signal_callback_handler);
    while ( cap.get( CV_CAP_PROP_POS_AVI_RATIO )  != -1 ) {
        Mat cameraImage, textImage;

        cap >> cameraImage;
        cameraImage.copyTo(textImage);
        stringstream ss;
        ss << cap.get( CV_CAP_PROP_POS_FRAMES ) << "/" << cap.get( CV_CAP_PROP_FRAME_COUNT );
        putText(textImage, ss.str(), Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0) );

        // Run body detection
        save->newFrame(cap.get( CV_CAP_PROP_POS_FRAMES ));
        body.runDetection(cameraImage, cap.get( CV_CAP_PROP_POS_FRAMES ));

        cout << body.getSize() << " detections found" << endl;
        // Do hand detection per body detection
        for ( int n = 0; n < body.getSize(); n++ ) {
            clock_t T1, T2;
            T1 = clock();
            save->newUpperBody(body.getRect()[n]);
            ss.str("");
            ss << "Body detection: " << n+1;
            imshow( ss.str(), body.getCutouts()[n]);

            // Run hand detection
            // hand.runDetection(body.getCutouts()[n], cap.get( CV_CAP_PROP_POS_FRAMES )); // !!! TEMPERARY
            body.getCutouts()[n].copyTo( cameraImage );
            for ( int i = 0; i < hand.getSize(); i++ ) {
                save->newHand(hand.getRect()[i]);   // Save hand detections to XML file
                hand.drawResult(cameraImage, hand.getRect()[i], Scalar(255, 0, 255)); // View hand detections on body detections
            }

            HighestLikelihood likeli;
            likeli.armConnectDetection(cameraImage, hand.getRect());

            T2 = clock();
            float diff = ( (float)T2 - (float)T1 ) / CLOCKS_PER_SEC;
            save->runtime(diff);

            imshow( "Hands", cameraImage );
            waitKey(-1);
        }
        waitKey(20);
        destroyAllWindows();


        // Skip frames
        cap.set( CV_CAP_PROP_POS_FRAMES, cap.get(CV_CAP_PROP_POS_FRAMES) + skipFrames - 1 );
    }
    save->~SaveDetections();

    
    return EXIT_SUCCESS;
}
