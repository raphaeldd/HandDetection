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
#include "FacedDetection.h"


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

    if(inputFile.substr(inputFile.length() - 4) != ".avi" && inputFile.substr(inputFile.length() - 4) != ".mp4") {
        cout << "No valid input file is found" << endl;
        return 1;
    }

    cout << "start frame: " << startFrame << endl;
    cout << "skip frames: " << skipFrames << endl;
    cout << "input file: " << inputFile << endl;


    // START OFFICIAL PROGRAM
    // ----------------------------------------
    // ---------- Opening image file ----------
    VideoCapture cap(inputFile);

    if(!cap.isOpened()) {
        cout << "Invalid input data" << endl;
        return -1;
    }

    cap.set(CV_CAP_PROP_POS_FRAMES, startFrame - 1);

    //  ---------- Initialize Detectors ----------
    cout << "---> Initalisation face detector." << endl;
    FaceDetection faceDetector("haarcascade_frontalface_alt.xml");
    cout << "---> Initalisation upper body detection." << endl;
    DetBody body("torso3comp.txt", -1.5);
    cout << "---> Initalisation hand detection." << endl;
    DetHand hand("hand.txt", "context-hand.txt", -0.5);

    // ---------- Program loop ----------
    cout << "---> Start program loop." << endl;
    save = new SaveDetections(inputFile.c_str());
    signal(SIGINT, signal_callback_handler);

    while(cap.get(CV_CAP_PROP_POS_FRAMES)  < cap.get(CV_CAP_PROP_FRAME_COUNT)) {
        cout << "-----> Frame " << cap.get(CV_CAP_PROP_POS_FRAMES) + 1 << "/" << cap.get(CV_CAP_PROP_FRAME_COUNT) << "." << endl;
        Mat cameraImage, textImage;
        cap >> cameraImage;
        cameraImage.copyTo(textImage);
        stringstream ss;

        // ---------- Run body detection ----------
        cout << "-----> Run upper body detection." << endl;
        save->newFrame(cap.get(CV_CAP_PROP_POS_FRAMES));
        imshow("Original frame", cameraImage);
        body.runDetection(cameraImage, cap.get(CV_CAP_PROP_POS_FRAMES));
        cout << body.getSize() << " detections found" << endl;

        // ---------- Do hand detection per body detection ----------
        for(int n = 0; n < body.getSize(); n++) {
            clock_t T = clock();
            save->newUpperBody(body.getRect()[n]);
            ss.str("");
            ss << "Body detection: " << n + 1;
            imshow(ss.str(), body.getCutouts()[n]);

            // ---------- Run face detection on upper body cut out ----------
            cout << "-----> Run Face detection for upper body detection " << n + 1 << "/" << body.getSize() << "." << endl;
            Rect face = faceDetector.detectFace(body.getCutouts()[n].clone());
            cout << "               Face found:      " << boolalpha << (face.area() > 0) << endl;


            // ---------- Run hand detection ----------
            cout << "-----> Run Hand detection on upper body detection " << n + 1 << "/" << body.getSize() << "." << endl;
            hand.runDetection(body.getCutouts()[n], cap.get(CV_CAP_PROP_POS_FRAMES), face);
            cout << "               Hand detections found:      " << hand.getLocationHand().size() << endl;
            cout << "               Context detections found:   " << hand.getLocationContext().size() << endl;
            cout << "               Arm detections found:       " << hand.getLocationArm().size() << endl;

            // ---------- Run Highest Likelihood eliminator ----------
            cout << "------> Run higest likelihood eliminator." << endl;
            HighestLikelihood likeli;
            Mat tmp = body.getCutouts()[n].clone();
            //likeli.run(&hand, face, body.getCutouts()[n].clone(), RotatedRect(Point(160, 33), Size(10, 10), 0), RotatedRect(Point(-35.6595, 195.683), Size(10, 10), 0));
            likeli.run(&hand, face, body.getCutouts()[n].clone(), RotatedRect(Point(160, 33), Size(0, 0), 0), RotatedRect(Point(-35.6595, 195.683), Size(0, 0), 0));
            cout << "               After elimination:       " << likeli.getResults().size() << endl;

            T = clock() - T;
            save->newFace(face);
            save->runtime((float)T / CLOCKS_PER_SEC);







            // ---------- View detections detHand ----------
            cout << "-----> View result detections for " << n + 1 << "/" << body.getSize() << "." << endl;
            Mat result = body.getCutouts()[n].clone();

            for(int r = 0; r < hand.getLocationHand().size(); r++) {
                hand.drawResult(result, hand.getLocationHand()[r], Scalar(255, 30, 30));
                save->newHand(hand.getLocationHand()[r]);
            }



            for(int r = 0; r < hand.getLocationContext().size(); r++) {
                hand.drawResult(result, hand.getLocationContext()[r], Scalar(30, 255, 30));
                save->newHand(hand.getLocationContext()[r]);
            }



            for(int r = 0; r < hand.getLocationArm().size(); r++) {
                hand.drawResult(result, hand.getLocationArm()[r], Scalar(30, 30, 255));
                save->newHand(hand.getLocationArm()[r]);
            }



            imshow(ss.str(), result);
            char key = waitKey(-1);

            if(key == 'q' || key == 'Q') {
                goto END;
            }
        }

        destroyAllWindows();
        // Skip frames
        cap.set(CV_CAP_PROP_POS_FRAMES, cap.get(CV_CAP_PROP_POS_FRAMES) + skipFrames - 1);
    }

END:
    save->~SaveDetections();
    cout << "---> Detection finished." << endl << endl;
    return EXIT_SUCCESS;
}
