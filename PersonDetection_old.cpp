#include "PersonDetection.h"

#define RatioTest
#define MaxCols 320
#define MaxRows 240

void detect(const Mixture & mixture, int width, int height, const HOGPyramid & pyramid,
            double threshold, double overlap, const string image,
            const string & images, vector<Detection> & detections, vector<Detection> & Facedetections, vector<Detection> & Bodydetections)
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
            // store as face detection
            Facedetections.push_back(detections[i]);
        } else {
            // store as body detection
            Bodydetections.push_back(detections[i]);
        }
    }

#endif
    /////////////////////////////////////////////////////////////
}

int PersonDetection(int framenr, Mixture & mixture, vector<UniqueDetections> & FinalFaceDetections, vector<UniqueDetections> & FinalBodyDetections, Mat CVimage)
{
    char FileName[200];
    string images("./Detections");
    int padding = 10;           // Amount of zero padding in HOG cells
    int interval = 10;          // Number of levels per octave in HOG pyramid
    double threshold = -0.55;   // Minimum detection TH
    double overlap = 0.55;      // Minimum overlap in nms
    //******************************************************//
    JPEGImage image("", CVimage);

    if(!image.empty()) {
        cout << endl << "calcultation of image " << framenr << endl;
        HOGPyramid pyramid(image, padding, padding, interval);
        // Compute the detections
        vector<Detection> detections;
        vector<Detection> Facedetections;
        vector<Detection> Bodydetections;
        detect(mixture, image.width(), image.height(), pyramid, threshold, overlap,
               FileName, images, detections, Facedetections, Bodydetections);

        // extra nms for Face detections
        for(int j = 0; j < Facedetections.size(); j++) {
            const FFLD::Rectangle & rect = Facedetections[j];
            Point P1(rect.left(), rect.top());
            Point P2(rect.right(), rect.bottom());
            Rect R1(P1, P2);
            int assigned = 0;

            for(int DetSize = 0; DetSize < FinalFaceDetections.size(); DetSize++) {
                Rect R2 = FinalFaceDetections[DetSize].detect;
                float res = CalculateOverlap(R1, R2);

                if(res > 0.5) {
                    assigned = 1;
                    FinalFaceDetections[DetSize].detect = R1;

                    if(Facedetections[j].score > FinalFaceDetections[DetSize].BestScore) {
                        FinalFaceDetections[DetSize].BestScore = Facedetections[j].score;
                        break;
                    }
                }
            }

            if(assigned == 0) {
                UniqueDetections UniqueDet;
                UniqueDet.BestScore = Facedetections[j].score;
                cout << "score = " << Facedetections[j].score << endl;
                UniqueDet.detect = R1;
                FinalFaceDetections.push_back(UniqueDet);
            }
        }

        // extra nms for Body detections
        for(int j = 0; j < Bodydetections.size(); j++) {
            const FFLD::Rectangle & rect = Bodydetections[j];
            Point P1(rect.left(), rect.top());
            Point P2(rect.right(), rect.bottom());
            Rect R1(P1, P2);
            int assigned = 0;

            for(int DetSize = 0; DetSize < FinalBodyDetections.size(); DetSize++) {
                Rect R2 = FinalBodyDetections[DetSize].detect;
                float res = CalculateOverlap(R1, R2);

                if(res > 0.5) {
                    assigned = 1;
                    FinalBodyDetections[DetSize].detect = R1;

                    if(Bodydetections[j].score > FinalBodyDetections[DetSize].BestScore) {
                        FinalBodyDetections[DetSize].BestScore = Bodydetections[j].score;
                        break;
                    }
                }
            }

            if(assigned == 0) {
                UniqueDetections UniqueDet;
                UniqueDet.BestScore = Bodydetections[j].score;
                cout << "score = " << Bodydetections[j].score << endl;
                UniqueDet.detect = R1;
                FinalBodyDetections.push_back(UniqueDet);
            }
        }

        // clear vectors
        detections.clear();
        Facedetections.clear();
        Bodydetections.clear();
    }
}

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

float CalculateOverlap(Rect CurrentDetection, Rect PreviousDetection)
{
    Rect intersection;
    float res = 0;
    // find overlapping region between GT and detection
    intersection.x = (PreviousDetection.x < CurrentDetection.x) ? CurrentDetection.x : PreviousDetection.x;
    intersection.y = (PreviousDetection.y < CurrentDetection.y) ? CurrentDetection.y : PreviousDetection.y;
    intersection.width = (PreviousDetection.x + PreviousDetection.width < CurrentDetection.x + CurrentDetection.width) ? PreviousDetection.x + PreviousDetection.width : CurrentDetection.x + CurrentDetection.width;
    intersection.width -= intersection.x;
    intersection.height = (PreviousDetection.y + PreviousDetection.height < CurrentDetection.y + CurrentDetection.height) ? PreviousDetection.y + PreviousDetection.height : CurrentDetection.y + CurrentDetection.height;
    intersection.height -= intersection.y;

    if((intersection.width <= 0) || (intersection.height <= 0)) {
        intersection = cvRect(0, 0, 0, 0);
    }

    float areaIntersect = intersection.width * intersection.height;
    float areaGT = PreviousDetection.width * PreviousDetection.height;
    float areaDetect = CurrentDetection.width * CurrentDetection.height;
    float res1 =  areaIntersect / areaGT;
    float res2 = areaIntersect / areaDetect;

    if(res1 >= 0.5 && res2 >= 0.5) {
        res = max(res1, res2);
#ifdef DebugInfo
        cout << "Detection is valid" << endl;
#endif
    } else {
        res = 0;
#ifdef DebugInfo
        cout << "Detection is invalid" << endl;
#endif
    }

    return res;
}
