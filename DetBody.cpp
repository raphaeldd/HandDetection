#include "DetBody.h"


DetBody::DetBody(String modelBody, double threshold)
{
    // initialization: open mixtureBody + initialize patchwork class + calculate filter transform
    int res = Init(mixtureBody, modelBody);

    if(res == -1) {
        exit(1);
    }

    this->TH = threshold;  // -0.5
}


void DetBody::runDetection(Mat image , int frameNumber)
{
    Mat detImage;
    image.copyTo(detImage);
    this->detections.clear();
    this->pos.clear();
    this->FinalDetections.clear();
    this->FinalTorsoDetections.clear();
    this->FinalUpperBodyDetections.clear();
    PersonDetection(frameNumber, mixtureBody, FinalDetections, FinalUpperBodyDetections, FinalTorsoDetections, detImage, TH);

    // Put detections in detection vectors
    for(unsigned int k = 0; k < FinalTorsoDetections.size(); k++) {
        this->pos.push_back(FinalTorsoDetections[k].detect);
    }

    for(unsigned int k = 0; k < FinalUpperBodyDetections.size(); k++) {
        this->pos.push_back(FinalUpperBodyDetections[k].detect);
    }

    for(unsigned int k = 0; k < FinalDetections.size(); k++) {
        this->pos.push_back(FinalDetections[k].detect);
    }

    groupRectangles(this->pos, 1, 1000);
    cout << "       detections: " << this->pos.size() << endl;

    for(unsigned int n = 0; n < this->pos.size(); n++) {
        // Stretching bounding box
        this->pos[n].x = this->pos[n].x - this->pos[n].width * 0.8;
        this->pos[n].width = (this->pos[n].width * 0.8 * 2) + this->pos[n].width;
        this->pos[n].height = this->pos[n].height * 1.2;

        // Secure an in image coutout
        if(this->pos[n].x < 0) {
            this->pos[n].width = this->pos[n].width + this->pos[n].x;
            this->pos[n].x = 0;
        }

        if(this->pos[n].x + this->pos[n].width > image.cols - this->pos[n].x)  {
            this->pos[n].width = image.cols - this->pos[n].x;
        }

        if(this->pos[n].y + this->pos[n].height > image.rows - this->pos[n].y)  {
            this->pos[n].height = image.rows - this->pos[n].y;
        }

        // Cutout
        this->detections.push_back(image(this->pos[n]).clone());
    }
}

vector<Mat> DetBody::getCutouts()
{
    return this->detections;
}

vector<Rect> DetBody::getRect()
{
    return this->pos;
}

int DetBody::getSize()
{
    return this->detections.size();
}
