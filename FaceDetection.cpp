#include "FacedDetection.h"

FaceDetection::FaceDetection(string faceModelFile)
{
    // Face detection
    //CascadeClassifier face_cascade;

    // haarcascade_frontalface_alt.xml {
    if(!face_cascade.load(faceModelFile)) {
        cout << "Error loading haarcascade xml file." << endl;
    }
}

Rect FaceDetection::detectFace(Mat In)
{
    vector<Rect> faces;
    //face_cascade.detectMultiScale(In, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    face_cascade.detectMultiScale(In, faces, 1.1, 3);

    // Take first face
    if(faces.size() > 0) {
        this->face = faces[0];
        return faces[0];
    } else {
        cout << "       No face found" << endl;
        return Rect();
    }
}


