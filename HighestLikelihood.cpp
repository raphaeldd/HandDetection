#include "HighestLikelihood.h"

HighestLikelihood::HighestLikelihood()
{
}

void HighestLikelihood::run(DetHand* hand, Rect face, Mat img, RotatedRect lefty, RotatedRect righty)
{
    // ---------- Merging scores and convert to relative position ----------
    vector<RotatedRect> results;
    vector<double> score;
    int j = 0;

    cout << "                   hand scores: " << endl;
    for (int i = 0; i < hand->getLocationHand().size(); i++) {
        results.push_back(hand->getLocationHand()[i]);
        score.push_back(hand->getScoreHand()[i]);
        cout << "                           Score: " << score.back() << endl;
        j++;
    }

    cout << "                   Context scores: " << endl;
    for (int i = 0; i < hand->getLocationContext().size(); i++) {
        results.push_back(hand->getLocationContext()[i]);
        score.push_back(hand->getScoreContext()[i]);
        cout << "                           Score: " << score.back() << endl;
        j++;
    }

    cout << "                   Arm scores: " << endl;
    for (int i = 0; i < hand->getLocationArm().size(); i++) {
        results.push_back(hand->getLocationArm()[i]);
        score.push_back(hand->getScoreArm()[i]);
        cout << "                           Score: " << score.back() << endl;
        j++;
    }
    cout << "           Merge size: " << results.size() << "    Hand size: " << hand->getLocationHand().size() << "    Context size: " << hand->getLocationContext().size() << "    Arm size: " << hand->getLocationArm().size() << endl;



    // ---------- Skin pixel elimination ----------
    cout << "-------> Skin eliminator." << endl;
    this->skinEliminator(results, score, .1, hand->getSkin());
    cout << "               " << results.size() << endl;

    // ---------- Size elimination ----------
    cout << "-------> Size eliminator" << endl;
    this->removeTobigHandBox(results, score, face, 0.1);
    cout << "               " << results.size() << endl;

    // ---------- Clustring ----------
    cout << "-------> Clustering detections." << endl;
    this->clustering(results, score, .1);
    cout << "               " << results.size() << endl;

    // ---------- Lower score face area ----------
    cout << "-------> Lower face hands." << endl;
    this->lowerScoreFaceHand(results, score, face, .1, -1);
    cout << "               " << results.size() << endl;


    // ---------- Change score based on prediction location hand ----------
    cout << "-------> Change score distance tracking prediction." << endl;

    vector<RotatedRect> resultsRighty = hand->absToRel(results, Point( face.x + face.width/2, face.y + face.height/2 ));;
    vector<double> scoreRighty = score;

    this->findRighty(resultsRighty, scoreRighty, righty, .7);
    this->righty = resultsRighty.at(0);
    this->rightyScore = scoreRighty.at(0);

    vector<RotatedRect> resultsLefty = hand->absToRel(results, Point( face.x + face.width/2, face.y + face.height/2 ));;
    vector<double> scoreLefty = score;
    this->findLefty(resultsLefty, scoreLefty, lefty, .7);
    this->lefty = resultsLefty.at(0);
    this->leftyScore = scoreLefty.at(0);



    // ---------- TEMPORARY VIEW RESULTS ----------
    for ( int i = 0; i < results.size(); i++ ) {
        hand->drawResult(img, results.at(i), Scalar((1 + score.at(i)) * 100, (1 + score.at(i)) * 100, (1 + score.at(i)) * 100));
        cout << "       hand: " << results.at(i).center.x << ", " << results.at(i).center.y;
        cout << " \t\trotation: " << results.at(i).angle;
        cout << " \t\tScore: " << score.at(i) << endl;
    }
    hand->drawResult(img, hand->relToAbs(resultsRighty, Point( face.x + face.width/2, face.y + face.height/2 )).at(0), Scalar(255, 10, 10));
    hand->drawResult(img, hand->relToAbs(resultsLefty, Point( face.x + face.width/2, face.y + face.height/2 )).at(0), Scalar(10, 10, 255));

    imshow("Result", img);

    this->results = results;

}

vector<RotatedRect> HighestLikelihood::getResults()
{
    return this->results;
}

void HighestLikelihood::clustering(vector<RotatedRect>& hand, vector<double>& score, double eps) {

    vector<RotatedRect> cluster;
    vector<double> scoreCluster;
    Cluster rlike(eps, 30);
    vector<int> label;
    int labelSize = partition(hand, label , rlike);

    cluster.assign(labelSize, RotatedRect(Point2f(0, 0), Size2f(0, 0), 0));
    scoreCluster.assign(labelSize, 0);
    vector<int> Sum(labelSize, 0);
    vector<pair<double, double> > angle(labelSize, make_pair(0, 0));

    for (int i = 0; i < hand.size(); i++) {
        int n = label[i];

        // Simple mean caclulation
        angle[n].first += cos(hand[i].angle * CV_PI / 180) * score[i];
        angle[n].second += sin(hand[i].angle * CV_PI / 180) * score[i];
        cluster[n].center.x += hand[i].center.x;
        cluster[n].center.y += hand[i].center.y;
        cluster[n].size.height += hand[i].size.height;
        cluster[n].size.width += hand[i].size.width;
        scoreCluster[n] += score[n];
        Sum[n]++;

    }

    for(int n = 0; n < labelSize; n++) {
        cluster[n].angle = atan2(angle[n].second, angle[n].first) * 180 / CV_PI;
        cluster[n].center.x = cluster[n].center.x / Sum[n];
        cluster[n].center.y = cluster[n].center.y / Sum[n];
        cluster[n].size.height = cluster[n].size.height / Sum[n];
        cluster[n].size.width = cluster[n].size.width / Sum[n];
    }
    hand.clear();
    score.clear();

    hand = cluster;
    score = scoreCluster;
}

void HighestLikelihood::skinEliminator(vector<RotatedRect>& hand, vector<double>& score, double TH, Mat skin)
{
    for(int i = 0; i < hand.size(); ) {
        Mat M, rotated, cropped;
        float angle = hand.at(i).angle;
        Size hand_size = hand.at(i).size;

        // Make cutout of ratated image (skin)
        if(hand.at(i).angle < -45) {
            angle += 90;
            swap(hand_size.width, hand_size.height);
        }

        M = getRotationMatrix2D(hand.at(i).center, angle, 1.0);
        warpAffine(skin, rotated, M, skin.size(), INTER_CUBIC);
        getRectSubPix(rotated, hand_size, hand.at(i).center, cropped);

        // Remove detection based on non zero pixels vs area detection
        if ( (double)countNonZero(cropped) / (double)hand.at(i).size.area() < TH) {
            hand.erase(hand.begin() + i);
            score.erase(score.begin() + i);
        } else {
            i++;
        }
    }
}

void HighestLikelihood::removeTobigHandBox(vector<RotatedRect>& hand, vector<double>& score, Rect face, double eps)
{
    for(int i = 0; i < hand.size(); i++) {
        if( hand.at(i).size.area() > (face.size().area() * (1 + eps))) {
            hand.erase(hand.begin() + i);
            score.erase(score.begin() + i);
        }
    }
}

void HighestLikelihood::lowerScoreFaceHand(vector<RotatedRect>& hand, vector<double>& score, Rect face, double eps, int lower) {
    face.x -= face.width * eps;
    face.y -= face.height * eps;
    face.width += face.width * eps * 2;
    face.height += face.height * eps * 2;

    for ( int i = 0; i < hand.size(); i++ ) {
        if (face.contains(hand[i].center)) {
            score[i] -= lower;
        }
    }
}


void HighestLikelihood::closestPoint(vector<RotatedRect>& hand, vector<double>& score, Point predictedPoint) {

    //      calc max min dst
    float min = numeric_limits<float>::max(), max = numeric_limits<float>::min();
    for (int i = 0; i < hand.size(); i++) {
        float tmp = this->dstCalc(hand.at(i).center, predictedPoint);
        if (tmp > max) max = tmp;
        if (tmp < min) min = tmp;
    }

    //      Normalise for score
    float diff = max - min;
    //vector<float> scoreDist;
    for (int i = 0; i < hand.size(); i++) {
        //scoreDist.push_back(-((this->dstCalc(hand.at(i).center, predictedPoint)-min)/diff-1));
        score.at(i) += -((this->dstCalc(hand.at(i).center, predictedPoint)-min)/diff-1) * 10;
    }



}

void HighestLikelihood::findRighty(vector<RotatedRect>& resultsRighty, vector<double>& scoreRighty, RotatedRect predictedPoint, double eps) {

    if ( predictedPoint.size.area() > 0 ) {
        Mat plot = Mat::zeros( 200, resultsRighty.size() * 10, CV_8UC3);

        // There is a prediction
        this->closestPoint(resultsRighty, scoreRighty, predictedPoint.center);

        double max = 0, min = numeric_limits<double>::max();
        for (int i = 0; i < resultsRighty.size(); i++ ){
            if ( max < scoreRighty.at(i) ) {
                max = scoreRighty.at(i);
            }
            if ( min > scoreRighty.at(i) ) {
                min = scoreRighty.at(i);
            }
            rectangle(plot, Rect(i * 10, 0, 10, scoreRighty.at(i)*10), Scalar(10, 255, 10), CV_FILLED);
        }
        // cluster score above eps based on max en minimals
        line(plot, Point(0, (min + (max * eps)) * 10), Point(plot.cols, (min + (max * eps)) * 10), Scalar(20, 20, 255));
        for ( int i = 0; i < resultsRighty.size(); i++ ) {
            if (scoreRighty.at(i) - min < max * eps) {
                scoreRighty.erase(scoreRighty.begin()+i);
                resultsRighty.erase(resultsRighty.begin()+i);
                i--;
            }
        }

        // Give prediction more wait in clustering
        resultsRighty.push_back(predictedPoint);
        scoreRighty.push_back(0);
        resultsRighty.push_back(predictedPoint);
        scoreRighty.push_back(0);

        this->clustering(resultsRighty, scoreRighty, 1);
        cout << "                               Righty size: " <<  resultsRighty.size() << endl;

        flip(plot, plot, 0);
        imshow("Righty plot", plot);
        imwrite("RightyPlot.png", plot);
    } else {
        double max = numeric_limits<double>::min();
        int adrMax = numeric_limits<int>::max();
        // if no prediction (first run) give score based on side from face and take higest score hand on that side
        for ( int i = 0; i < resultsRighty.size(); i++ ) {
            if (resultsRighty.at(i).center.x < 0 && max < scoreRighty.at(i) ) {
                max = scoreRighty.at(i);
                adrMax = i;
            }
        }
        if ( adrMax < resultsRighty.size() ) {
            resultsRighty.at(0) = resultsRighty.at(adrMax);
            scoreRighty.at(0) = scoreRighty.at(adrMax);
            resultsRighty.erase(resultsRighty.begin() + 1, resultsRighty.end());
            scoreRighty.erase(scoreRighty.begin() + 1, scoreRighty.end());
        } else {
            resultsRighty.clear();
            resultsRighty.push_back(RotatedRect());
            scoreRighty.clear();
            scoreRighty.push_back(0);
        }
    }
}

void HighestLikelihood::findLefty(vector<RotatedRect>& resultsLefty, vector<double>& scoreLefty, RotatedRect predictedPoint, double eps) {

    if ( predictedPoint.size.area() > 0 ) {
        Mat plot = Mat::zeros( 200, resultsLefty.size() * 10, CV_8UC3);

        // There is a prediction
        this->closestPoint(resultsLefty, scoreLefty, predictedPoint.center);

        double max = 0, min = numeric_limits<double>::max();
        for (int i = 0; i < resultsLefty.size(); i++ ){
            if ( max < scoreLefty.at(i) ) {
                max = scoreLefty.at(i);
            }
            if ( min > scoreLefty.at(i) ) {
                min = scoreLefty.at(i);
            }
            rectangle(plot, Rect(i * 10, 0, 10, scoreLefty.at(i)*10), Scalar(10, 255, 10), CV_FILLED);
        }
        // cluster score above eps based on max en minimals
        line(plot, Point(0, (min + (max * eps)) * 10), Point(plot.cols, (min + (max * eps)) * 10), Scalar(20, 20, 255));
        for ( int i = 0; i < resultsLefty.size(); i++ ) {
            if (scoreLefty.at(i) - min < max * eps) {
                scoreLefty.erase(scoreLefty.begin()+i);
                resultsLefty.erase(resultsLefty.begin()+i);
                i--;
            }
        }

        // Give prediction more wait in clustering
        resultsLefty.push_back(predictedPoint);
        scoreLefty.push_back(0);
        resultsLefty.push_back(predictedPoint);
        scoreLefty.push_back(0);

        this->clustering(resultsLefty, scoreLefty, 1);
        cout << "                               Righty size: " <<  resultsLefty.size() << endl;

        flip(plot, plot, 0);
        imshow("Lefty plot", plot);
        imwrite("LeftyPlot.png", plot);


    } else {
        double max = numeric_limits<double>::min();
        int adrMax = numeric_limits<int>::max();
        // if no prediction (first run) give score based on side from face and take higest score hand on that side
        for ( int i = 0; i < resultsLefty.size(); i++ ) {
            if (resultsLefty.at(i).center.x > 0 && max < scoreLefty.at(i) ) {
                max = scoreLefty.at(i);
                adrMax = i;
            }
        }
        if ( adrMax < scoreLefty.size() ) {
            resultsLefty.at(0) = resultsLefty.at(adrMax);
            resultsLefty.erase(resultsLefty.begin() + 1, resultsLefty.end());
            scoreLefty.at(0) = scoreLefty.at(adrMax);
            scoreLefty.erase(scoreLefty.begin() + 1, scoreLefty.end());
        } else {
            resultsLefty.clear();
            resultsLefty.push_back(RotatedRect());
            scoreLefty.clear();
            scoreLefty.push_back(0);
        }
    }
}

RotatedRect toRelative(RotatedRect hand, Rect face) {

}

float HighestLikelihood::dstCalc( Point pt1, Point pt2 ) {
    return ( sqrt( pow( float(pt1.x - pt2.x), 2) + pow( float(pt1.y - pt2.y), 2) ) );
}


