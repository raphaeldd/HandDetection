#include "Cluster.h"

Cluster::Cluster(double eps, int sigma)
{
    this->eps = eps;
    this->sigma = sigma;
}

bool Cluster::operator()(RotatedRect a, RotatedRect b)
{
    //return  abs(((int)a.angle + 180 - (int)b.angle) % 360 - 180) <= sigma  &&
    //cout << "       " << sqrt(pow(a.center.x - b.center.x, 2) + pow(a.center.y - b.center.y, 2)) << " <= " << (a.size.area() * pow(eps, 2)) << endl;
    return        sqrt(pow(a.center.x - b.center.x, 2) + pow(a.center.y - b.center.y, 2)) <= (a.size.area() * pow(eps, 2));
}

