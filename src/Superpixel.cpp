#include "Superpixel.h"

Superpixel::Superpixel(int id, std::vector<cv::Point> points, cv::Point center, int label, cv::Mat histogram, cv::Vec3b color) {

    this->id = id;
    this->points = points;
    this->center = center;
    this->label = label;
    this->histogram = histogram;
    this->color = color;
}