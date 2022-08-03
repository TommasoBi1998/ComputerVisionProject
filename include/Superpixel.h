#ifndef SUPERPIXEL_H
#define SUPERPIXEL_H

#include <opencv2/core.hpp>

class Superpixel {
    public:

        /**
        * @param id Identifier of superpixel 
        * @param points Positions of superpixel's points
        * @param center Center of superpixel
        * @param label Label for the classification of superpixel
        * @param histogram A matrix representing the histogram of the superpixel, useful for svm classification
        * @param color The color of the superpixel
        */
        Superpixel(int id, std::vector<cv::Point> points, cv::Point center, int label, cv::Mat histogram, cv::Vec3b color);

        int id;
        std::vector<cv::Point> points;
        cv::Point center;
        int label;
        cv::Mat histogram;
        cv::Vec3b color;
};

#endif // SUPERPIXEL_H