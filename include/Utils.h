#ifndef UTILS_H
#define UTILS_H

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/xfeatures2d/nonfree.hpp>

#include <fstream>
#include <iostream>

#include "Superpixel.h"

/**
 * @brief Creates a Rect object from the vector of points
 * 
 * @param points Vector of points
 * 
 * @return cv::Rect bounding box
 */
cv::Rect getRectBoundBox(std::vector<cv::Point> points);
/**
 * @brief Get SIFT features from image
 * 
 * @param input Input image
 * @param descriptors Matrix containing SIFT features descriptors
 * @param keypoints Vector containing SIFT features keypoints
 */
void extractSIFT(cv::Mat &input, cv::Mat &descriptors, std::vector<cv::KeyPoint> &keypoints);
/**
 * @brief Read images from a given path
 * 
 * @param images Vector of images belonging to a given path
 * @param path Path of the images' folder
 */
void readImages(std::vector<cv::Mat> &images, cv::String &path);
/**
 * @brief Extract and compute SIFT features from images
 * 
 * @param images Input images
 * @param descriptors Set of descriptors for keypoints
 * @param keypoints Set of SIFT keypoints
 */
void keypointsDescriptorsExtraction(std::vector<cv::Mat> &images, std::vector<cv::Mat> &descriptors, std::vector<std::vector<cv::KeyPoint>> &keypoints);
/**
 * @brief Fills superpixels that have predicted label equal to 1 and computes the segmentation accuracy
 *  
 * @param image Input image 
 * @param superpixels Vector of Superpixel objects
 * @param det_mask Mask to compare the segmentation against
 * 
 * @return float Accuracy value
 */
float fillSuperpixelsAndAccuracy(cv::Mat &image, std::vector<Superpixel> &superpixels, cv::Mat det_mask);
/**
 * @brief Create a Adjacency Matrix object
 * 
 * @param superpixels Vector of Superpixel objects
 * @param labelImage Matrix which relates points with corresponding superpixel
 * @param img Input image
 * 
 * @return std::vector<std::vector<int>> Adjacency matrix
 */
std::vector<std::vector<int>> createAdjacencyMatrix(std::vector<Superpixel> &superpixels, cv::Mat labelImage, cv::Mat img);
/**
 * @brief Computes HSV distance from two superpixels' color
 * 
 * @param color1 Color of first superpixel
 * @param color2 Color of second superpixel
 * 
 * @return float Distance
 */
float colorHSVDistance(cv::Vec3b color1, cv::Vec3b color2);
/**
 * @brief Calculates mean color of superpixel's points
 * 
 * @param sp Superpixel object
 * @param img Input image
 * 
 * @return cv::Vec3b Mean color
 */
cv::Vec3b superpixelMeanColor(Superpixel sp, cv::Mat img);
/**
 * @brief Applies a set of thresholds to eliminate false positive superpixels based on skin color
 * 
 * @param superpixels Vector of Superpixel objects
 * @param img Input image
 */
void checkSkinColor(std::vector<Superpixel> &superpixels, cv::Mat img);
/**
 * @brief Merges adjacent superpixels based on graph theory
 * 
 * @param superpixels Vector of Superpixel objects
 * @param merged_superpixels Vector of merged Superpixel objects
 * @param labelImage Matrix which relates points with corresponding superpixel
 * @param inputImage Input image
 */
void mergeSuperpixels(std::vector<Superpixel> &superpixels, std::vector<Superpixel> &merged_superpixels, cv::Mat labelImage, cv::Mat inputImage);
/**
 * @brief Draws bounding box around superpixels that have been labeled as 1 and computes the Intersection over Union
 * 
 * @param merged_superpixels Vector of merged Superpixel objects
 * @param img Input image
 * @param fn Path of txt file of "det" folder
 * 
 * @return float IoU
 */
float drawBoundingBoxAndIoU(std::vector<Superpixel> &merged_superpixels, cv::Mat &img, cv::String fn);
/**
 * @brief Checks if the image is a 3 channel gray scale image
 * 
 * @param img Input image
 * 
 * @return true 
 * @return false 
 */
bool isGrayScale(cv::Mat img);

#endif // UTILS_H