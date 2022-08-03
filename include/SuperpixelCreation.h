#ifndef SUPERPIXELCREATION_H
#define SUPERPIXELCREATION_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/xfeatures2d/nonfree.hpp>

#include <opencv2/ml.hpp>

//#include <opencv2/ximgproc/slic.hpp>
#include <opencv2/ximgproc/seeds.hpp>

#include "Superpixel.h"
#include "Utils.h"

/**
 * @brief Segments the image using SEEDS superpixel, initializes the superpixels with default values,
 * populates with points and computes centers.
 *
 * @param img Image to segment
 * @param labelImage Matrix which relates points with corresponding superpixel 
 * @param num_superpixels Number of superpixels
 * @return std::vector<Superpixel> Vector containing Superpixel objects
 */
std::vector<Superpixel> createSegmentation(cv::Mat img, cv::Mat &labelImage, int num_superpixels);
/**
 * @brief Creates a BoW trained vocabulary, useful to quantize training image features
 * 
 * @param path_img Path of training images folder
 * @param labelImage Matrix which relates points with corresponding superpixel 
 * @return cv::BOWImgDescriptorExtractor Bag of Words classifier
 */
cv::BOWImgDescriptorExtractor buildVocabulary(cv::String path_img, cv::String dictionary_path, cv::Mat &labelImage);
/**
 * @brief Creates bounding box for each superpixel, calculates associated image, computes keypoints (keeping only 
 * superpixel's keypoints) and computes histogram related to bounding box
 * 
 * @param inputImage Input image
 * @param bowDE Trained Bag of Words descriptor extractor
 * @param superpixels Vector of Superpixel objects
 * @param labelImage Matrix which relates points with corresponding superpixel
 */
void computeHistogram(cv::Mat inputImage, cv::BOWImgDescriptorExtractor &bowDE, std::vector<Superpixel> &superpixels, cv::Mat &labelImage);
/**
 * @brief Aggregation of neighboring superpixels' histograms in order to obtain a better classification
 * 
 * @param superpixels Vector of Superpixel objects 
 * @param labelImage Matrix which relates points with corresponding superpixel
 * @param img Input image
 */
void aggregateHistograms(std::vector<Superpixel> &superpixels, cv::Mat labelImage, cv::Mat img);

/**
 * @brief Perform hand detection and segmentation using an SVM classifier trained on BoW histograms
 * 
 * @param dictionary_path Path to pretrained dictionary
 * @param svm_path Path to pretrained svm
 * @param rgb_path Path to test set images
 * @param det_path Path to txt files containing the ground truth bounding boxes
 * @param mask_path Path to ground truth mask for segmentation
 */
void handDetection(cv::String dictionary_path, cv::String svm_path, cv::String rgb_path, cv::String det_path, cv::String mask_path);

#endif // SUPERPIXELCREATION_H