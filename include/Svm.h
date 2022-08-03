#ifndef SVM_H
#define SVM_H

#include <opencv2/imgcodecs.hpp>

#include "SuperpixelCreation.h"
#include "Superpixel.h"

/**
 * @brief Prepares Dataset for svm training
 * 
 * @param dictionary_path Path to pretrained dictionary
 * @param path_img_dict Path to images used to create the dictionary
 * @param path_img Path of training images folder
 * @param path_yml Path of training yml files folder
 * @param labelImage Matrix which relates points with corresponding superpixel 
 * 
 * @return cv::Ptr<cv::ml::TrainData> Pointer to Training Dataset
 */
cv::Ptr<cv::ml::TrainData> prepareTrainingDataset(cv::String dictionary_path, cv::String path_img_dict, cv::String path_img, cv::String path_yml, cv::Mat &labelImage);
/**
 * @brief Trains the svm model
 * 
 * @param training_set Training set constructed by prepareTrainingDataset function
 * @param svm_path Path to pretrained svm
 */
void train(cv::Ptr<cv::ml::TrainData> training_set, cv::String svm_path);
/**
 * @brief Given a *.yml file, pre-created with Matlab, the function creates a mask that has value 255
 * if the pixel belongs to hand, value 0 otherwise
 * 
 * @param training_path Path of *.yml files' folder
 * 
 * @return cv::Mat Containing the mask
 */
cv::Mat maskCreation(cv::String training_path);
/**
 * @brief For each Superpixel, it assings value 1 if at least 50% of pixels of Superpixel belong to hand, -1 otherwise
 * 
 * @param superpixels Vector of Superpixel objects
 * @param path Path of *.yml files' folder
 */
void assignLabel(std::vector<Superpixel> &superpixels, cv::String path);

#endif // SVM_H