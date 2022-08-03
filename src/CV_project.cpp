#include "Svm.h"

int main(int argc, char** argv) {

    // Variables containing all the necessary paths
    cv::Mat labelImage;
    cv::String path_img_dict = "../../Dataset/hands_small";
    cv::String path_img = "../../Dataset/training-set/photos";
    cv::String path_yml = "../../Dataset/training-set/masks";
    cv::String dictionary_path = "../../Dataset/dictionary.yml";
    cv::String rgb_path = "../../Dataset/rgb";
    cv::String det_path = "../../Dataset/det";
    cv::String svm_path = "../../Dataset/svm.yml";
    cv::String mask_path = "../../Dataset/mask";
    
    // Try to load svm from file
    try { cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(svm_path); } 

    catch(const cv::Exception e) { // File not found..

        std::cout << "File not found! Need to retrain svm" << std::endl;

        // Build vocabulary for BoG and prepare training set for SVM
        cv::Ptr<cv::ml::TrainData> training_set = prepareTrainingDataset(dictionary_path, path_img_dict, path_img, path_yml, labelImage);

        // Train SVM
        train(training_set, svm_path);
    }

    // Detect hands in test set images
    handDetection(dictionary_path, svm_path, rgb_path, det_path, mask_path);

    return 0;
}
