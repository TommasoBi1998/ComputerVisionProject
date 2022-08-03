#include "Svm.h"

cv::Mat maskCreation(cv::String training_path) {

    // Read masks from files
    cv::Mat training_mask;
    cv::FileStorage fs(training_path, cv::FileStorage::READ);
    fs["mask"] >> training_mask;
    fs.release();

    std::vector<cv::Point> coordinates_first, coordinates_second, coordinates_third, coordinates_forth;
    int hand_counter = 1;
    int i = 0;

    cv::Mat mask_image(720, 1280, CV_8UC1, cv::Scalar(0));

    switch (hand_counter)
    {
        case 1:
            while (static_cast<int>(round(training_mask.at<float>(i,0))) != -1 && i < training_mask.rows) {
                cv::Point points;
                points = cv::Point(static_cast<int>(round(training_mask.at<float>(i,0))), static_cast<int>(round(training_mask.at<float>(i,1))));
                coordinates_first.push_back(points);
                i++;
            }
            if(!coordinates_first.empty()) {
                cv::fillPoly(mask_image, coordinates_first, 255);
            }
            hand_counter++;
            i++;
        
        case 2:
            while (static_cast<int>(round(training_mask.at<float>(i,0))) != -1 && i < training_mask.rows) {

                cv::Point points;
                points = cv::Point(static_cast<int>(round(training_mask.at<float>(i,0))), static_cast<int>(round(training_mask.at<float>(i,1))));
                coordinates_second.push_back(points);
                i++;
            }
            if(!coordinates_second.empty()) { 

                cv::fillPoly(mask_image, coordinates_second, 255);
            }
            hand_counter++;
            i++;

        case 3:
            while (static_cast<int>(round(training_mask.at<float>(i,0))) != -1 && i < training_mask.rows) {

                cv::Point points;
                points = cv::Point(static_cast<int>(round(training_mask.at<float>(i,0))), static_cast<int>(round(training_mask.at<float>(i,1))));
                coordinates_third.push_back(points);
                i++;
            }
            if(!coordinates_third.empty()) { 

                cv::fillPoly(mask_image, coordinates_third, 255);
            }
            hand_counter++;
            i++;

        case 4:
            while (static_cast<int>(round(training_mask.at<float>(i,0))) != -1 && i < training_mask.rows) {

                cv::Point points;
                points = cv::Point(static_cast<int>(round(training_mask.at<float>(i,0))), static_cast<int>(round(training_mask.at<float>(i,1))));
                coordinates_forth.push_back(points);
                i++;
            }
            if(!coordinates_forth.empty()) { 

                cv::fillPoly(mask_image, coordinates_forth, 255);
            }
            break;
    }

    return mask_image;
}

void assignLabel(std::vector<Superpixel> &superpixels, cv::String path) {

    // Create mask from file
    cv::Mat mask_image = maskCreation(path);

    cv::Mat test_mask = mask_image.clone();

    // Assign label 1 if at least 50% of the superpixel is covered by the mask
    for(Superpixel& sp: superpixels) {

        float point_count = 0.0;
        float division = 0.0;

        for(cv::Point p: sp.points) {

            if(mask_image.at<uchar>(p.y , p.x) == 255) {
                test_mask.at<uchar>(p.y , p.x) = 255;
                point_count++;
            }
        }

        division = point_count / sp.points.size();

        if(division > 0.5) {

            sp.label = 1;
            
        }
        else {
            
            sp.label = -1;
        }   
    }


}

cv::Ptr<cv::ml::TrainData> prepareTrainingDataset(cv::String dictionary_path, cv::String path_img_dict, cv::String path_img, cv::String path_yml, cv::Mat &labelImage) {

    // Folder of training yml files 
    std::vector<cv::String> fn_yml;
	cv::glob(path_yml, fn_yml, true);

    // Folder of training images
    std::vector<cv::String> fn_img;
	cv::glob(path_img, fn_img, true);

    // Create DIFT descriptor extractor
    cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SIFT::create(); 

    // Create a nearest neighbor matcher
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();

    // Create BoW descriptor extractor
    cv::BOWImgDescriptorExtractor BoWDE = cv::BOWImgDescriptorExtractor(extractor, matcher);
    
    cv::Mat vocabulary_descriptors;
    cv::FileStorage fs(dictionary_path, cv::FileStorage::READ);
    
    // If vocabulary.yml exists -> skip vocabulary creation
    if(fs.isOpened()) {

        fs["vocabulary"] >> vocabulary_descriptors;
        fs.release();

        std::cout << "-------------------------" << std::endl;
        std::cout << "Dictionary found, no need to rebuild." << std::endl;

        // Set the dictionary with the vocabulary we created in the first step
        BoWDE.setVocabulary(vocabulary_descriptors);
    } else {

        std::cout << "-------------------------" << std::endl;
        std::cout << "Dictionary not found, rebuilding..." << std::endl;
        BoWDE = buildVocabulary(path_img_dict, dictionary_path, labelImage);
    }
    
    std::cout << "-------------------------" << std::endl;
    std::cout << "Preparing training set." << std::endl;

    // Creation Containers for training histograms
    std::vector<Superpixel> superpixels;
    cv::Mat training_histogram = cv::Mat(0, BoWDE.descriptorSize(), CV_32F);
    cv::Mat labels;
    std::vector<int> superpixels_labels; 
    cv::Mat nullHistogram = cv::Mat(cv::Size(400, 1), CV_32F, cv::Scalar(0.0));

    for(size_t i = 0; i < fn_yml.size(); ++i) {
        
        // Load the image
        cv::Mat img = cv::imread(fn_img[i]);

        std::cout << "image " << i + 1 << " out of " << fn_img.size() << std::endl;

        // Segment the image
        superpixels = createSegmentation(img, labelImage, 200);

        // Compute BoW histogram for each superpixel
        computeHistogram(img, BoWDE, superpixels, labelImage);

        // Aggregate superpixels' histogram to improve performance
        //aggregateHistograms(superpixels, labelImage, img);

        // Assign label to each superpixel
        assignLabel(superpixels, fn_yml[i]);

        for(Superpixel& sp: superpixels) {
            if(sp.label == 0) { continue; }

            // Avoid null histogram
            if (countNonZero(sp.histogram != nullHistogram) == 0) { continue; }

            // Training data population
            training_histogram.push_back(sp.histogram);
            superpixels_labels.push_back(sp.label);
        }
    }

    // Prepare training dataset for SVM
    // Convert superpixels_labels into cv::Mat for svm input 
    cv::Mat(superpixels_labels).copyTo(labels);

    // Prepare a TrainData object
    cv::Ptr<cv::ml::TrainData> training_set = cv::ml::TrainData::create(training_histogram, cv::ml::ROW_SAMPLE, labels);

    return training_set;
}

void train(cv::Ptr<cv::ml::TrainData> training_set, cv::String svm_path) {

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::CHI2);

    // Train and use k = 5 fold cross validation
    std::cout << "-------------------------" << std::endl;
    std::cout << "Starded SVM training." << std::endl;
    svm->trainAuto(training_set, 5);
    std::cout << "Ended SVM training." << std::endl;

    // Save the model to file
    svm->save(svm_path); 
}