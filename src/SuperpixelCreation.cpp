#include "SuperpixelCreation.h"

std::vector<Superpixel> createSegmentation(cv::Mat img, cv::Mat &labelImage, int num_superpixels) {
    
    // Creation of a vector of superpixels
    std::vector<Superpixel> superpixels;

    // Mat containing the contour image
    cv::Mat contourImage;

    // Creation of superpixels
    // Blurring image and converting to Lab color space to improve segmentation performance
    cv::Mat blurred_img = img.clone();
    cv::GaussianBlur(blurred_img, img, cv::Size(3,3), 1.0);
    cv::cvtColor(blurred_img, blurred_img, cv::COLOR_BGR2Lab);
    
    //cv::Ptr<cv::ximgproc::SuperpixelSLIC> SLIC_superpixel = cv::ximgproc::createSuperpixelSLIC(img, cv::ximgproc::MSLIC, 100, 10.0);

    // Creation of SuperpixelsSEEDS object
    cv::Ptr<cv::ximgproc::SuperpixelSEEDS> SEEDS_superpixel = cv::ximgproc::createSuperpixelSEEDS(blurred_img.cols, blurred_img.rows, blurred_img.channels(), num_superpixels, 5, 5, 5, true); 

    SEEDS_superpixel->iterate(img, 30);

    // Image containing the mask of the superpixel segmentation
    SEEDS_superpixel->getLabelContourMask(contourImage);

    //SLIC_superpixel->iterate(5);
    //SLIC_superpixel->getLabelContourMask(contourImage);

    // Mat containing the segmentation labeling
    SEEDS_superpixel->getLabels(labelImage);
    //SLIC_superpixel->getLabels(labelImage);

    // Initialization of superpixels
    for(int i = 0; i < num_superpixels; ++i) {

        superpixels.push_back(Superpixel(i, std::vector<cv::Point>(), cv::Point(0,0), 0, cv::Mat(), cv::Vec3b()));
    }

    // Population of superpixels
    for(int y = 0; y < labelImage.rows; ++y) {

        for(int x = 0; x < labelImage.cols; ++x){

            // Filling the points field of each superpixel
            superpixels[labelImage.at<int>(y, x)].points.push_back(cv::Point(x, y));

            // Fill also the center field
            superpixels[labelImage.at<int>(y, x)].center += cv::Point(x, y);
        }
    }

    // Adjust centers with the mean of the points
    for(Superpixel& sp: superpixels) {

        if (sp.points.size() == 0) { continue; }

        sp.center = sp.center / static_cast<int>(sp.points.size());
    }

    return superpixels;
}

void computeHistogram(cv::Mat inputImage, cv::BOWImgDescriptorExtractor &bowDE, std::vector<Superpixel> &superpixels, cv::Mat &labelImage) {
    
    // Done for each superpixel
    for(Superpixel& sp: superpixels) {

        if (sp.points.size() < 10){ // too few points -> empty descriptor

            sp.histogram = cv::Mat(cv::Size(bowDE.descriptorSize(), 1), CV_32F, cv::Scalar(0.0));
            continue;
        }

        // Creation of bounding box
        cv::Rect box = getRectBoundBox(sp.points);
        cv::Mat sp_bounding_box = inputImage(box);

        // Bounding box too small -> skip
        if (sp_bounding_box.rows < 10 || sp_bounding_box.cols < 10) {
        
            sp.histogram = cv::Mat(cv::Size(bowDE.descriptorSize(), 1), CV_32F, cv::Scalar(0.0));
            continue;
        }

        // Detect keypoints using SIFT feature detection
        cv::Mat descr;
        std::vector<cv::KeyPoint> keypoints;

        extractSIFT(sp_bounding_box, descr, keypoints);

        // Keep only the keypoints that are actually inside the superpixel
        for(int i = static_cast<int>(keypoints.size()) - 1; i >= 0; i--) {

            // Keypoint position wrt original img
            cv::Point keypoint_img = cv::Point2d(keypoints[i].pt.x + box.x, keypoints[i].pt.y + box.y); 
    
            if (labelImage.at<int>(keypoint_img) != sp.id){

                keypoints.erase(keypoints.begin() + i);
            }
        }

        // Compute the histogram
        if(keypoints.size() == 0) {

            // Suppress if there are not enough keypoints 
            sp.histogram = cv::Mat(cv::Size(bowDE.descriptorSize(), 1), CV_32F, cv::Scalar(0.0));
        }
        else {

            bowDE.compute(sp_bounding_box, keypoints, sp.histogram);
        }
        
    }

    return;
}

void aggregateHistograms(std::vector<Superpixel> &superpixels, cv::Mat labelImage, cv::Mat img) {

    // Create the adjacecy matrix for the superpixels
    std::vector<std::vector<int>> adj_matrix = createAdjacencyMatrix(superpixels, labelImage, img);

    // New temporary superpixel vector
    std::vector<Superpixel> new_superpixels(superpixels);
    int j = 0;

    // Find the neighborhood of dimension 2 of the superpixels
    for(Superpixel& sp: superpixels) {

        // Find the first order neighbors
        std::vector<int> neighbors;
        for(int i = 0; i < adj_matrix.size(); i++) {

            if(adj_matrix[sp.id][i] > 0) {

                neighbors.push_back(i);
            }
        }

        // Expand the neighborhood with second order neighbors
        std::vector<int> snd_neighbors;

        for(int neighbor: neighbors) {
            for(int i=0; i<adj_matrix.size(); i++){

                if (adj_matrix[neighbor][i] > 0) {

                    snd_neighbors.push_back(i);
                }
            }
        }

        sort(snd_neighbors.begin(), snd_neighbors.end());
        snd_neighbors.erase( unique(snd_neighbors.begin(), snd_neighbors.end() ), snd_neighbors.end() );


        std::vector<std::pair<int, int>> two_neighborhood;
        for (int neighbor : snd_neighbors) {

            // Preprocessing for HSV distance
            cv::Vec3b color_current_sp = superpixelMeanColor(sp, img);
            cv::Vec3b color_neighbor_sp = superpixelMeanColor(superpixels[neighbor], img);

            // Population with respect to color distance and current neighbor
            two_neighborhood.push_back( {colorHSVDistance(color_current_sp, color_neighbor_sp), neighbor} );
        }

        // Sort the neighborhood based on distance
        sort(two_neighborhood.begin(), two_neighborhood.end());

        // Add the 2-closest superpixels's histogram to current one
        for(int i = 0; i < two_neighborhood.size(); ++i){

            add(new_superpixels[j].histogram, superpixels[two_neighborhood[i].second].histogram, new_superpixels[j].histogram);
        }

        // l1 normalize the fresh new histogram 
        float l1 = 0;
        for(int i = 0; i < new_superpixels[j].histogram.cols; ++i) {

            l1 += new_superpixels[j].histogram.at<float>(0, i);
        }

        if (l1 != 0.0) {

            new_superpixels[j].histogram = new_superpixels[j].histogram / l1;
        }

        j++;
    }

    // Copy the superpixels with aggregated histograms into original vector
    superpixels = new_superpixels;

    // ========================================================================================================
    // Our first approach to the nighboring problem, this approach is a little more naive
    // but we found that the use of the adjacency matrix works better.

    // Create a copy of the superpixel vector to avoid modifying histograms more than once
    // std::vector<Superpixel> superpixels_copy(superpixels);
    // int i = 0;

    // // Directions, x is the orizontal axis, y is the vertical axis (down), a point is represented as (c, r)
    // for (Superpixel& sp: superpixels) {
        
    //     std::vector<cv::Mat> neighboringHistograms;

    //     // Start from center moving throught superpixel until go out of boundary
    //     cv::Point current_point = sp.center;

    //     // Go up
    //     while (labelImage.at<int>(current_point) == labelImage.at<int>(sp.center) && current_point.y > 0) {

    //         current_point.y = current_point.y - 1;
    //     }
    //     if (labelImage.at<int>(current_point) != labelImage.at<int>(sp.center)) {

    //         // Add neighboring superpixel
    //         neighboringHistograms.push_back(superpixels[labelImage.at<int>(current_point)].histogram);
    //     }

    //     current_point = sp.center;

    //     // Go down
    //     while (labelImage.at<int>(current_point) == labelImage.at<int>(sp.center) && current_point.y < labelImage.rows - 1) {

    //             current_point.y = current_point.y + 1;
    //     }
    //     if (labelImage.at<int>(current_point) != labelImage.at<int>(sp.center)) {

    //         // Add neighboring superpixel
    //         neighboringHistograms.push_back(superpixels[labelImage.at<int>(current_point)].histogram);
    //     }

    //     current_point = sp.center;

    //     // Go left
    //     while (labelImage.at<int>(current_point) == labelImage.at<int>(sp.center) && current_point.x > 0) {

    //         current_point.x = current_point.x - 1;
    //     }

    //     if (labelImage.at<int>(current_point) != labelImage.at<int>(sp.center)) {

    //         // Add neighboring superpixel
    //         neighboringHistograms.push_back(superpixels[labelImage.at<int>(current_point)].histogram);
    //     }

    //     current_point = sp.center;

    //     // Go right
    //     while (labelImage.at<int>(current_point) == labelImage.at<int>(sp.center) && current_point.x < labelImage.cols - 1) {

    //         current_point.x = current_point.x + 1;
    //     }

    //     if (labelImage.at<int>(current_point) != labelImage.at<int>(sp.center)) {

    //         // Add neighboring superpixel
    //         neighboringHistograms.push_back(superpixels[labelImage.at<int>(current_point)].histogram);
    //     }

    //     // Add the N closest superpixels's histogram to current one
    //     for(int i = 0; i < neighboringHistograms.size(); ++i){
    //         add(superpixels_copy[i].histogram, neighboringHistograms[i], superpixels_copy[i].histogram);
    //     }
    //     // l1 normalize the fresh new histogram 
    //     float l1 = 0;
    //     for(int x = 0; x < superpixels_copy[i].histogram.cols; ++x) {

    //         l1 = l1 + superpixels_copy[i].histogram.at<float>(0, x);
    //     } 
    //     if (l1 != 0.0) {

    //         superpixels_copy[i].histogram = superpixels_copy[i].histogram / l1;
    //     }
    //     i++;
    // }

    // superpixels = superpixels_copy;
    // ========================================================================================================
    
    return;
}

cv::BOWImgDescriptorExtractor buildVocabulary(cv::String path_img, cv::String dictionary_path, cv::Mat &labelImage) {
    
    std::vector<cv::String> fn_img;
	cv::glob(path_img, fn_img, true);
    cv::Mat vocabulary_descriptors;
    std::vector<Superpixel> superpixels;

    for(size_t i = 0; i < fn_img.size(); ++i) {

        // Load the image
        cv::Mat img = cv::imread(fn_img[i]);

        std::cout << "Building vocabulary from image " << i << " out of " << fn_img.size() << std::endl;

        // Extract keypoints and descriptors with SIFT features
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> keypoints;
        extractSIFT(img, descriptors, keypoints);

        // Segment the image with superpixels
        superpixels = createSegmentation(img, labelImage, 200);

        std::vector<cv::Mat> superpixel_descriptors(superpixels.size());

        // Assing the keypoints to the relative superpixel
        for(int i = 0; i < keypoints.size(); ++i) {

            superpixel_descriptors[labelImage.at<int>(keypoints[i].pt)].push_back(descriptors.row(i));
        }

        // Calculate superpixels' descriptors
        for (cv::Mat &descriptor : superpixel_descriptors) {

            cv::Mat new_descriptors = cv::Mat(1, descriptor.size().width, descriptor.type(), cv::Scalar(0));

            // Compute the mean vector
            for (int i = 0; i < descriptor.size().width; i++) {

                int mean_col = static_cast<int>(mean(descriptor.col(i)).val[0]);
                new_descriptors.at<float>(0, i) = static_cast<float>(mean_col);
            }

            descriptor = new_descriptors.clone();
        }

        // Create the vocabulary from the superpixel descriptors
        for (cv::Mat descriptor : superpixel_descriptors) {

            if (descriptor.size() == cv::Size(0, 0)) { continue; } 

            // Add the aggregate descriptor as an entry in the vocabulary
            vocabulary_descriptors.push_back(descriptor);
        }
    }

    // Cluster the vocabulary using K-Means
    int num_clusters = 400;
    cv::BOWKMeansTrainer BoWTrain(num_clusters);
    vocabulary_descriptors = BoWTrain.cluster(vocabulary_descriptors);
    
    // Create SIFT descriptor extractor
    cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SIFT::create(); 

    // Create a nearest neighbor matcher
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();

    // Create BoW descriptor extractor
    cv::BOWImgDescriptorExtractor BoWDE = cv::BOWImgDescriptorExtractor(extractor, matcher);

    // Set the dictionary with the vocabulary we created in the first step
    BoWDE.setVocabulary(vocabulary_descriptors);

    // Saving the file
    cv::FileStorage fs1(dictionary_path, cv::FileStorage::WRITE);
    fs1 << "vocabulary" << vocabulary_descriptors;
    fs1.release();

    std::cout << "Building done." << std::endl;

    return BoWDE;

}


void handDetection(cv::String dictionary_path, cv::String svm_path, cv::String rgb_path, cv::String det_path, cv::String mask_path) {
    
    // Read bounding boxes files
    std::vector<cv::String> txt_bounding;
    cv::glob(det_path, txt_bounding, true);

    // Read masks files
    std::vector<cv::String> mask_files;
    cv::glob(mask_path, mask_files, true);

    // Read test images and save into a vector of matrix
    std::vector<cv::Mat> images;
    readImages(images, rgb_path);

    // Read the vocabulary 
    // Create the vocabulary descriptor
    cv::Mat vocabulary_descriptors;

    // Create Sift descriptor extractor
    cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SIFT::create(); 

    // Create a nearest neighbor matcher
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();

    // Create BoW descriptor extractor
    cv::BOWImgDescriptorExtractor BoWDE(extractor, matcher);

    // Read the file
    cv::FileStorage fs1(dictionary_path, cv::FileStorage::READ);
    fs1["vocabulary"] >> vocabulary_descriptors;
    fs1.release();

    // Set the dictionary with the vocabulary we created in the first step
    BoWDE.setVocabulary(vocabulary_descriptors);

    // Load the pre trained svm
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(svm_path);

    std::vector<float> IoU_vector;
    std::vector<float> accuracy_vector;

    // Create segmentation for test images
    std::vector<cv::Mat> out_images;
    for(size_t i = 0; i < images.size(); ++i) {
        
        cv::Mat labelImage;

        std::vector<Superpixel> superpixels = createSegmentation(images[i], labelImage, 450);

        computeHistogram(images[i], BoWDE, superpixels, labelImage);

        for(Superpixel& sp : superpixels) {

            float prediction = svm->predict(sp.histogram, cv::noArray());

            sp.label = static_cast<int>(prediction);
        }

        // Eliminate unwanted superpixels based on skin color
        // To solve bgr images with only gray scale values
        if (!isGrayScale(images[i])) {

            checkSkinColor(superpixels, images[i]);
        }

        bool found_one_label = false;
        for (Superpixel &sp : superpixels) {

            if(sp.label == 1) {

                found_one_label = true;
                break;
            }
        }

        if (!found_one_label) {

            std::cout << "-------------------------" << std::endl;
            std::cout << "No hands found on image... skipping to the next one." << std::endl;
        }
        else {
            // Merge adjacent superpixels into a single new superpixel
            std::vector<Superpixel> merged_superpixels;

            mergeSuperpixels(superpixels, merged_superpixels, labelImage, images[i]);

            cv::Mat detection_image = images[i].clone();

            // +++++++++++ TASK 1 +++++++++++
            // Draw bounding boxes for each superpixel and calculate IoU
            std::cout << "-------------------------" << std::endl;

            std::vector<cv::String> fn2;
            cv::glob(rgb_path, fn2, true);
            cv::String substr = fn2[i].substr(18);

            std::cout << "Detecting and segmenting hands on image " << substr << std::endl;

            IoU_vector.push_back( drawBoundingBoxAndIoU(merged_superpixels, detection_image, txt_bounding[i]) );

            // +++++++++++ TASK 2 +++++++++++
            // Color superpixels' points for segmentation visualization and calculate accuracy
            cv::Mat segmentation_image = images[i].clone();

            cv::Mat mask = cv::imread(mask_files[i]);
            accuracy_vector.push_back( fillSuperpixelsAndAccuracy(segmentation_image, merged_superpixels, mask) );

            // Concatenate the three image for better visualization
            cv::Mat final_image;
            cv::hconcat(images[i], detection_image, final_image);
            cv::hconcat(final_image, segmentation_image, final_image);

            if (final_image.rows > 500) {

                cv::resize(final_image, final_image, final_image.size() / 2);
            }

            // Show final images
            cv::imshow("Image", final_image);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

    return;
}