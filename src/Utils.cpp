#include "Utils.h"

cv::Rect getRectBoundBox(std::vector<cv::Point> points) {

    int max_x, max_y, min_x, min_y;
    max_x = max_y = 0;
    min_x = min_y = INT_MAX;
    
    // Compute max and min position of the superpixel
    for(cv::Point p: points){

        max_x = cv::max(max_x, p.x);
        max_y = cv::max(max_y, p.y);
        min_x = cv::min(min_x, p.x);
        min_y = cv::min(min_y, p.y);
    }

    cv::Point top_left = cv::Point(min_x, min_y); 
    cv::Point bottom_right = cv::Point(max_x, max_y);

    return cv::Rect(top_left, bottom_right);
}

float drawBoundingBoxAndIoU(std::vector<Superpixel> &merged_superpixels, cv::Mat &img, cv::String fn) {

    cv::Mat our_mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8U);

    for (Superpixel &sp : merged_superpixels) {

        // Create bounding box
        cv::Rect boundingBox = getRectBoundBox(sp.points);

        // Draw rectangle onto the image
        cv::rectangle(img, boundingBox.tl(), boundingBox.br(), sp.color, 4, cv::LINE_8);

        // Populate mask for IoU calculation
        for (int i = boundingBox.tl().y; i < boundingBox.tl().y + boundingBox.height; ++i) {

            for (int j = boundingBox.tl().x; j < boundingBox.tl().x + boundingBox.width; ++j) {

                our_mask.at<uchar>(i, j) = 255;
            }
        }
    }

    // Import from det folder
    std::ifstream istrm;
    std::vector<std::vector<int>> det;
    int x;
    
    istrm.open(fn, std::ios::in);

    if (!istrm.is_open()) {

        std::cout << "Unable to open file";
        exit(1); // terminate with error
    }

    int row = 0;
    int col = 0;
    if (istrm.is_open()) {

        std::vector<int> temp;
        while (istrm >> x) {
            col++;
            temp.push_back(x);
             
            if(col % 4 == 0) {
                row++;

                det.push_back(temp);

                temp.erase(temp.begin(), temp.end());
            }
        }
    }
    
    istrm.close();

    // Populate mask with the pixels of the hands given in the dataset
    cv::Mat mask_det = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8U);

    for (int i = 0; i < det.size(); ++i) {

        cv::Rect rect = cv::Rect(cv::Point(det[i][0], det[i][1]), cv::Point(det[i][0] + det[i][2], det[i][1] + det[i][3]));

        for (int j = rect.tl().y; j < rect.tl().y + det[i][3]; ++j) {
            for (int k = rect.tl().x; k < rect.tl().x + det[i][2]; ++k) {

                mask_det.at<uchar>(j, k) = 255;
            }
        }

    }

    // Calculate intersection
    cv::Mat intersection_mask = our_mask & mask_det;

    // Calculate union
    cv::Mat union_mask = our_mask | mask_det;

    float intersection_count = static_cast<float>(cv::countNonZero(intersection_mask));

    float union_count = static_cast<float>(cv::countNonZero(union_mask));

    // Calculate IoU
    float IoU = intersection_count / union_count;

    std::cout << "IoU = " << IoU << std::endl;

    return IoU;
}

bool isGrayScale(cv::Mat img) {
    
    int count = 0;
    for ( int i = 0; i < img.rows; ++i) {
        for ( int j = 0; j < img.cols; ++j) {

            if (static_cast<int>(img.at<cv::Vec3b>(i, j)[0]) == static_cast<int>(img.at<cv::Vec3b>(i, j)[1]) && static_cast<int>(img.at<cv::Vec3b>(i, j)[1]) == static_cast<int>(img.at<cv::Vec3b>(i, j)[2])) {

                count++;
            }
        }
    }

    return (count == img.rows * img.cols);
}

void extractSIFT(cv::Mat &input, cv::Mat &descriptors, std::vector<cv::KeyPoint> &keypoints) {

    //extract SIFT features and draw the keypoints
    cv::Ptr<cv::SIFT> points = cv::SIFT::create(0, 3, 0.01, 10.0, 1.6);
    points->detect(input, keypoints);
    points->compute(input, keypoints, descriptors);

    // Draw the keypoints in the image
    // cv::drawKeypoints(input, keypoints, input);

    return;
}

void readImages(std::vector<cv::Mat> &images, cv::String &path) {

    std::vector<cv::String> fn;
	cv::glob(path, fn, true);

    for(size_t i = 0; i < fn.size(); ++i) {

        // Load the image
        cv::Mat img = cv::imread(fn[i]);

        // Verify that the image is non-empty
        if ( img.empty() ) {

            std::cout << fn[i] << " is invalid!" << std::endl; // invalid image, skip it.
            continue;
        }
        
        images.push_back(img);
    }

    return;
}

float fillSuperpixelsAndAccuracy(cv::Mat &image, std::vector<Superpixel> &superpixels, cv::Mat det_mask) {

    // Convert to gray scale to create binary mask
    cv::cvtColor(det_mask, det_mask, cv::COLOR_BGR2GRAY);

    // Color superpixels in the image
    for(Superpixel& sp: superpixels) {
        if (sp.label == 1) {

            for (cv::Point p: sp.points) {

                image.at<cv::Vec3b>(p) = sp.color;
            }
        }
    }

    // Binary mask containing our segmentation
    cv::Mat our_mask = cv::Mat::zeros(image.rows, image.cols, CV_8U);

    for (Superpixel &sp : superpixels) {

        for (cv::Point pt : sp.points) {

            our_mask.at<uchar>(pt) = 255;
        }
    }

    // True positive mask
    int tp = cv::countNonZero(det_mask & our_mask);

    // True negative mask
    int tn = cv::countNonZero((255 - our_mask) & (255 - det_mask));

    // Calculate accuracy
    float accuracy = ( static_cast<float>(tp) + static_cast<float>(tn) ) / (image.rows * image.cols);

    std::cout << "Accuracy = " << accuracy << std::endl;

    return accuracy;
}

std::vector<std::vector<int>> createAdjacencyMatrix(std::vector<Superpixel> &superpixels, cv::Mat labelImage, cv::Mat img) {

    std::vector<std::vector<int>> adj_matrix;
    adj_matrix.resize(superpixels.size());
    for(std::vector<int>& row: adj_matrix) row.resize(superpixels.size());

    for(Superpixel& sp: superpixels){

        // Find a neighbor in every direction
        cv::Point dirs[] = {cv::Point( 0, 1), cv::Point(1, 0), cv::Point( 0, -1), cv::Point(-1, 0)};

        for(cv::Point p: sp.points) for(cv::Point dir: dirs){

            if ((p + dir).x < 0 || (p + dir).x > img.cols-1 || (p + dir).y < 0 || (p + dir).y > img.rows-1) { continue; }
   
            // Populate adjacency matrix
            if (labelImage.at<int>(p + dir) != sp.id){

                adj_matrix[sp.id][labelImage.at<int>(p + dir)]++;
                adj_matrix[labelImage.at<int>(p + dir)][sp.id]++;
            }
        }
    }

    return adj_matrix;
}

cv::Vec3b superpixelMeanColor(Superpixel sp, cv::Mat img) {

    // Three variables, one for each channel
    cv::Vec3b colors = cv::Vec3b(0,0,0);    
    
    for( cv::Point &pt : sp.points) {
        
        // Calculate the sum of colors of superpixel
        colors[0] += img.at<cv::Vec3b>(pt)[0];
        colors[1] += img.at<cv::Vec3b>(pt)[1]; 
        colors[2] += img.at<cv::Vec3b>(pt)[2]; 
    }

    // Calculate the mean of color
    int num_points = static_cast<int>(sp.points.size());
    colors[0] /= num_points;
    colors[1] /= num_points; 
    colors[2] /= num_points;

    return colors;
}

float colorHSVDistance(cv::Vec3b color1, cv::Vec3b color2) {
     
    cv::Mat3f color_first_mat, color_second_mat;

    cv::Mat3b(color1).convertTo(color_first_mat, CV_32FC3);
    cv::Mat3b(color2).convertTo(color_second_mat, CV_32FC3);
    
    cvtColor(color_first_mat, color_first_mat, cv::COLOR_BGR2HSV);
    cvtColor(color_second_mat, color_second_mat, cv::COLOR_BGR2HSV);


    float dist = static_cast<float>(sqrt( pow(color_first_mat[0] - color_second_mat[0], 2) + pow(color_first_mat[1] - color_second_mat[1], 2) + pow(color_first_mat[2] - color_second_mat[2], 2)));

    return dist;
}

void checkSkinColor(std::vector<Superpixel> &superpixels, cv::Mat img) {

    cv::Mat hsv_image;
    cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::Mat bgra;
    cv::cvtColor(img, bgra, cv::COLOR_BGR2BGRA);

    for(int i = 0; i < mask.rows; ++i) {

        for(int j = 0; j < mask.cols; ++j) {

            // Set of boolean for thresholds
            bool c1 = static_cast<int>(bgra.at<cv::Vec4b>(i,j)[0]) > 20;
            bool c2 = static_cast<int>(bgra.at<cv::Vec4b>(i,j)[1]) > 40;
            bool c3 = static_cast<int>(bgra.at<cv::Vec4b>(i,j)[2]) > 95;
            bool c4 = static_cast<int>(bgra.at<cv::Vec4b>(i,j)[2]) > static_cast<int>(bgra.at<cv::Vec4b>(i,j)[1]);
            bool c5 = static_cast<int>(bgra.at<cv::Vec4b>(i,j)[2]) > static_cast<int>(bgra.at<cv::Vec4b>(i,j)[0]);
            bool c6 = std::abs(static_cast<int>(bgra.at<cv::Vec4b>(i,j)[2]) - static_cast<int>(bgra.at<cv::Vec4b>(i,j)[1])) > 15;
            bool c7 = static_cast<int>(bgra.at<cv::Vec4b>(i,j)[3]) > 15;
            bool c8 = (static_cast<int>(hsv_image.at<cv::Vec3b>(i,j)[0]) >= 0 && static_cast<int>(hsv_image.at<cv::Vec3b>(i,j)[0]) <= 13) || (static_cast<int>(hsv_image.at<cv::Vec3b>(i,j)[0]) >= 40 && static_cast<int>(hsv_image.at<cv::Vec3b>(i,j)[0]) <= 50);
            bool c9 = static_cast<int>(hsv_image.at<cv::Vec3b>(i,j)[1]) > 58  && static_cast<int>(hsv_image.at<cv::Vec3b>(i,j)[1]) < 178;
            

            if( c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9) {

                mask.at<uchar>(i,j) = 255;  
            }
        }
    }

    for(Superpixel &sp : superpixels) {

        float point_count = 0.0;
        float division = 0.0;

        // Assign -1 to superpixels that have a percentage of skin minor of 0.5
        for(cv::Point p: sp.points) {

            if(mask.at<uchar>(p.y , p.x) == 255) {
                point_count++;
            }
        }

        division = point_count / sp.points.size();

        if(division < 0.5) {

            sp.label = -1;
            
        }
    }
    
    return;
}

void mergeSuperpixels(std::vector<Superpixel> &superpixels, std::vector<Superpixel> &merged_superpixels, cv::Mat labelImage, cv::Mat inputImage) {

    // Create adjacency matrix
    std::vector<std::vector<int>> adj_matr = createAdjacencyMatrix(superpixels, labelImage, inputImage);

    // Vector initialization with empty superpixel with label 1 and random color
    for (int i = 0; i < superpixels.size(); ++i) {

        if (superpixels[i].label == 1) {

            // To color we use three randomic numbers
            int b_random_color = rand()%256;
            int g_random_color = rand()%256;
            int r_random_color = rand()%256;

            merged_superpixels.push_back(Superpixel(superpixels[i].id, std::vector<cv::Point>(), cv::Point(0,0), 1, cv::Mat(), cv::Vec3b(b_random_color, g_random_color, r_random_color)));
        }
    }

    // Merging algorithm as explained in the paper
    std::vector<std::vector<int>> merged_id(merged_superpixels.size());
    int z = 0;

    for (Superpixel &sp: superpixels) {
        
        if(sp.label == -1) {

            continue;
        }

        if(sp.label == 1) {

            merged_id[z].push_back(sp.id);

            for(int i = 0; i < adj_matr[sp.id].size(); ++i) {

                if((adj_matr[sp.id][i] > 0) && (superpixels[i].label == 1)) {

                    merged_id[z].push_back(i);
                }
            }
            z++;
        }
        
    }

    if (merged_id.size() == 0) {

        merged_superpixels = superpixels;
        return;
    }
    
    for(int id = 0; id < merged_id.size() - 1; ++id) {

        for(int current = 0; current < merged_id[id].size(); ++current) {

            int find = 0;
            for(int i = id + 1; i < merged_id.size(); ++i) {

                for(int j = 0; j < merged_id[i].size(); ++j) {

                    if(merged_id[id][current] == merged_id[i][j]) {
                        
                        find = 1;
                        break;
                    }
                }

                if(find == 1) {

                    for(int k = 0; k < merged_id[i].size(); ++k) {

                        merged_id[id].push_back(merged_id[i][k]);
                    }

                    merged_id.erase(merged_id.begin()+i);
                    break;
                }
            }
        }
    }

    for(int i = 0; i < merged_id.size(); ++i) {

        std::unique(merged_id[i].begin(), merged_id[i].end());
    }


    for(int i = 0; i < merged_id.size(); ++i) {

        for(int j = 0; j < merged_id[i].size(); ++j) {

            if (merged_id[i][j] == -1) {
                continue;
            }

            for(cv::Point &pt : superpixels[merged_id[i][j]].points) {

                merged_superpixels[i].points.push_back(pt);
            }
        }
        
    }


    for (int i = static_cast<int>(merged_superpixels.size()) - 1; i >= 0; --i) {

        if (merged_superpixels[i].points.size() == 0) {

            merged_superpixels.erase(merged_superpixels.begin() + i);
        }
    }

    return;
}