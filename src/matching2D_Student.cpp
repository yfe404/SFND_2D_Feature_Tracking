#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
      
      int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
      matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
      // FLANN requires descriptors to be of type CV_32F (float32).
      // At the time of writing (May 2019), there is a potential bug in the current implementation of the OpenCV,
      // which requires a conversion of the binary descriptors into floating point vectors, which is inefficient.
      // Yet still there is an improvement in speed, albeit not as large as it potentially could be.
      if (descSource.type() != CV_32F) {
	  descSource.convertTo(descSource, CV_32F);
	  descRef.convertTo(descRef, CV_32F);
	}

      matcher = cv::FlannBasedMatcher::create();		
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)


      // Before calling batchDistance, add the following logging:
std::cout << "Matrix descSource properties:" << std::endl;
std::cout << " - Rows: " << descSource.rows << std::endl;
std::cout << " - Cols: " << descSource.cols << std::endl;
std::cout << " - Type: " << descSource.type() << " (CV_32F = " << CV_32F << ", CV_8U = " << CV_8U << ")" << std::endl;

std::cout << "Matrix descRef properties:" << std::endl;
std::cout << " - Rows: " << descRef.rows << std::endl;
std::cout << " - Cols: " << descRef.cols << std::endl;
std::cout << " - Type: " << descRef.type() << " (CV_32F = " << CV_32F << ", CV_8U = " << CV_8U << ")" << std::endl;


 
 matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
	vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
	
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
      int bytes = 32; // length of the descriptor in bytes, valid values are: 16, 32 (default) or 64 . 
      bool use_orientation = false; // sample patterns using keypoints orientation.

      extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
      int  	nfeatures = 500; // 	The maximum number of features to retain.
      float  	scaleFactor = 1.2f; // 	Pyramid decimation ratio, greater than 1.
      int  	nlevels = 8; // 	The number of pyramid levels. The smallest level will have linear size equal to input_image_linear_size/pow(scaleFactor, nlevels - firstLevel). 
      int  	edgeThreshold = 31; // 	This is size of the border where the features are not detected. It should roughly match the patchSize parameter. 
      int  	firstLevel = 0; // The level of pyramid to put source image to. Previous layers are filled with upscaled source image.
      int  	WTA_K = 2; //	The number of points that produce each element of the oriented BRIEF descriptor
      cv::ORB::ScoreType  	scoreType = cv::ORB::HARRIS_SCORE; // The default HARRIS_SCORE means that Harris algorithm is used to rank features FAST_SCORE alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute
      int  	patchSize = 31; // size of the patch used by the oriented BRIEF descriptor. 
      int  	fastThreshold = 20; // 	the fast threshold
      
      extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }

    else if (descriptorType.compare("FREAK") == 0)
    {

      bool  	orientationNormalized = true; // Enable orientation normalization.
      bool  	scaleNormalized = true;	// Enable scale normalization.
      float  	patternScale = 22.0f; // Scaling of the description pattern.
      int  	nOctaves = 4; // Number of octaves covered by the detected keypoints.
      const std::vector< int > &selectedPairs = std::vector< int >();//  (Optional) user defined selected pairs indexes, 
      
      extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves, selectedPairs);
    }
    else if (descriptorType.compare("AKAZE") == 0)
      {
      cv::AKAZE::DescriptorType  descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB; //  Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
      int descriptor_size = 0; //Size of the descriptor in bits. 0 -> Full size
      int descriptor_channels = 3; // Number of channels in the descriptor (1, 2, 3)
      float threshold = 0.001f;	// Detector response threshold to accept point
      int nOctaves = 4; // Maximum octave evolution of the image
      int nOctaveLayers = 4; // Default number of sublevels per scale level
      cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2; // Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER 
	
      extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
      }
    else if (descriptorType.compare("SIFT") == 0)
      {
	
	int  	nfeatures = 0; // The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
	int  	nOctaveLayers = 3; // The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
	double  	contrastThreshold = 0.04; // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
	double  	edgeThreshold = 10; // The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
	double  	sigma = 1.6; // sigma	The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number. 
	
	extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
      }
    else {
      
    }
    
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
  // compute detector parameters based on image size
    int blockSize = 2;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    //    double minDistance = (1.0 - maxOverlap) * blockSize;
    //    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
    double k = 0.04; // Harris parameter (see equation for details)

    // Apply keypoints detection
    double t = (double)cv::getTickCount();

    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);


    // locate local maxima in the Harris response matrix
  
    for (int j = 0; j < dst_norm.rows; j++)
    {
        for (int i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            {
                cv::KeyPoint newKeypoint;
                newKeypoint.pt = cv::Point2f(i, j);
                newKeypoint.size = 2 * apertureSize;
                newKeypoint.response = response;

                // perform a non-maximum suppression (NMS) in a local neighborhood around each maximum
                bool isOverlapped = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double overlap = cv::KeyPoint::overlap(newKeypoint, *it);
                    if (overlap > maxOverlap)
                    {
                        isOverlapped = true;
                        if (newKeypoint.response > (*it).response)
                        {
                            *it = newKeypoint; // replace the keypoint with a higher response one
                            break;
                        }
                    }
                }

                // add the new keypoint which isn't consider to have overlap with the keypoints already stored in the list
                if (!isOverlapped)
                {
                    keypoints.push_back(newKeypoint);
                }
            }
        }
    }

    
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, DetectorType detectorType, bool bVis) {

  cv::Ptr<cv::FeatureDetector> detector;
  string name; 
  
  switch(detectorType){
  case DetectorType::akaze : detector = cv::AKAZE::create(); name = "AKAZE"; break;
  case DetectorType::brisk : detector = cv::BRISK::create(); name = "BRISK"; break;
  case DetectorType::fast : detector = cv::FastFeatureDetector::create(); name = "FAST"; break;
  case DetectorType::orb : detector = cv::ORB::create(); name = "ORB"; break;
  case DetectorType::sift : detector = cv::xfeatures2d::SIFT::create(); name = "SIFT"; break;
    
  }

  double t = (double)cv::getTickCount();
  detector->detect(img, keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << name << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}


// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
