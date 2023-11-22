/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";
    
    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // results
    string resPath = "../output/task8/";
    
    
    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    bool bVis = false;            // visualize results

    // task 7
    int N_DETECTORS = 7;

    vector<string> detectorNames(N_DETECTORS);
    detectorNames[DetectorType::akaze] = "akaze";
    detectorNames[DetectorType::brisk] = "brisk";
    detectorNames[DetectorType::fast] = "fast";
    detectorNames[DetectorType::harris] = "harris";
    detectorNames[DetectorType::orb] = "orb";
    detectorNames[DetectorType::shitomasi] = "shitomasi";
    detectorNames[DetectorType::sift] = "sift";
	
    vector<string> descriptorTypes = {"AKAZE", "BRISK", "BRIEF", "ORB", "FREAK", "SIFT"};

    /*LOOP OVER ALL DETECTORS */
    for (int i = 0;  i < N_DETECTORS; ++i) {
      DetectorType dt = DetectorType(i);
      std::vector<std::vector<float>> sizes;

      /*LOOP OVER ALL DESCRIPTOR TYPES */

      
      for (const string& descriptorType: descriptorTypes) {
	vector<int> n_matches;
	bool compat = true; // compat between detector and descriptor 

	if (descriptorType.compare("AKAZE") == 0 && dt != DetectorType::akaze)
	  continue;

	if (descriptorType.compare("ORB") == 0 && dt == DetectorType::sift){ // tries to allocate  to much memory
	  continue;
	}

	  
	cout << "+++++++++++"  << detectorNames[dt] << " - " << descriptorType << std::endl; 


	deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
	/*LOOP OVER ALL IMAGES */
	for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
	  {
	    std::vector<float> _sizes;
	    /* LOAD IMAGE INTO BUFFER */
	  
	    // assemble filenames for current index
	    ostringstream imgNumber;
	    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
	    string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

	    // load image from file and convert to grayscale
	    cv::Mat img, imgGray;
	    img = cv::imread(imgFullFilename);
	    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

	    // push image into data frame buffer
	    DataFrame frame;
	    frame.cameraImg = imgGray;
	    if ( dataBuffer.size() == dataBufferSize ){
	      dataBuffer.pop_front();
	    }
	    dataBuffer.push_back(frame);
	    assert(dataBuffer.size() <= dataBufferSize);
	    
	    cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
	    
	    /* DETECT IMAGE KEYPOINTS */
	    // extract 2D keypoints from current image
	    vector<cv::KeyPoint> keypoints; // create empty feature list for current image
	    
	    switch(dt) {
	  case DetectorType::akaze : detKeypointsModern(keypoints, img, dt, bVis); break;
	    case DetectorType::brisk : detKeypointsModern(keypoints, img, dt, bVis); break;
	    case DetectorType::fast : detKeypointsModern(keypoints, img, dt, bVis); break;
	    case DetectorType::harris : detKeypointsHarris(keypoints, imgGray, bVis); break;
	    case DetectorType::orb : detKeypointsModern(keypoints, img, dt, bVis); break;
	    case DetectorType::shitomasi : detKeypointsShiTomasi(keypoints, imgGray, bVis); break;
	    case DetectorType::sift : detKeypointsModern(keypoints, img, dt, bVis); break;
	    default: throw std::runtime_error("Error unknown detector type.");
	    }
	    
	    cv::Rect vehicleRect(535, 180, 180, 150);
	    keypoints.erase(std::remove_if(begin(keypoints), end(keypoints), [vehicleRect](cv::KeyPoint kp) { return !vehicleRect.contains(kp.pt); }), end(keypoints));
        
	    // push keypoints and descriptor for current frame to end of data buffer
	    (dataBuffer.end() - 1)->keypoints = keypoints;
	    cout << "#2 : DETECT KEYPOINTS done" << endl;

	    cv::Mat descriptors;
	    //try {
	      descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
	    
	      // push descriptors for current frame to end of data buffer
	      (dataBuffer.end() - 1)->descriptors = descriptors;
	      
	      
	      if (dataBuffer.size() > 1) // wait until at least two images have been processed
		{
		  /* MATCH KEYPOINT DESCRIPTORS */

		  vector<cv::DMatch> matches;
		  string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
		  if (descriptorType.compare("SIFT") == 0) {
		    string descriptorType = "DES_HOG"; // DES_BINARY, DES_HOG
		  } else {
		    string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
		  }
		  string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN
		  
		  matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
				   (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
				   matches, descriptorType, matcherType, selectorType);
		  
		  (dataBuffer.end() - 1)->kptMatches = matches;
		  
		  cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
		  n_matches.push_back(matches.size());
		  
		}
	      //} catch(...) {
	      // ignore incompatibilities between detector / descriptor / matcher
	      //compat = false;
	      //}

	  } // eof loop over all images

	// write result to file
	if (compat){ 
	  std::ofstream file(resPath + detectorNames[i] + "_" + descriptorType + ".dat");
	  if (!file.is_open()) {
	    throw std::runtime_error("Error opening file for writing.");
	  }
	  
	  for (const auto& val : n_matches) {
	    file << val << " ";
	  }
	  
	  file.close();
	} // endif compat
      } // eof loop over all descriptor types
    } // eof loop over all detectors
    
    return 0;
}
