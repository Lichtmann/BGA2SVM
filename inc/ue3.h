// ue3.h

// makes sure this file is included only once
#pragma once

#include "opencv2\opencv.hpp"
#include "opencv2\ml.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <sys/stat.h>

//#include "gnuplot-iostream.h"

#define YourClassifierType cv::ml::NormalBayesClassifier

// define path
const std::string IMAGE_DIR_GTSRB = "D:/GTSRB/Final_Training/Images/";
//const std::string IMAGE_DIR_GTSRB = "C:/Users/Oli K/projects/BGA2/data/GTSRB/Final_Training/Images/";

// create categories of traffic signs
const std::vector<unsigned int> PROHIBITORY = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16 };
const std::vector<unsigned int> MANDATORY = { 33, 34, 35, 36, 37, 38, 39, 40 };
const std::vector<unsigned int> DANGER = { 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
const std::vector<unsigned int> ALL;

// define data structures
struct imageLabelGTSRB
{
    std::string filename;
    unsigned int classID;

    imageLabelGTSRB(std::string file, unsigned int id) : filename(file), classID(id) {}
};

// check if file exists
bool fileExists(const std::string fileName)
{
    struct stat stFileInfo;
    const int intStat = stat(fileName.c_str(), &stFileInfo);
    return (intStat == 0);
}

// check if classID is relevant
// i.e. return true if vector is empty or classID is contained in the vector
bool isClassRelevant(const std::vector<unsigned int> relevantClasses, const unsigned int classID)
{
    return (relevantClasses.empty() || (std::find(relevantClasses.begin(), relevantClasses.end(), classID) != relevantClasses.end()));
}

// read the GTSRB dataset
void readDataSet(std::vector<imageLabelGTSRB> &records, std::vector<unsigned int> &nSamplesPerClass,
    std::vector<unsigned int> relevantClasses = ALL)
{
    unsigned int nSamples = 0;
    for (unsigned int c = 0;; ++c)
    {
        if (!isClassRelevant(relevantClasses, c))
        {
            // check whether there ought to be more relevant classes
            if (c > *std::max_element(std::begin(relevantClasses), std::end(relevantClasses)))
            {
                // end completely
                break;
            }
            else
            {
                // if class is not relevant we can just skip it
                nSamplesPerClass.push_back(0);
                continue;
            }
            break;
        }
        bool foundFileForClass = false;

        for (int t = 0;; ++t)
        {
            bool foundFileForTrack = false;

            // +=4: we won't use every single frame of each track, just every 4th
            // we have plenty of data and want a subset of good diversity
            for (int e = 0;; e += 4)
            {
                char fileName[32];
                sprintf_s(fileName, "%05d/%05d_%05d.ppm", c, t, e);
                std::string filePath = IMAGE_DIR_GTSRB + fileName;

                if (fileExists(filePath))
                {
                    foundFileForClass = true;
                    foundFileForTrack = true;
                    nSamples++;
                    records.push_back(imageLabelGTSRB(filePath, c));
                }
                else
                {
                    break;
                }
            }
            if (false == foundFileForTrack)
            {
                std::cout << "[info]\tfound " << nSamples << " samples of class " << c << "." << std::endl;
                nSamplesPerClass.push_back(nSamples);
                nSamples = 0;
                break;
            }
        }
        if (false == foundFileForClass) break;
    }
}


// this methods splits the dataset given by records (and further specified by nSamplesPerClass)
// into training and validation records. trainRatio specifies the ratio of training data.
// e.g. trainRatio = 0.75 means that 75% of records will be training, 25% validation data.
// optionally, you can also retrieve the number of samples per class in the training and validation
// set by specifying nSamplesPerClassTrain and nSamplesPerClassVal respectively.
// > note that this method is deterministic and gives equal results for equal inputs!
void splitDataSet(std::vector<imageLabelGTSRB> records, std::vector<unsigned int> nSamplesPerClass,
    std::vector<imageLabelGTSRB> &trainRecs, std::vector<imageLabelGTSRB> &valRecs, double trainRatio = 0.75,
    std::vector<unsigned int> &nSamplesPerClassTrain = std::vector<unsigned int>(), 
    std::vector<unsigned int> &nSamplesPerClassVal = std::vector<unsigned int>())
{
    unsigned int offset = 0;
    // for each class separately
    for (unsigned int c = 0; c < nSamplesPerClass.size(); ++c)
    {
        // compute critical indices
        unsigned int lastTrainIndex = offset + static_cast<unsigned int>(trainRatio * nSamplesPerClass[c]);
        unsigned int lastValIndex = offset + nSamplesPerClass[c];

        // get the number of samples in each set right
        unsigned int nSamplesTrain = static_cast<unsigned int>(trainRatio * nSamplesPerClass[c]);
        unsigned int nSamplesVal = nSamplesPerClass[c] - nSamplesTrain;
        nSamplesPerClassTrain.push_back(nSamplesTrain);
        nSamplesPerClassVal.push_back(nSamplesVal);

        // insert elements into other vectors accordingly
        trainRecs.insert(trainRecs.end(), records.begin() + offset, records.begin() + lastTrainIndex);
        valRecs.insert(valRecs.end(), records.begin() + lastTrainIndex, records.begin() + lastValIndex);

        // remember current position
        offset += nSamplesPerClass[c];
    }

    std::cout << "[debug]\t#samples train: " << trainRecs.size() << std::endl;
    std::cout << "[debug]\t#samples val:   " << valRecs.size() << std::endl;
    std::cout << "[debug]\trealized ratio: " << 1.*trainRecs.size() / (trainRecs.size() + valRecs.size()) << std::endl;
}


// compute the hog features for a roi - to be implemented in main.cpp
void computeFeatures(cv::Mat roi, std::vector<float> &features);

// get the dimension of descriptor - to be implemented in main.cpp
unsigned int getFeatureVectorDimension();

// this method computes the features for a given dataset
// it organizes the features and labels conveniently in cv::Mat format used for most OpenCV classifiers
void computeFeaturesToMat(std::vector<imageLabelGTSRB> records, cv::Mat &features, cv::Mat &labels)
{
    // initialize data 
    // in the mat, each row is a sample vector
    // each column represents one feature dimension
    unsigned int rows = records.size();
    unsigned int cols = getFeatureVectorDimension();
    features = cv::Mat(rows, cols, CV_32FC1);
    labels = cv::Mat(rows, 1, CV_32FC1);

    // probably we need to deal with mapping the labels to contiguous values
    // let's postpone this. idea is to deliver a map, which can than be used at classification time.

    std::cout << "[info]\tcomputing features " << std::flush;
    unsigned int step_size = rows / 10;
    unsigned int step = step_size;

    // for each sample in records
    for (unsigned int s = 0; s < rows; ++s)
    {
        if (s >= step) // stupid visualization of progress
        {
            step += step_size;
            std::cout << "." << std::flush;
        }

        // read image

        cv::Mat roi = cv::imread(records[s].filename);

        // resize image and compute features
        std::vector<float> descriptors;
        computeFeatures(roi, descriptors);
        
        // copy descriptors to features mat
        assert(cols == descriptors.size());
        for (unsigned int f = 0; f < cols; ++f)
        {
            features.at<float>(s, f) = descriptors[f];
        }

        // set labels
        labels.at<float>(s) = static_cast<float>(records[s].classID);

    }

    std::cout << " done!" << std::endl;
}



// this function displays the first two principal components on a gnuplot
void visualizePCA(const cv::Mat &features, const cv::Mat &labels)
{
    //Gnuplot gp;

    //// assert features.rows == labels.rows
    //if (features.rows != labels.rows)
    //{
    //    std::cout << "[error] cannot display pca. (dimension mismatch.)" << std::endl;
    //    return;
    //}

    //float current_label = -1.;
    //std::vector<std::pair<double, double> > xy_pts;
    //gp << "set xrange [-5:5]\nset yrange [-5:5]\n";
    //gp << "plot";
    //gp << std::fixed << std::setprecision(0); // for nice display of float labels

    //for (int i = 0; i < features.rows; i++)
    //{
    //    if (labels.at<float>(i, 0) != current_label)
    //    {
    //        // plot vector if not empty (will only be empty first time)
    //        if (!xy_pts.empty())
    //        {
    //            gp << gp.file1d(xy_pts) << "with points title 'class ID: " << current_label << "',";
    //        }

    //        // create shiny new vector
    //        xy_pts = std::vector<std::pair<double, double> >();

    //        // remember the new label
    //        current_label = labels.at<float>(i, 0);
    //    }

    //    // push the data point to the vector
    //    xy_pts.push_back(
    //        std::make_pair(
    //            features.at<float>(i, 0),
    //            features.at<float>(i, 1)));
    //}

    //// aand plot the last vector
    //gp << gp.file1d(xy_pts) << "with points title 'class ID: " << current_label << "'" << std::endl;
    //std::cout << "[info]\tclose the gnuplot window to continue." << std::endl;
}


// plot the learning curve
void visualizeLearningCurve(std::vector<std::pair<double, double> > trainError, 
    std::vector<std::pair<double, double> > valError)
{
   /* Gnuplot gp;
    gp << "set xrange [0:1]\nset yrange [0:1]\n";
    gp << "plot";
    gp << gp.file1d(trainError) << "with line title 'training error',";
    gp << gp.file1d(valError) << "with line title 'validation error'";
    gp << std::endl;
    std::cout << "[info]\tclose the gnuplot window to continue." << std::endl;*/
}