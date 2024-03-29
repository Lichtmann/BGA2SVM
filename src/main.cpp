
// main.cpp

#include <map>
#include <vector>
#include <cmath>
#include<fstream>
#include "ue3.h"


// define the relevant classes
// we just consider a binary classification problem, so only two classes are relevant!
// example hard: 1 - 30er, 2 - 50er
// example easy: 1 - 30er, 38 - keep right

// alle von prohiboitory

// input und predictions gegenüberstellen

const std::vector<unsigned int> CONSIDERED_CLASS_IDs = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16 };

cv::PCA pca;
cv::HOGDescriptor hog;

unsigned int getFeatureVectorDimension()
{
	int dem = hog.getDescriptorSize();
	std::cout << "[Done]\twinSize = " << hog.winSize << std::endl;
	std::cout << "[Done]\tblockSize = " << hog.blockSize << std::endl;
	std::cout << "[Done]\tblockStride = " << hog.blockStride << std::endl;
	std::cout << "[Done]\tcellSize = " << hog.cellSize << std::endl;
	std::cout << "[Done]\tnbins = " << hog.nbins << std::endl;
	std::cout << "[Done]\tgetFeatureVectorDimension() = " << dem << std::endl;
    return 	hog.getDescriptorSize();
}


void computeFeatures(cv::Mat roi, std::vector<float> &features)
{
	cv::resize(roi, roi, cv::Size(64, 128));
	hog.winSize = roi.size();
	hog.compute(roi, features);
}

/*
INPUTS:
    - your classifier (adjust the type to your chosen model)
    - the (projected) features as cv::Mat
    - the labels as cv::Mat

OUTPUTS:
    - the correct classification rate on the given data set
*/
double evaluateClassifier(const YourClassifierType &classifier, const cv::Mat &features, const cv::Mat &labels, double train_ratio = 0.0, bool output = false)
{
    if (features.empty())
    {
        std::cout << "[error]\tevaluateClassifier(): given feature vector is empty." << std::endl;
        return 0.0;
    }

    cv::Mat predictions;
    classifier.predict(features, predictions);

	predictions.convertTo(predictions, CV_32SC1);

    // compare predictions to labels
    cv::Mat diff = labels != predictions;
    int misclassifications = cv::countNonZero(diff);
    std::cout << "[info]\tmisclassifications: " << misclassifications << " of " << labels.rows << std::endl;

	if (output)
	{
		// write predictions to csv
		// create and open output data file
		int ratio = train_ratio * 1000.0;
		std::ofstream file_handler;
		std::string output_filename = "../data/labels_predictions" + std::to_string(ratio) + ".csv"; 
		file_handler.open(output_filename);
		for (size_t i = 0; i < predictions.rows; i++)
		{
			file_handler << labels.at<int>(i, 0);
			file_handler << ",";
			file_handler << predictions.at<int>(i, 0);
			file_handler << "\n";
		}
	}

	double ccr;
    ccr = 1.*(labels.rows - misclassifications) / labels.rows;
    std::cout << "[info]\tccr: " << ccr << std::endl;
    return ccr;
}

/*
INPUTS:
    - your classifier (adjust the type to your chosen model)
    - the training records used for training

OUTPUTS:
    - the correct classification rate on the training data set
*/
double trainClassifier(YourClassifierType &classifier, std::vector<imageLabelGTSRB> trainingRecords,bool perform_pca, bool visualize_pca)
{
    // check if there are any records
    if (trainingRecords.empty())
    {
        std::cout << "[error]\ttrainClassifier(): vector of trainingRecords is empty." << std::endl;
        return 0;
    }

    /*** COMPUTE FEATURES FOR EACH IMAGE ***/
    std::cout << "[info]\tcomputing features of training images" << std::endl;
    cv::Mat trainingFeatures;
    cv::Mat trainingLabels;
    computeFeaturesToMat(trainingRecords, trainingFeatures, trainingLabels);

	cv::Mat projectedTrainingFeatures;
	if (perform_pca)
	{
		/*** PRINCIPAL COMPONENTS ANALYSIS ***/
		std::cout << "[info]\tperforming principal components analysis" << std::endl;
		pca = cv::PCA(trainingFeatures, cv::Mat(), cv::PCA::Flags::DATA_AS_ROW, 35);
		// TODO: create the pca using your training data


		std::cout << "[info]\tprojecting training features to new feature space" << std::endl;
		pca.project(trainingFeatures, projectedTrainingFeatures);

		if (visualize_pca)
		{
			/*** DISPLAY FIRST TWO PRINCIPAL COMPONENTS ***/
			std::cout << "[info]\tdisplaying the first two principal components" << std::endl;
			visualizePCA(projectedTrainingFeatures, trainingLabels);
		}
	}

    /*** TRAIN THE CLASSIFIER ***/
    // TODO: choose a classifier model (e.g. SVM, k-Nearest-Neighbor, NormalBayes, NeuralNetwork, ...) and train
    std::cout << "[info]\ttraining the classifier" << std::endl;
	cv::Mat responses(cv::Size(trainingLabels.cols,trainingLabels.rows),CV_32SC1);

	// train methods requires integer values for label data input
	for (size_t i = 0; i < trainingLabels.rows; i++)
	{
		for (size_t j = 0; j < trainingLabels.cols; j++)
				responses.at<int>(i, j) = (int)trainingLabels.at<float>(i, j);
	}

	if (perform_pca)
	{
		classifier.train(projectedTrainingFeatures, cv::ml::ROW_SAMPLE, responses);
	}
	if (!perform_pca)
	{
		classifier.train(trainingFeatures, cv::ml::ROW_SAMPLE, responses);
	}

    /*** CHECK THE TRAINING ERROR ***/
    std::cout << "[info]\ttraining error of the classifier" << std::endl;
    double training_ccr;
    training_ccr = evaluateClassifier(classifier, projectedTrainingFeatures, responses);

    return training_ccr;
}

/*
TODO:
    - split data into training and validation dataset
    - you can subsplit the training dataset with the splitDataSet function too:
        - just use split with 0.25, 0.5 and 0.75 to get 1/3, 2/3, 3/3 of training data
    - for each of the three training sets, train and evaluate a classifier
    - save training and validation errors (1-CCR) in the given vectors
*/
void createLearningCurve(std::vector<std::vector<std::pair<double, double>>> error_storage)
{
    std::vector<std::pair<double, double> > trainError;
    std::vector<std::pair<double, double> > valError;
    // push your data into these vectors
	for (size_t i = 0; i < error_storage.size(); i++)
	{
		trainError.push_back(error_storage[i][0]);
		valError.push_back(error_storage[i][1]);
	}
    // example for 10% of training data and validation CCR of 40% (= Error of 60%)

    // plot the learning curve
    std::cout << "[info]\tdisplaying the learning curve" << std::endl;
    visualizeLearningCurve(trainError, valError);
}

//
std::vector<double> trainAndPredict (std::vector<imageLabelGTSRB> &records, std::vector<unsigned int> nSamplesPerClass, double trainRatio , bool perform_pca, bool visualize_pca=false)
{
	std::cout << "\n---------------------------------------" << std::endl;


	/*** SPLIT INTO TRAINING AND VALIDATION DATA ***/
	std::cout << "[info]\tsplit the records into training and validation set" << std::endl;
	std::vector<imageLabelGTSRB> trainingRecords;
	std::vector<imageLabelGTSRB> validationRecords;

	splitDataSet(records, nSamplesPerClass, trainingRecords, validationRecords, trainRatio);

	/*** CREATE A CLASSIFIER AND TRAIN IT ***/
	// TODO: make this work
	cv::Ptr<YourClassifierType> classifier = YourClassifierType::create();
	double training_ccr;
	training_ccr = trainClassifier(*classifier, trainingRecords, perform_pca, visualize_pca);

	cv::Mat validationFeatures;
	cv::Mat validationLabels;
	computeFeaturesToMat(validationRecords, validationFeatures, validationLabels);

	validationLabels.convertTo(validationLabels, CV_32SC1);

	/*** PRINCIPAL COMPONENTS ANALYSIS ***/
	//std::cout << "[info]\tperforming principal components analysis" << std::endl;

	std::cout << "[info]\tprojecting validation features to new feature space" << std::endl;
	cv::Mat projectedValidationFeatures;
	pca.project(validationFeatures, projectedValidationFeatures);

	/*** EVALUATE IT ON THE VALIDATION DATA ***/

	std::cout << "[info]\tvalidation error of the classifier" << std::endl;
	double evaluation_ccr;
	evaluation_ccr = evaluateClassifier(*classifier, projectedValidationFeatures, validationLabels,trainRatio,true);

	// create outout array and return
	std::vector<double> output_crr{ training_ccr,evaluation_ccr };
	return output_crr;
}


// main method
// takes care of program flow
int main (int argc, char* argv[])
{

    /*** READING INPUT DATA ***/
    std::cout << "[info]\treading training data of relevant classes.." << std::endl;
    
    // stores path and class data
    std::vector<imageLabelGTSRB> records;

    // stores number of read samples
    std::vector<unsigned int> nSamplesPerClass;

    // fills in the values to records and nSamplesPerClass for the specified classes
    readDataSet(records, nSamplesPerClass, CONSIDERED_CLASS_IDs);
    std::cout << "[info]\t" << records.size() << " samples in total." << std::endl;

    // flag to switch between task 1 and 2
	// task 1 single rund task 2 multiple runs for training curve
    // switch here to work on task 2 (learning curve)
    const int task = 2;

    if (task == 1)
    {
		// perform training and predict on dataset
		trainAndPredict(records, nSamplesPerClass,0.9,true,true);
    }

	else if (task == 2)
    {
		// perform training and predict on dataset several times 
		// to create learning curve

		int steps = 30; // number of runs 30 for 15% training data
		double trainRatio = 0.005; // starting with 0.005% training data
		double ratiostep = 0.005;



		// first is ratio, second pointer to err data
		std::vector<std::vector<std::pair<double, double>>> error_storage;


		// create and open output data file
		std::ofstream file_handler;
		std::string output_filename = "../data/results.csv";
		file_handler.open(output_filename);

		for (size_t i = 0; i < steps; i++)
		{
			bool perform_pca = true;
			std::vector<double> error = trainAndPredict(records, nSamplesPerClass, trainRatio + ratiostep *(double)i, perform_pca); // raise trainingRatio every step
			
			// cast back data 
			double training_crr = error[0]; 
			double validation_crr = error[1];
			// store in container
			std::vector<std::pair<double, double>> recent_error;
			recent_error.push_back(std::make_pair(trainRatio + ratiostep * (double)i, 1- training_crr));
			recent_error.push_back(std::make_pair(trainRatio + ratiostep * (double)i, 1- validation_crr));

			error_storage.push_back(recent_error);
			
			// write to csv
			file_handler << trainRatio + ratiostep * (double)i;	// training data in %
			file_handler << ",";
			file_handler << 1 - training_crr;
			file_handler << ",";
			file_handler << 1 - validation_crr;
			file_handler << "\n";
			
		}

		file_handler.close();

        std::cout << "[info]\tcreating learning curve" << std::endl;

        createLearningCurve(error_storage);
    }
    

    std::cout << "[info]\tfinished program! goodbye. :)" << std::endl;

	system("pause");
	return 0;
}



	