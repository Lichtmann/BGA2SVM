
// main.cpp

#include <map>
#include <vector>
#include <cmath>

#include "ue3.h"


// define the relevant classes
// we just consider a binary classification problem, so only two classes are relevant!
// example hard: 1 - 30er, 2 - 50er  
// example easy: 1 - 30er, 38 - keep right
const std::vector<unsigned int> CONSIDERED_CLASS_IDs = { 1, 2 };


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
double evaluateClassifier(const YourClassifierType &classifier, const cv::Mat &features, const cv::Mat &labels)
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
double trainClassifier(YourClassifierType &classifier, std::vector<imageLabelGTSRB> trainingRecords, bool visualize_pca)
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


    /*** PRINCIPAL COMPONENTS ANALYSIS ***/
    std::cout << "[info]\tperforming principal components analysis" << std::endl;
	pca = cv::PCA(trainingFeatures, cv::Mat(), cv::PCA::Flags::DATA_AS_ROW,10);
	// TODO: create the pca using your training data


    std::cout << "[info]\tprojecting training features to new feature space" << std::endl;
    cv::Mat projectedTrainingFeatures;
	pca.project(trainingFeatures, projectedTrainingFeatures);

	if (visualize_pca)
	{
		/*** DISPLAY FIRST TWO PRINCIPAL COMPONENTS ***/
		std::cout << "[info]\tdisplaying the first two principal components" << std::endl;
		visualizePCA(projectedTrainingFeatures, trainingLabels);

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


	classifier.train(projectedTrainingFeatures,cv::ml::ROW_SAMPLE, responses);


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
double* trainAndPredict (std::vector<imageLabelGTSRB> &records, std::vector<unsigned int> nSamplesPerClass, double trainRatio , bool visualize_pca)
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
	training_ccr = trainClassifier(*classifier, trainingRecords, visualize_pca);

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
	evaluation_ccr = evaluateClassifier(*classifier, projectedValidationFeatures, validationLabels);

	// create outout array and return
	double output_crr[2] { training_ccr,evaluation_ccr };
	return output_crr;

}



// main method
// takes care of program flow
int main (int argc, char* argv[])
{
    /*** READING INPUT DATA ***/
    std::cout << "[info]\treading training data of relevant classes.." << std::endl;
    
    // stores path and class data
    std::vector<imageLabelGTSRB> records;  //list<filename,classID>

    // stores number of read samples
    std::vector<unsigned int> nSamplesPerClass; //

    // fills in the values to records and nSamplesPerClass for the specified classes
    readDataSet(records, nSamplesPerClass, CONSIDERED_CLASS_IDs/*{ 1, 2 }*/);
    std::cout << "[info]\t" << records.size() << " samples in total." << std::endl;

    // flag to switch between task 1 and 2
	// task 1 single rund task 2 multiple runs for training curve
    // switch here to work on task 2 (learning curve)
    const int task = 1;

    if (task == 1)
    {
		// perform training and predict on dataset
		trainAndPredict(records, nSamplesPerClass,0.9,true);

    }

	else if (task == 2)
    {
		// perform training and predict on dataset several times 
		// to create learning curve

		int steps = 8; // number of runs
		double trainRatio = 0.1; // starting with 10% training data
		double ratiostep = 0.1;
		// first is ratio, second pointer to err data
		std::vector<std::vector<std::pair<double, double>>> error_storage;


		for (size_t i = 0; i < steps; i++)
		{
			bool visualize_pca = false;
			double* error = trainAndPredict(records, nSamplesPerClass, trainRatio + ratiostep *(double)i, visualize_pca); // raise trainingRatio every step
			
			// cast back data 
			// and cut decimal 
			double training_crr = error[0]; 
			double validation_crr = error[1];
			// store in container
			std::vector<std::pair<double, double>> recent_error;
			recent_error.push_back(std::make_pair(trainRatio + ratiostep * (double)i, 1- training_crr));
			recent_error.push_back(std::make_pair(trainRatio + ratiostep * (double)i, 1- validation_crr));


			error_storage.push_back(recent_error);

		}

        std::cout << "[info]\tcreating learning curve" << std::endl;

        createLearningCurve(error_storage);
    }
    

    std::cout << "[info]\tfinished program! goodbye. :)" << std::endl;

	system("pause");
	return 0;
}


