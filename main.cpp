#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "mnist.h"
#include "primarycaps.h"
#include "digicaps.h"
#include "reconstruct.h"

using cv::Mat;
using std::vector;
using std::cerr;
using std::cout;

// will train weights on images 0..500, 40 iterations
#define CNT_TRAINING_IMGS  500
#define CNT_TRAINING_LOOPS 40

// will test/evaluate on images 0+0...0+1001
#define CNT_TESTING_IMGS 1001
#define CNT_TESTING_IMGS_OFFSET 0


static int do_training();
static int do_testing();
static int do_image_reconstruction(int index);
static void show_images(const cv::Mat* img1, cv::Mat* img2);
static float get_mse(int label, const float results[10][16]);
static void get_mse_stats(int nruns, float *mserrors, float &std_dev, double &avg);


int main(int argc, char* argv[])
{
	if (argc) {
		if (strcasecmp(argv[1],"train")==0) {
			return do_training();
		}
		if (strcasecmp(argv[1],"test")==0) {
			return do_testing();
		}
		if (strcasecmp(argv[1],"image")==0) {
			while (1) {
				int index;
				std::cout << "enter image index: ";
				std::cin >> index;
				std::cout << "\n";
				if (index < 0) { break; }
				do_image_reconstruction(index);
			}
			return 0;
		}
	}
	std::cout << "Invoke ./PANTHRO [train|test|image]\n";
	return 0;
}

static int do_testing()
{
	// Mnist number dataset
	const std::vector< cv::Mat > images = Mnist::getInstance().getImages();
	const std::vector< int > labels = Mnist::getInstance().getLabels();

	// PrimaryCaps as defined in whitepaper (convolutions & reshaping of mnist images)
	PrimaryCaps *pcaps[CNT_TESTING_IMGS];
	for (size_t i=0; i<CNT_TESTING_IMGS; ++i) {
		pcaps[i] = new PrimaryCaps(&images[i+CNT_TESTING_IMGS_OFFSET]);
	}

	// debugging/evaluation vars for later
	float mserrors[CNT_TRAINING_IMGS];
	float std_dev; // mserror
	double avg;    // mserror
	int numIncorrect = 0;

	// Running through network
	DigiCaps dcaps = DigiCaps::getInstance();
	float digicap_outputs[10][16];
	float output_lengths[10];
	for (size_t i=0; i<CNT_TESTING_IMGS; ++i) {
		int label = labels[i+CNT_TESTING_IMGS_OFFSET];

		// run & evaluate
		dcaps.forward_prop(pcaps[i]->get_reshaped_pcaps(), label, digicap_outputs);
		for (auto j=0; j<10; ++j) {
			output_lengths[j] = 0.0;
			for (auto k=0; k<16; ++k) {
				output_lengths[j] += std::pow(digicap_outputs[j][k],2);
			}
			output_lengths[j] = std::sqrt(output_lengths[j]);
		}
		for (auto j=0; j<10; ++j) {
			if (j != label && output_lengths[j] > output_lengths[label]) {
				++numIncorrect;
				break;
			}
		}
		mserrors[i] = get_mse(label, digicap_outputs);

		// log progress
		if (i && i % 100 == 0) {
			std::cout << "completed " << i+1 << " of " << CNT_TESTING_IMGS << " (numIncorrect: " << numIncorrect << ")\n";
		}
	}

	// print results
	int pctCorrect = static_cast<int>( 100.0 * (1.0 - ((float)numIncorrect/(float)CNT_TESTING_IMGS)) );
	get_mse_stats(CNT_TESTING_IMGS, mserrors, std_dev, avg);
	std::cout << "# Runs: " << CNT_TESTING_IMGS << "\n"
	          << "% Corr: " << pctCorrect << "%\n"
			  << "mean sq err(avg, stddev): " << avg << ", " << std_dev << "\n"
			  << "image index start offset: " << CNT_TESTING_IMGS_OFFSET << "\n\n";
	return 0;
}

static int do_training()
{
	// Mnist number dataset (http://yann.lecun.com/exdb/mnist/)
	const std::vector< cv::Mat > images = Mnist::getInstance().getImages();
	const std::vector< int > labels = Mnist::getInstance().getLabels();

	// PrimaryCaps as defined in whitepaper (convolutions & reshaping of mnist images)
	PrimaryCaps *pcaps[CNT_TRAINING_IMGS];
	for (size_t i=0; i<CNT_TRAINING_IMGS; ++i) {
		pcaps[i] = new PrimaryCaps(&images[i]);
	}

	// just debugging vars for later
	float mserrors[CNT_TRAINING_IMGS];
	float std_dev; // mserror
	double avg;    // mserror

	// runs through https://arxiv.org/pdf/1710.09829.pdf
	DigiCaps dcaps = DigiCaps::getInstance();
	Reconstruct recon = Reconstruct::getInstance();
	float digicap_outputs[10][16]; // output of digicaps network for each label
	for (size_t x=0; x<CNT_TRAINING_LOOPS; ++x) {

		// training
		for (size_t i=0; i<CNT_TRAINING_IMGS; ++i) {
			dcaps.forward_prop(pcaps[i]->get_reshaped_pcaps(), labels[i], digicap_outputs);
			recon.fwd_and_bwd_prop(digicap_outputs[labels[i]], labels[i], &images[i], nullptr);
			dcaps.backward_prop(pcaps[i]->get_reshaped_pcaps(), labels[i], digicap_outputs, recon.recon_errs);
			mserrors[i] = get_mse(labels[i], digicap_outputs); // for debugging, can comment-out
		}
		dcaps.updateWeights();
		recon.update_weights();

		// serialize trained weights to file
		if (x && x % 5 == 0) { 
			printf("\nsaving weights to file\n");
			dcaps.saveWeightsToFile();
			recon.write_weights_to_files();			
		}

		// log progress
		get_mse_stats(CNT_TRAINING_IMGS, mserrors, std_dev, avg);
		printf("Run %zu MSErrors:  avg= %f  stdev=%f\n", x, avg, std_dev );
	}

	// done
	dcaps.saveWeightsToFile();
	recon.write_weights_to_files();
	return 0;
}

static int do_image_reconstruction(int index)
{
	// Mnist number dataset
	const std::vector< cv::Mat > images = Mnist::getInstance().getImages();
	const std::vector< int > labels = Mnist::getInstance().getLabels();
	
	// PrimaryCaps as defined in whitepaper (convolutions & reshaping of mnist images)
	const cv::Mat *image_in = &images[index];
	PrimaryCaps *pcaps = new PrimaryCaps(image_in);

	// DigiCaps as defined in whitepaper
	const int label = labels[index];
	float digicap_outputs[10][16];
	DigiCaps dcaps = DigiCaps::getInstance();
	dcaps.forward_prop(pcaps->get_reshaped_pcaps(), label, digicap_outputs);
	delete pcaps; pcaps = 0;

	// Image reconstruction network
	cv::Mat image_out(cv::Size(28,28),CV_8UC1);
	Reconstruct recon = Reconstruct::getInstance();
	recon.fwd_and_bwd_prop(digicap_outputs[label], label, image_in, &image_out);

	// show em
	show_images(image_in, &image_out);
	return 0;
}

static float get_mse(int label, const float in[10][16]) 
{
	float errors[10] = {0};
	float results[16];
	for (int i=0; i<10; ++i) {
		double sum = 0;
		for (int j=0; j<16; ++j) {
			sum += std::pow(in[i][j],2);
		}
		float length = std::sqrt(sum);
		if (i == label) {
			errors[i] = std::pow(std::max(0.0, 0.9-length),2);
		} else {
			errors[i] = .5 * std::pow(std::max(0.0, length-0.1),2);
		}
	}

	float mse = 0.0;
	for (int i=0; i<10; ++i) {
		mse += errors[i];
	}
	mse = mse/10.0;
	return mse;
}

static void get_mse_stats(int nruns, float *mserrors, float &std_dev, double &avg) 
{
	avg = 0.0;
	for (size_t i=0; i<nruns; ++i) {
		avg += mserrors[i];
	}
	avg /= (double)nruns;
		
	double sum = 0.0;
	for (size_t i=0; i<nruns; ++i) {
		sum += std::pow(mserrors[i] - avg, 2);
	}
	sum /= (double)nruns;
	std_dev = std::sqrt(sum);
}

static void show_images(const cv::Mat* img__1, cv::Mat* img2)
{	
	// put original back into original format (i'd upp'ed to float in mnist.cpp to avoid casting all over)
	cv::Mat img1;
	img__1->convertTo(img1, CV_8UC1);

	// resize with linear interpretation and concatenate the original + reconstructed images
	cv::Mat img1_bigger, img2_bigger;
	cv::resize(img1, img1_bigger, cv::Size(112,112));
	cv::resize(*img2, img2_bigger, cv::Size(112,112));
	cv::Mat img(cv::Size(112,224), img2->type());//img2->type());
	cv::hconcat(img1_bigger, img2_bigger, img);
	cv::putText(img, "Orig", cv::Point(1,12), cv::FONT_HERSHEY_COMPLEX_SMALL, .6, cv::Scalar(255,192,203), 1);
	cv::putText(img, "Recon", cv::Point(113,12), cv::FONT_HERSHEY_COMPLEX_SMALL, .6, cv::Scalar(255,192,203), 1);

	// show em
	cv::imshow("original vs reconstructed", img);
	cv::waitKey();
	cv::destroyAllWindows();
}
