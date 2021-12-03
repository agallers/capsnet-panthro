/* Andy Gallerstein (c) 2021, andy.gallerstein@gmail.com */
#include "reconstruct.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

using namespace std;
using cv::Mat;

#define DECODE_WEIGHTS_FILE_1 "../data/decode_weights1.txt"
#define DECODE_WEIGHTS_FILE_2 "../data/decode_weights2.txt"
#define DECODE_WEIGHTS_FILE_3 "../data/decode_weights3.txt"

#define LEARNING_WEIGHT .02

void helper_prop( float *input, size_t input_size,
                  float *output, size_t output_size,
                  float **dweights,
				  bool use_relu);


Reconstruct::Reconstruct()
{
	dweights1 = new float*[16]; // [16][512] 1st layer weights (relu)
	weightDeltas1 = new float*[16];
	weightVelocities1 = new float*[16];
	for (size_t i=0; i<16; ++i) {
		dweights1[i] = new float[512];
		weightDeltas1[i] = new float[512];
		weightVelocities1[i] = new float[512];
		memset(weightDeltas1[i], 0, sizeof(float) * 512);
		memset(weightVelocities1[i], 0, sizeof(float) * 512);
	}
	dweights2 = new float*[512]; // [512][1024] 2nd layer (relu)
	weightDeltas2 = new float*[512];
	weightVelocities2 = new float*[512];
	for (size_t i=0; i<512; ++i) {
		dweights2[i] = new float[1024];
		weightDeltas2[i] = new float[1024];
		weightVelocities2[i] = new float[1024];
		memset(weightDeltas2[i], 0, sizeof(float) * 1024);
		memset(weightVelocities2[i], 0, sizeof(float) * 1024);
	}
	dweights3 = new float*[1024]; // [1024][784] final layer (sigmoid * 255)
	weightDeltas3 = new float*[1024];
	weightVelocities3 = new float*[1024];
	for (size_t i=0; i<1024; ++i) {
		dweights3[i] = new float[784];
		weightDeltas3[i] = new float[784];
		weightVelocities3[i] = new float[784];
		memset(weightDeltas3[i], 0, sizeof(float) * 784);
		memset(weightVelocities3[i], 0, sizeof(float) * 784);
	}	

	memset( recon_errs, 0, sizeof(float)*16);
	read_weights_from_files();
}

Reconstruct::~Reconstruct()
{
	/*for (size_t i=0; i<16; ++i) {
		delete [] dweights1[i];
	}
	delete [] dweights1;
	for (size_t i=0; i<512; ++i) {
		delete [] dweights2[i];
	}
	delete [] dweights2;
	for (size_t i=0; i<1024; ++i) {
		delete [] dweights3[i];
	}	
	delete [] dweights3;*/
}

// helper for the function after this one
void helper_prop( float *input, size_t input_size,
                  float *output, size_t output_size,
                  float **dweights,
				  bool use_relu)
{
	for (size_t i=0; i<output_size; ++i) {
		output[i] = 0;
		for (size_t j=0; j<input_size; ++j) {
			output[i] += input[j] * dweights[j][i];
		}
		if (use_relu) {
			if (output[i] < 0) {
				output[i] = 0;
			}
		} else { // sigmoid
			output[i] = (float)1.0/((float)1.0+exp(float(-output[i])));			
		}
	}
}

// does basically everything
void Reconstruct::fwd_and_bwd_prop(float layer1out[16], 
                               int label, // label of the mnist input image (0..9)
                               const cv::Mat *img_in, // image from mnist input sample for this run
							   cv::Mat *img_out) // gonna make a grayscale 255 reconstructed image
{
	// Forward Propagation
	float layer2out[512];  // 1st layer weights (relu)
	float layer3out[1024]; // 2nd layer (relu)
	float layer4out[784];  // final layer (sigmoid * 255)
	helper_prop(layer1out,   16, layer2out,  512, dweights1, true);
	helper_prop(layer2out,  512, layer3out, 1024, dweights2, true);
	helper_prop(layer3out, 1024, layer4out,  784, dweights3, false);

	// Backward Propagation

	// Δ Error_total / Δ layer4out (SE derivative)
	float layer4_delta_err[784] = {0};
	for (size_t j=0; j<784; ++j) {
		size_t img_row = j/28;
		size_t img_col = j%28;
		float target = img_in->at<float>(img_row, img_col) / 255.;
		float output = layer4out[j];
		layer4_delta_err[j] = (output - target); // "sum of squared differences"
	}

	// Δ output / Δ input (sigmoid derivative)
	float layer4_delta_activation[784] = {0};
	for (size_t j=0; j<784; ++j) {
		float output = layer4out[j];
		layer4_delta_activation[j] = output * (1.0 - output);
	}

	// Δ input / Δ weight ()
	// Δ layer3out / Δ weight => layer3out (i.e. layer 4's inputs)

	// Δ Error_total / Δ weight(s)
	float weight3_deltas[1024][784] = {0};
	for (size_t i=0; i<1024; ++i) {
		for (size_t j=0; j<784; ++j) {
			weight3_deltas[i][j] = layer4_delta_err[j] * layer4_delta_activation[j] * layer3out[i];
		}
	}

	//////// -------- Next layer [512] -> [1024] ----------------

	// Δ Error_total / Δ layer3out ()
	float layer3_delta_err[1024] = {0};
	for (size_t j=0; j<1024; ++j) {
		for (size_t i=0; i<784; ++i) {
			layer3_delta_err[j] += 	layer4_delta_err[i] * layer4_delta_activation[i] * dweights3[j][i];
		}
	}

	// Δ output / Δ input (Relu derivative)
	float layer3_delta_activation[1024] = {0};
	for (size_t j=0; j<1024; ++j) {
		// don't have sum inputs, but can just assume if output 0 then input was <= 0
		layer3_delta_activation[j] = (layer3out[j] > 0 ? 1 : 0);
	}

	// Δ input / Δ weight ()
	// Δ layer2out / Δ weight => layer2out (i.e. layer 3's inputs, sans the weights)

	// Δ Error_total / Δ weight(s)
	float weight2_deltas[512][1024] = {0};
	for (size_t i=0; i<512; ++i) {
		for (size_t j=0; j<1024; ++j) {
			weight2_deltas[i][j] = layer3_delta_err[j] * layer3_delta_activation[j] * layer2out[i];
		}
	}

	//////// -------- Next layer [16] -> [512] ----------------

	// Δ Error_total / Δ layer2out ()
	float layer2_delta_err[512] = {0};
	for (size_t j=0; j<512; ++j) {
		for (size_t i=0; i<1024; ++i) {
			layer2_delta_err[j] += 	layer3_delta_err[i] * layer3_delta_activation[i] * dweights2[j][i];
		}
	}

	// Δ output / Δ input (Relu derivative)
	float layer2_delta_activation[512] = {0};
	for (size_t j=0; j<512; ++j) {
		// don't have sum inputs, but can just assume if output 0 then input was <= 0
		layer2_delta_activation[j] = (layer2out[j] > 0 ? 1 : 0);
	}

	// Δ input / Δ weight ()
	// Δ layer1out / Δ weight => layer1out (i.e. layer 2's inputs, sans the weights)

	// Δ Error_total / Δ weight(s)
	float weight1_deltas[16][512] = {0};
	for (size_t i=0; i<16; ++i) {
		for (size_t j=0; j<512; ++j) {
			weight1_deltas[i][j] = layer2_delta_err[j] * layer2_delta_activation[j] * layer1out[i];
		}
	}

	//------------------------ for updating my weights ----------

	// i do the actual update later (w/ learning rate) after i gather up these over training batch
	for (size_t i=0; i < 16; ++i) {
		for (size_t j=0; j<512; ++j) {
			weightDeltas1[i][j] += weight1_deltas[i][j];
		}
	}
	for (size_t i=0; i < 512; ++i) {
		for (size_t j=0; j<1024; ++j) {
			weightDeltas2[i][j] += weight2_deltas[i][j];
		}
	}
	for (size_t i=0; i < 1024; ++i) {
		for (size_t j=0; j<784; ++j) {
			weightDeltas3[i][j] += weight3_deltas[i][j];
		}
	}

	//------ layer1 (actual inputs from digiCaps) error math

	// Δ Error_total / Δ layer1out ()
	float layer1_delta_err[16] = {0};
	for (size_t j=0; j<16; ++j) {
		for (size_t i=0; i<512; ++i) {
			layer1_delta_err[j] += 	layer2_delta_err[i] * layer2_delta_activation[i] * dweights1[j][i];
		}
	}
	for (size_t j=0; j<16; ++j) {
		recon_errs[j] = layer1_delta_err[j] * -.0005; //* -.0005;
	}

	//------- build the reconstructed image ----
	if (img_out != nullptr) {
		for (size_t r=0; r<28; ++r) {
			for (size_t c=0; c<28; ++c) {
				float x = layer4out[ (r*28)+c ] * 255.;
				if (x > 255) { x = 255; }
				if (x < 0) { x = 0; }
				img_out->at<uint8_t>(r,c) = static_cast<uint8_t>(x);
			}
		}
	}
}


void Reconstruct::update_weights()
{
	// w = w - learningRate * delta;
	for (size_t i=0; i < 16; ++i) {
		for (size_t j=0; j<512; ++j) {
			weightVelocities1[i][j] = .8 * weightVelocities1[i][j] + .2 * weightDeltas1[i][j];
			dweights1[i][j] -= LEARNING_WEIGHT * weightVelocities1[i][j];
			weightDeltas1[i][j] = 0.0;
		}
	}
	for (size_t i=0; i < 512; ++i) {
		for (size_t j=0; j<1024; ++j) {
			weightVelocities2[i][j] = .8 * weightVelocities2[i][j] + .2 * weightDeltas2[i][j];
			dweights2[i][j] -= LEARNING_WEIGHT * weightVelocities2[i][j];
			weightDeltas2[i][j] = 0.0;
		}
	}
	for (size_t i=0; i < 1024; ++i) {
		for (size_t j=0; j<784; ++j) {
			weightVelocities3[i][j] = .8 * weightVelocities3[i][j] + .2 * weightDeltas3[i][j];
			dweights3[i][j] -= LEARNING_WEIGHT * weightVelocities3[i][j];
			weightDeltas3[i][j] = 0.0;
		}
	}	
}

// just used this 1 time to store initial weights in a file
void Reconstruct::create_weights()
{
	// // float dweights1[16][512];
    // static random_device rd;
    // static mt19937 gen(rd());
  	// std::normal_distribution<float> dis1(0.0, 0.125);
	// for (size_t row=0; row<16; ++row) {
	// 	for (size_t col=0; col<512; ++col) {
	// 		dweights1[row][col] = dis1(gen);
	// 	}
	// }

	// std::normal_distribution<float> dis2(0.0, .0625);
	// for (size_t row=0; row<512; ++row) {
	// 	for (size_t col=0; col<1024; ++col) {
	// 		dweights2[row][col] = dis2(gen);
	// 	}
	// }

	// std::uniform_real_distribution<float> dis3(-0.06, 0.06);
	// for (size_t row=0; row<1024; ++row) {
	// 	for (size_t col=0; col<784; ++col) {
	// 		dweights3[row][col] = dis3(gen);
	// 	}
	// }
	
}

// just used this 1 time to store initial weights in a file
void Reconstruct::write_weights_to_files() {
	
	// float dweights1[16][512];
	std::ofstream file1(DECODE_WEIGHTS_FILE_1, std::ios::out);
	for (size_t row=0; row<16; ++row) {
		for (size_t col=0; col<512; ++col) {
			file1 << dweights1[row][col] << ' ';
		}
		file1 << '\n';
	}
	file1.close();

	// float dweights2[512][1024];
	std::ofstream file2(DECODE_WEIGHTS_FILE_2, std::ios::out);
	for (size_t row=0; row<512; ++row) {
		for (size_t col=0; col<1024; ++col) {
			file2 << dweights2[row][col] << ' ';
		}
		file2<< '\n';
	}
	file2.close();

	// float dweights3[1024][784];
	std::ofstream file3(DECODE_WEIGHTS_FILE_3, std::ios::out);
	for (size_t row=0; row<1024; ++row) {
		for (size_t col=0; col<784; ++col) {
			file3 << dweights3[row][col] << ' ';
		}
		file3 << '\n';
	}
	file3.close();
}

// loading weights from file so i can do dev w/ discriminate performance/outputs
void Reconstruct::read_weights_from_files() {
	std::string linestr;
	size_t row = 0;

	// float dweights1[16][512];
	{
	std::ifstream file1(DECODE_WEIGHTS_FILE_1, std::ios::in);
    if (file1.is_open() == false) {
        std::cerr << "error read 1!\n";
    }
	while (std::getline(file1, linestr)) {
		std::istringstream iss(linestr);
		for (size_t col=0; col<512; ++col) {
			iss >> dweights1[row][col];
		}
		if (++row == 16) {
			break;
		}
	}
	file1.close();
}
	// float dweights2[512][1024];
	{
	std::ifstream file2(DECODE_WEIGHTS_FILE_2, std::ios::in);
    if (file2.is_open() == false) {
        std::cerr << "error read 2!\n";
    }
	row = 0;
	while (std::getline(file2, linestr)) {
	    std::istringstream iss(linestr);
		for (size_t col=0; col<1024; ++col) {
			iss >> dweights2[row][col];
		}
		if (++row == 512) {
			break;
		}
	}
	file2.close();
	}
	// float dweights3[1024][784];
	std::ifstream file3(DECODE_WEIGHTS_FILE_3, std::ios::in);
    if (file3.is_open() == false) {
        std::cerr << "error read 3!\n";
    }
	row = 0;
	while (std::getline(file3, linestr)) {
	    std::istringstream iss(linestr);
		for (size_t col=0; col<784; ++col) {
			iss >> dweights3[row][col];
		}
		if (++row == 1024) {
			break;
		}
	}
	file3.close();

	// networkOutput[i] * (1-networkOutput[i]) * (truth[i] - networkOutput[i]);
}

