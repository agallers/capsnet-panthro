/* Andy Gallerstein (c) 2021, andy.gallerstein@gmail.com */
#include "primarycaps.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define PCAPS_KERNELS_FILE "../data/kernels.txt"

static cv::Mat kernels[256];

void init_kernels();
cv::Mat* convolve(const cv::Mat &input, const cv::Mat &kernel, size_t step_size);

PrimaryCaps::PrimaryCaps(const cv::Mat* mnist_image)
{
    // for persistent convolution of sample data
    init_kernels();

    // Convolve the 1 @ 28x28 image => 256 @ 6x6's (each cell is still a scalar)
    std::vector<cv::Mat*> conv1; // 256 convolutions of original image (6x6, 1 channel)
	for (size_t j=0; j<256; ++j) { // kernels is array 256 9x9 filters
		cv::Mat *m = convolve(*mnist_image, kernels[j], 1); // drops to [20x20]
		conv1.push_back( convolve(*m, kernels[j], 2) ); // drops to [6x6]
		delete m;
	}

    // Reshape from 256 matrix's holding scalars to 32 matrix's holding vectors*8)
    // actually just save a step reshape directly into an 1152,8 matrix
    size_t u_index = 0;
    this->u = cv::Mat(1152,8,CV_32FC1);
    for (size_t row=0; row<6; ++row) { // go through all 36 pixels
        for (size_t col=0; col<6; ++col) { // (6x6=36)
            for (size_t j=0; j<256; j+=8) { // for every existing convolution
                for (size_t k=0; k<8; ++k) { // combine scalars of 8 existing mat's => vector(8)
                     this->u.at<float>(u_index,k) = conv1[j+k]->at<float>(row,col);
                }
				++u_index;
			}
		}
	}

    // squash each vector so its |length| < 1 
    long double lengthSquared, squishScalar, normalizer;
    for (size_t row=0; row<u.rows; ++row) {
        lengthSquared = 0.0;
        for (size_t col=0; col<u.cols; ++col) {
            lengthSquared += (u.at<float>(row,col) * u.at<float>(row,col));
        }
        squishScalar = lengthSquared / (lengthSquared+1.);
        normalizer = sqrt(lengthSquared) + 1e-8;
        for (size_t col=0; col<u.cols; ++col) {
            u.at<float>(row,col) /= normalizer;
            u.at<float>(row,col) *= squishScalar;
        }
    }

    // cleanup (intermediate data)
    for(int i=conv1.size()-1; i>=0; --i) {
        delete conv1[i];
    }
}

cv::Mat* convolve(const cv::Mat &input, const cv::Mat &kernel, size_t step_size)
{
	size_t new_cols = ((input.cols - kernel.cols)/step_size) + 1;
	size_t new_rows = ((input.rows - kernel.rows)/step_size) + 1;
	cv::Mat *output = new cv::Mat(new_rows, new_cols, input.type());
    size_t c,r;
	for (c=0; c<new_cols; ++c) {
		for (r=0; r<new_rows; ++r) {
			/*long*/ double d = kernel.dot( input.colRange( c*step_size, c*step_size + kernel.cols ).rowRange( r*step_size, r*step_size + kernel.rows) );
            if (d < 0) { d = 0; }
            else if (d > 255) { d = 255; }
            //else if (isnan(d) || isinf(d)) { d = 0; }
			output->at<float>(r,c) = d;
		}
	}
	return output;
}

void init_kernels()
{
    static int initialized = 0;
    if (initialized) {
        return;
    }
    initialized = 1;
    std::ifstream file(PCAPS_KERNELS_FILE, std::ios::in);
    if (file.is_open() == false) {
        std::cerr << "error read kernels!\n";
    }
    std::string linestr;
    size_t row = 0;
    for (size_t i=0; i<256; ++i) {	
        kernels[i] = cv::Mat(9,9,CV_32F); 
        for (size_t j=0; j<9; ++j) {
            std::getline(file, linestr);	
            std::istringstream iss(linestr);
            for(size_t k=0; k<9; ++k) {
                iss >> kernels[i].at<float>(j,k);
            }
        }
    }
    file.close();
}
