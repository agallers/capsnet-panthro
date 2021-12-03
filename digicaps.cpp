#include "digicaps.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdio.h>


using cv::Mat;

#define LEARNING_RATE 0.02

static const char* weight_files[] = {
    "../data/w0.txt",
    "../data/w1.txt",
    "../data/w2.txt",
    "../data/w3.txt",
    "../data/w4.txt",
    "../data/w5.txt",
    "../data/w6.txt",
    "../data/w7.txt",
    "../data/w8.txt",
    "../data/w9.txt",
};

static void helper_init_weights(const char* filename, cv::Mat *w);


DigiCaps::DigiCaps()
{
    for(auto i=0; i<10; ++i) {
        for (auto j=0; j<1152; ++j) {
            weightDeltas[i][j] = Mat(cv::Size(16,8), CV_32FC1, cv::Scalar(0));
            weightVelocities[i][j] = Mat(cv::Size(16,8), CV_32FC1, cv::Scalar(0));
        }
        weights[i] = new cv::Mat[1152];
        helper_init_weights(weight_files[i], weights[i]);
    }
}

void DigiCaps::forward_prop(const cv::Mat *u, const int label, float results[10][16])
{
    // u => û (10 of them, weights from u + extra 8 values)
    std::vector< cv::Mat > uhat(10);
	for (auto i=0; i<10; ++i) {
        uhat[i] = Mat(1152,16,CV_32FC1);
		for (auto j=0; j<1152; ++j) {
            uhat[i].row(j) = u->row(j) * weights[i][j];
		}
	}

    // initialize coefficients (for next step) evenly
	float coefficients[10][1152];
	float bees[10][1152];
	for(size_t i=0; i<10; ++i) {
		for(size_t j=0; j<1152; ++j) {
			coefficients[i][j] = 0.1;
			bees[i][j] = 0.0;
		}
	}

	// route - really just computing the coefficients here
	for(size_t i=0; i<1152; ++i) {
		for (size_t iteration=0; iteration<3; ++iteration) { // diminishing returns after 3or4 loops

			// softmax coefficients
			if (iteration > 0) { // can skip first time
				long double sum_b_exps = 0.0;
				for (size_t j=0; j<10; ++j) {
					sum_b_exps += exp(bees[j][i]);
				}
				for (size_t j=0; j<10; ++j) {
					coefficients[j][i] = exp(bees[j][i]) / sum_b_exps;
				}
			} else if (iteration==2) {
                break;
            }

			// weighted sum
            Mat temp = cv::Mat(1,16,uhat[0].type(),cv::Scalar(0));
			for(size_t j=0; j<10; ++j) {
				temp += uhat[j].row(i) * coefficients[j][i];
			}

			// squash
			long double lengthSquared = cv::norm( temp.row(0), cv::NORM_L2SQR);
			long double squishScalar = lengthSquared / (lengthSquared+1);
			long double normalizer = sqrt(lengthSquared) + 1e-8;
			temp /= normalizer;
			temp *= squishScalar;

			// update coefficients
			for(size_t j=0; j<10; ++j) {
				bees[j][i] += temp.dot(uhat[j].row(i));
			}
		}
	}

    // get the result vectors for all 10 digits (each vector 1x16, see whitepaper)
    cv::Mat outputs[10]; // for speed
    for (size_t i=0; i<10; ++i) {

        // sum all 1152 for a digiCap
        outputs[i] = cv::Mat(1,16,uhat[0].type(),cv::Scalar(0));
        for (size_t j=0; j<1152; ++j) {
			outputs[i] += uhat[i].row(j) * coefficients[i][j];
		}

        // squash
        long double lengthSquared = cv::norm( outputs[i].row(0), cv::NORM_L2SQR);
        long double squishScalar = lengthSquared / (lengthSquared+1);
        outputs[i] *= squishScalar / (sqrt(lengthSquared) + 1e-8);
        for (auto j=0; j<16; ++j) {
            results[i][j] = outputs[i].at<float>(0,j);
        }
    }

}

void DigiCaps::backward_prop(const cv::Mat *u, const int label, const float results[10][16], float *recon_errors)
{
    // Compute error for each digit
    std::vector< cv::Mat > errors(10);
    for (int i=0; i<10; ++i) {
        
        // unit vectors 
        double lengthSq = 0;
        for (int j=0; j<16; ++j) {
            lengthSq += pow(results[i][j],2);
        }
        double length = sqrt(lengthSq);

        // derivative of the squash function (is 2x/[(1+x^2)^2])
        double activationDerivativeLength = (2.0 * length) / pow(lengthSq+1,2);

        // derivative of the loss function
        double errorGradient = 0.0;
        {
            double t_k = (i==label ? 1.0 : 0.0);
            double m_plus = 0.9;
            double m_minus = 0.1;
            double lambda = 0.5;
            if (length < m_plus) {
                if (length <= m_minus) {
                    errorGradient = -2. * t_k * (m_plus - length);
                } else {
                    errorGradient = 2. * ((lambda * (t_k - 1) * (m_minus - length)) + t_k * (length - m_plus));
                }
            } else {
                errorGradient = 2. * lambda * (t_k - 1) * (m_minus - length);
            }
        }

        // loss fxn defined in the whitepaper
        double rawMarginLoss = 0.0;
        if (i==label) {
            rawMarginLoss = pow( std::max(0.0, 0.9-length), 2);
        } else {
            rawMarginLoss = 0.5 * pow( std::max(0.0,length-0.1), 2);
        }

        // add them up
        errors[i] = cv::Mat(1,16,CV_32FC1);
        for (int j=0; j<16; ++j) { // make it a unit vector
            errors[i].at<float>(0,j) = results[i][j]/length;
        }
        errors[i] *= activationDerivativeLength * errorGradient * rawMarginLoss * LEARNING_RATE;
        
        // if doing image reconstruction loss, add that error too
        if (i==label && recon_errors != nullptr) {
            for (int j=0; j<16; ++j) {
                errors[i].at<float>(0,j) += (results[i][j]/length) * recon_errors[j] * LEARNING_RATE;
            }
        }
    }

    // then backprop
    Mat ut;
    for (int j=0; j<1152; ++j) {
        cv::transpose(u->row(j), ut);
        for (int i=0; i<10; ++i) {
            weightDeltas[i][j] -= ut * errors[i];// * coefficients[i][j]; // i think i forgot to include coefficients?
        }
    }
}

void DigiCaps::updateWeights()
{
    for (int i = 0; i < 10; i++) {
        for(int j = 0; j < 1152; ++j) {
            weightVelocities[i][j] = 0.9 * weightVelocities[i][j] + 0.1 * weightDeltas[i][j];
            weights[i][j] += weightVelocities[i][j];
            weightDeltas[i][j] = 0;
        }
    }
}

static void helper_init_weights(const char* filename, cv::Mat *w)
{
	std::ifstream file(filename, std::ios::in);
    if (file.is_open() == false) {
        std::cerr << "error read weightss!\n";
    }
	std::string linestr;
	size_t row = 0;
	for (size_t i=0; i<1152; ++i) {
		w[i] = cv::Mat(8,16,CV_32FC1);
		for (size_t j=0; j<8; ++j) {
			std::getline(file, linestr);
			std::istringstream iss(linestr);
			for (size_t k=0; k<16; ++k) {
				iss >> (w[i].at<float>(j,k));
			}
		}
	}
	file.close();
}

void DigiCaps::saveWeightsToFile()
{
    for (auto x=0; x<10; ++x) {
        std::ofstream file(weight_files[x], std::ofstream::trunc);
        if (file.is_open() == false) {
            std::cerr << "error write weightss!\n";
        }
        char str[10];
        for (size_t i=0; i<1152; ++i) {
            for (size_t j=0; j<8; ++j) {
                for (size_t k=0; k<16; ++k) {
                    sprintf(str," %-1.4f", weights[x][i].at<float>(j,k));
                    if (str[1] != '-') {
                        file << ' ';
                    }
                    file << str;
                }
                file << '\n';
            }
        }
        file.close();
    }
}
