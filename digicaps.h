/* Andy Gallerstein (c) 2021, andy.gallerstein@gmail.com */

#ifndef PANTHRO_DIGICAPS_H
#define PANTHRO_DIGICAPS_H

#include <opencv2/core.hpp>

// Digit Capsules 
class DigiCaps {
    public:
        static DigiCaps& getInstance() {
            static DigiCaps instance; 
            return instance;
        }
        ~DigiCaps() {}
        void forward_prop(const cv::Mat *u, const int label, float results[10][16]);
        void backward_prop(const cv::Mat *u, const int label, const float results[10][16], float *recon_errs);
        void updateWeights();
        void saveWeightsToFile();
        
    private:
        DigiCaps();
        cv::Mat *weights[10];
        cv::Mat weightDeltas[10][1152];
        cv::Mat weightVelocities[10][1152];
};

#endif // PANTHRO_DIGICAPS_H
