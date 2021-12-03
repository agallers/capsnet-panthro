#ifndef PANTHRO_PRIMARYCAPS_H
#define PANTHRO_PRIMARYCAPS_H

#include <opencv2/core.hpp>

// Primary Capsules (take 28x28 image => convolutions => ...)
class PrimaryCaps {
    public:
        PrimaryCaps(const cv::Mat* img);
        const cv::Mat* get_reshaped_pcaps() const { return &u; } 
    private:
        cv::Mat u; // 1152x8, each row is a vector size 8 whose |length| < 1 
};

#endif // PANTHRO_PRIMARYCAPS_H