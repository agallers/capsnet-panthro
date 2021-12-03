/* Andy Gallerstein (c) 2021, andy.gallerstein@gmail.com */

#ifndef PANTHRO_RECONSTRUCT_H
#define PANTHRO_RECONSTRUCT_H
#include <vector>
#include <opencv2/core.hpp>

class Reconstruct {
    public:
        static Reconstruct& getInstance() {
            static Reconstruct instance; 
            return instance;
        }
        ~Reconstruct();
        void fwd_and_bwd_prop(float input16[16], int label, const cv::Mat *img_in, cv::Mat *img_out);
        void update_weights();
void write_weights_to_files();
        float recon_errs[16];
    private:
        Reconstruct();
        
        void read_weights_from_files();
        void create_weights();

        // ==> 512 nodes (relu)
        float **dweights1;//[16][512]; // 1st layer weights (relu)
        float **dweights2;//[512][1024]; // 2nd layer (relu)
        float **dweights3;//[1024][784]; // final layer (sigmoid * 255)

        float **weightDeltas1;//[16][512];
        float **weightDeltas2;//[512][1024];
	    float **weightDeltas3;//[1024][784];
		
	    float **weightVelocities1;//[16][512];
        float **weightVelocities2;//[512][1024];
	    float **weightVelocities3;//[1024][784];
};

#endif // PANTHRO_RECONSTRUCT_H
