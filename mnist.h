#ifndef PANTHRO_MNIST_H
#define PANTHRO_MNIST_H
#include <vector>
#include <opencv2/core.hpp>

using std::vector;

// reads MNIST image files
class Mnist {
    public:
        static Mnist& getInstance() {
            static Mnist instance; 
            return instance;
        }
        ~Mnist();
        Mnist(Mnist const&) = delete;
        void operator=(Mnist const&) = delete;
        uint32_t numImages() { return image_cnt; }
        uint32_t numRowsPerImage() { return image_nrows; }
        uint32_t numColsPerImage() { return image_ncols; }
        const std::vector< cv::Mat > getImages() const { return images; }
        const std::vector< int > getLabels() const { return labels; }
        const cv::Mat* getImageAt(size_t idx) { return &images[idx]; }
        const int getLabelAt(size_t idx) { return labels[idx]; }
        
    private:
        Mnist();
        void load_images();
        std::vector< cv::Mat > images;
        std::vector< int > labels;
        uint32_t image_cnt;//The number of images in the mnist image set file 
        uint32_t image_nrows;//The number of rows in the image  
        uint32_t image_ncols;//The number of image columns  
};

#endif // PANTHRO_MNIST_H