#include "mnist.h"
#include <iostream>
#include <fstream>

#define IMAGES_FILE "../data/mnist/t10k-images-idx3-ubyte"
#define LABELS_FILE "../data/mnist/t10k-labels-idx1-ubyte"

#define TRAIN_IMAGES_FILE "../data/mnist/train-images-idx3-ubyte"
#define TRAIN_LABELS_FILE "../data/mnist/train-labels-idx1-ubyte"


uint32_t swap_endian(uint32_t val);

using namespace std;

Mnist::Mnist() : image_cnt(0), image_nrows(0), image_ncols(0)
{
    load_images();
}

Mnist::~Mnist()
{

}

void Mnist::load_images()
{
    //Read image files and label files in mnist database in binary format  
    std::ifstream mnist_image(IMAGES_FILE, std::ios::in | std::ios::binary);
    std::ifstream mnist_label(LABELS_FILE, std::ios::in | std::ios::binary);
    if (mnist_image.is_open() == false) {
        std::cerr << "open mnist image file error!\n";
        return;
    }
    if (mnist_label.is_open() == false) {
        std::cerr << "open mnist label file error!\n";
        return;
    }

    uint32_t magic;//Magic number in the file (magic number)  
    uint32_t num_items;//The number of images in the mnist image set file  
    uint32_t num_label;//Number of labels in the mnist label set file  
    uint32_t rows;//The number of rows in the image  
    uint32_t cols;//The number of image columns  

    //Read the magic number  
    mnist_image.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051)  {
        std::cerr << "this is not the mnist image file\n";
        return;
    }
    mnist_label.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049) {
        std::cerr << "this is not the mnist label file\n";
        return;
    }

    //Number of read images/tags  
    mnist_image.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    mnist_label.read(reinterpret_cast<char*>(&num_label), 4);
    num_label = swap_endian(num_label);
    if (num_items != num_label) {
        std::cerr << "the image file and label file are not a pair\n";
    }

    //Read the number of image rows and columns  
    mnist_image.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    mnist_image.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    //Read the image  
    for (int i = 0; i != num_items; i++) {
        char* pixels = new char[rows * cols];
        mnist_image.read(pixels, rows * cols);
        char label;
        mnist_label.read(&label, 1);
        labels.push_back((int)label);
        images.push_back(cv::Mat(rows, cols, CV_8UC1));
        for (int m = 0; m != rows; m++) {
            uchar* ptr = images[i].ptr<uchar>(m);
            for (int n = 0; n != cols; n++) {
                ptr[n] = (uint8_t)pixels[m * cols + n];
            }
        }
		cv::Mat m(rows, cols, CV_32F);
		images[i].convertTo(m, CV_32F);
		m.copyTo(images[i]);
    }
    image_cnt = images.size();//The number of images in the mnist image set file 
    image_nrows = rows;//The number of rows in the image  
    image_ncols = cols;//The number of image columns
}

// helper
uint32_t swap_endian(uint32_t val)
{ 
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}
