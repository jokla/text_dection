// ./cvv_text_det nature.png nature1.png

extern "C" {
#include "ccv.h"
}

#include <opencv2/opencv.hpp>


using namespace cv;
int main(){
    Mat img_ocv;
    img_ocv = imread("python.png",CV_LOAD_IMAGE_GRAYSCALE | CCV_IO_NO_COPY);
    ccv_dense_matrix_t* image = 0;
    ccv_read(img_ocv.data, &image, CCV_IO_GRAY_RAW, img_ocv.rows, img_ocv.cols, img_ocv.step[0]);
    ccv_write(image,"python_gray.png",0,CCV_IO_PNG_FILE, 0);
    return 0;
}
