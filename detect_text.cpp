// ./cvv_text_det nature.png nature1.png

extern "C" {
#include "ccv.h"
}

#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


static unsigned int get_current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

using namespace cv;
int main(){
    Mat img_ocv;
    //img_ocv = imread("1.jpg",CV_LOAD_IMAGE_GRAYSCALE | CCV_IO_NO_COPY);
    img_ocv = imread("1.jpg", CV_LOAD_IMAGE_COLOR);

    ccv_dense_matrix_t* image = 0;
    //ccv_read(img_ocv.data, &image, CCV_IO_GRAY_RAW, img_ocv.rows, img_ocv.cols, img_ocv.step[0]);
    ccv_read("1.jpg", &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);

    //ccv_read(img_ocv.data, &image,CCV_IO_RGB_RAW | CCV_IO_ANY_RAW | CCV_IO_GRAY, img_ocv.rows, img_ocv.cols, img_ocv.step[0]);

    unsigned int elapsed_time = get_current_time();


    ccv_swt_param_t params = ccv_swt_default_params;

            params.direction = CCV_DARK_TO_BRIGHT;
           // params.direction = CCV_BRIGHT_TO_DARK;



    ccv_dense_matrix_t* swt = 0;
    ccv_swt(image, &swt, CCV_8U, params);

    ccv_matrix_t* vis_swt = 0;

    ccv_visualize(swt, &vis_swt, 0);

    ccv_dense_matrix_t * vis_swt_w =  ccv_get_dense_matrix(vis_swt);

    ccv_write(vis_swt_w, "testDTB.png", 0, CCV_IO_PNG_FILE, 0);

    ccv_array_t* words = ccv_swt_detect_words(image, ccv_swt_default_params);


    elapsed_time = get_current_time() - elapsed_time;
    if (words)
    {

        for (unsigned int i = 0; i < words->rnum; i++)
        {
            ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
            printf("%d %d %d %d\n", rect->x, rect->y, rect->width, rect->height);

            //x, y, x + width, y + height)
            rectangle( img_ocv, Point( rect->x, rect->y ), Point( rect->x + rect->width , rect->y + rect->height ), Scalar(255, 0, 0, 0),2, 8, 0 );

        }
        printf("total : %d in time %dms\n", words->rnum, elapsed_time);
        ccv_array_free(words);
    }
    imshow("Image",img_ocv);
    std::vector<int> compression_params; //vector that stores the compression parameters of the image
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); //specify the compression technique
    compression_params.push_back(100); //specify the compression quality
     cv::imwrite("Result.jpg", img_ocv, compression_params); //write the image to file

    waitKey( 0 );
    ccv_matrix_free(image);

    ccv_drain_cache();

    return 0;
}
