// ./cvv_text_det nature.png nature1.png

extern "C" {
#include "ccv.h"
}

#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <tesseract/strngs.h>
#include <tesseract/genericvector.h>


static unsigned int get_current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

using namespace cv;
int main(){
    Mat img_ocv;
    img_ocv = imread("TestImage1.jpg",CV_LOAD_IMAGE_GRAYSCALE | CCV_IO_NO_COPY);
    ccv_dense_matrix_t* image = 0;
    ccv_read(img_ocv.data, &image, CCV_IO_GRAY_RAW, img_ocv.rows, img_ocv.cols, img_ocv.step[0]);
    // ccv_read("1.jpg", &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);

    // std::vector<Rect> roi;

    Mat img_disp = img_ocv.clone();




    // turn of dictionaries -> only possible during init
    GenericVector<STRING> vars_vec;
    vars_vec.push_back("load_system_dawg");
    vars_vec.push_back("load_freq_dawg");
    vars_vec.push_back("load_punc_dawg");
    vars_vec.push_back("load_number_dawg");
    vars_vec.push_back("load_unambig_dawg");
    vars_vec.push_back("load_bigram_dawg");
    vars_vec.push_back("load_fixed_length_dawgs");
    //vars_vec.push_back("user_patterns_suffix");

    GenericVector<STRING> vars_values;
    vars_values.push_back("F");
    vars_values.push_back("F");
    vars_values.push_back("F");
    vars_values.push_back("F");
    vars_values.push_back("F");
    vars_values.push_back("F");
    vars_values.push_back("F");
    //vars_values.push_back("pharma-words");

    // Tesseract
    tesseract::TessBaseAPI tess;
    if( tess.Init(NULL, "eng", tesseract::OEM_DEFAULT, NULL, 0, &vars_vec, &vars_values, false))
    {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }
    printf("Using tesseract c++ API: %s\n", tess.Version());
    tess.SetVariable("tessedit_char_whitelist",
                     "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                     "0123456789");
    tess.SetVariable("language_model_penalty_non_dict_word", "0");
    tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);

    // Check if change of init parameters was successful
    STRING var_value;
    tess.GetVariableAsString("load_system_dawg", &var_value);
    printf("Variable 'load_system_dawg' is set to '%s'\n", var_value.string());



    unsigned int elapsed_time = get_current_time();
    ccv_array_t* words = ccv_swt_detect_words(image, ccv_swt_default_params);
    elapsed_time = get_current_time() - elapsed_time;
    if (words)
    {


        for (unsigned int i = 0; i < words->rnum; i++)
        {
            ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
            printf("%d %d %d %d\n", rect->x, rect->y, rect->width, rect->height);

            cv::Rect roi_rect(rect->x-5, rect->y-5,  rect->width+10, rect->height+10);

            //roi.push_back(roi_rect);

            if ( (rect->width * rect->height) >= 800 )
            {

                //x, y, x + width, y + height)
                rectangle( img_disp, Point( rect->x, rect->y ), Point( rect->x + rect->width , rect->y + rect->height ), Scalar(255, 0, 0, 0),2, 8, 0 );

                Mat image_roi = img_ocv(roi_rect);
                Mat binary_roi;
                threshold(image_roi,binary_roi, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

                //imshow("Image cropped",image_roi);
                //imwrite("cropped.jpg"+i, image_roi);
                imshow("Image cropped bn", binary_roi);
                //imwrite("cropped_bn.jpg"+i, binary_roi);

                tess.SetImage((uchar*)binary_roi.data, binary_roi.cols, binary_roi.rows, 1, binary_roi.cols);

                // Get the text
                char* out = tess.GetUTF8Text();
                std::cout << "Result: " << out << std::endl;
                waitKey( 0 );

            }





        }
        printf("total : %d in time %dms\n", words->rnum, elapsed_time);
        ccv_array_free(words);
    }
    imshow("Image",img_disp);

    //    Mat image_roi = img_ocv(roi[0]);
    //    Mat binary_roi;
    //    threshold(image_roi,binary_roi, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    //    imshow("Image cropped",image_roi);
    //    imwrite("cropped.jpg", image_roi);
    //    imshow("Image cropped bn",binary_roi);
    //    imwrite("cropped_bn.jpg", binary_roi);

    //    // Pass it to Tesseract API
    //    tesseract::TessBaseAPI tess;
    //    tess.Init(NULL, "fra", tesseract::OEM_DEFAULT);
    //    tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    //    tess.SetImage((uchar*)binary_roi.data, binary_roi.cols, binary_roi.rows, 1, binary_roi.cols);

    //    // Get the text
    //    char* out = tess.GetUTF8Text();
    //    std::cout << "Result: " << out << std::endl;

    waitKey( 0 );
    ccv_matrix_free(image);

    ccv_drain_cache();

    return 0;
}
