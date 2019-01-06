#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
    const char* img_paths[] = {"22/1.pgm","22/2.pgm","22/3.pgm","22/4.pgm","22/5.pgm","22/6.pgm",
    "22/7.pgm","22/8.pgm","22/9.pgm","22/10.pgm","22/11.pgm","22/12.pgm","22/13.pgm"};
    int NUM_IMG = 13;
    //const char* img_paths[] = {"13/1.pgm","13/2.pgm","13/3.pgm","13/4.pgm","13/5.pgm"};
    //int NUM_IMG = 5;
    
    cout << "Loading images\n";
    
    vector<Mat> imgs;
    for(int i = 0; i < NUM_IMG; ++i) {
        Mat img = imread(img_paths[i], 0);
        imgs.push_back(img);
    }
    
    cout << "Computing gradients\n";
    
    vector<Mat> gradX, gradY;
    for(int i = 0; i < NUM_IMG; ++i) {
        Mat x,y;
        Scharr(imgs[i], x, CV_64F, 1 , 0 );
        Scharr(imgs[i], y, CV_64F, 0 , 1 );
        gradX.push_back(x);
        gradY.push_back(y);
    }
    
    int rows = imgs[0].rows;
    int cols = imgs[0].cols;
    
    cout << "Computing gradients median\n";
    
    Mat medianX(rows, cols, CV_64F);
    Mat medianY(rows, cols, CV_64F);
    
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            vector<double> vx, vy;
            
            for(int k = 0; k < NUM_IMG; ++k) {
                vx.push_back(gradX[k].at<double>(i,j));
                vy.push_back(gradY[k].at<double>(i,j));
            }
            
            sort(vx.begin(), vx.end());
            sort(vy.begin(), vy.end());
            
            medianX.at<double>(i,j) = vx[NUM_IMG/2 - 1];
            medianY.at<double>(i,j) = vy[NUM_IMG/2 - 1];
        }
    }
    
    cout << "Solving Poisson problem\n";
    
    Mat medianXX, medianYY;
    Scharr(medianX, medianXX, CV_64F, 1 , 0 );
    Scharr(medianY, medianYY, CV_64F, 0 , 1 ); 
    
    Scalar zero(0.0f);
    Mat u(rows,cols,CV_64F,zero);
    Mat u0(rows,cols,CV_64F,zero);
    u0 = imread(img_paths[0], 0);
    u0.convertTo(u0, CV_64F);
    u0.copyTo(u);
    //normalize(u0, u0, 0, 1, cv::NORM_MINMAX); 
    
    imshow("Original", imgs[0]);
    
    int ITER =2000;
    for(int k = 0; k < ITER; ++k) {
        for(int i = 1; i < rows - 1; ++i) {
            for(int j = 1; j < cols - 1; ++j) {
                //u.at<double>(i,j) = (u0.at<double>(i+1,j)+u.at<double>(i-1,j)+u0.at<double>(i,j+1)+u.at<double>(i,j-1)
                //-medianXX.at<double>(i,j)-medianYY.at<double>(i,j))/4.0f;
                
                u.at<double>(i,j) = ((2.0f/3.0f)*(u0.at<double>(i+1,j)+u.at<double>(i-1,j)+u0.at<double>(i,j+1)+u.at<double>(i,j-1))
                + (1.0f/6.0f)*(u0.at<double>(i+1,j+1)+u.at<double>(i+1,j-1)+u0.at<double>(i-1,j+1)+u.at<double>(i-1,j-1))
                -medianXX.at<double>(i,j)-medianYY.at<double>(i,j))*(3.0f/10.0f);
            }
        }
        
        for(int i = 0; i < rows; ++i) {
            u.at<double>(i,0) = u.at<double>(i,1);
            u.at<double>(i,cols - 1) = u.at<double>(i, cols - 2);
        }
        
        for(int i = 0; i < cols; ++i) {
            u.at<double>(0, i) = u.at<double>(1, i);
            u.at<double>(rows - 1, i) = u.at<double>(rows - 2, i);
        }
        
        cout << "\r" << k << "/" << ITER <<" Residual: " << setw(10) << norm(u-u0) << flush;
        
        for(int i = 0; i < rows; ++i)
            for(int j = 0; j < cols; ++j)
                u0.at<double>(i,j) = u.at<double>(i,j);
            
        Mat disp;
        normalize(u, disp, 0, 1, cv::NORM_MINMAX);
        imshow("Cleaned", disp);
        waitKey(1);
    }
    
    cout << "Done!\n";
    
    normalize(u, u, 0, 255, cv::NORM_MINMAX);
    
    Mat x,y,abs_grad_x, abs_grad_y;
    
    Scharr(u, x, CV_64F, 1 , 0 );
    Scharr(u, y, CV_64F, 0 , 1 );
    convertScaleAbs( y, abs_grad_y );
    convertScaleAbs( x, abs_grad_x );
    
    Mat grad_cleaned;
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_cleaned );
    
    Scharr(imgs[0], x, CV_64F, 1 , 0 );
    Scharr(imgs[0], y, CV_64F, 0 , 1 );
    convertScaleAbs( y, abs_grad_y );
    convertScaleAbs( x, abs_grad_x );
    
    Mat grad;
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    
    /// Create window
    namedWindow( "Original", CV_WINDOW_AUTOSIZE );
    namedWindow( "Gradient Original", CV_WINDOW_AUTOSIZE );
    namedWindow( "Cleaned", CV_WINDOW_AUTOSIZE );
    namedWindow( "Gradient Cleaned", CV_WINDOW_AUTOSIZE );
    
    normalize(u, u, 0, 1, cv::NORM_MINMAX);
    
    imshow("Original", imgs[0]);
    imshow("Cleaned", u);
    imshow("Gradient Original", grad);
    imshow("Gradient Cleaned", grad_cleaned);
    
    waitKey(0);
    
    return 0;
}