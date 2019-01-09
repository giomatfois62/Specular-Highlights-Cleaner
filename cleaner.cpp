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
    //const char* img_paths[] = {"22/1.pgm","22/2.pgm","22/3.pgm","22/4.pgm","22/5.pgm","22/6.pgm",
    //"22/7.pgm","22/8.pgm","22/9.pgm","22/10.pgm","22/11.pgm","22/12.pgm","22/13.pgm"};
    //int NUM_IMG = 13;
    const char* img_paths[] = {"test/1.jpg","test/2.jpg","test/3.jpg","test/4.jpg"};
    int NUM_IMG = 4;
    
    cout << "Loading images\n";
    
    vector<Mat> imgs;
    for(int i = 0; i < NUM_IMG; ++i) {
        Mat img = imread(img_paths[i], 0);
        img.convertTo(img, CV_64F);
        normalize(img, img, 0, 1, cv::NORM_MINMAX);
        imgs.push_back(img);
    }
    
    cout << "Computing gradients\n";
    
    vector<Mat> gradX, gradY;
    for(int i = 0; i < NUM_IMG; ++i) {
        Mat x,y;
        Sobel(imgs[i], x, CV_64F, 1 , 0, 5);
        Sobel(imgs[i], y, CV_64F, 0 , 1, 5);
        gradX.push_back(x);
        gradY.push_back(y);
    }
    
    int rows = imgs[0].rows;
    int cols = imgs[0].cols;
    
    cout << "Computing gradients median\n";
    
    Mat medianX(rows, cols, CV_64F);
    Mat medianY(rows, cols, CV_64F);
    Mat median(rows, cols, CV_64F);
    Mat imax(rows, cols, CV_64F);
    
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            vector<double> vx, vy;
            vector<double> im;
            
            for(int k = 0; k < NUM_IMG; ++k) {
                vx.push_back(gradX[k].at<double>(i,j));
                vy.push_back(gradY[k].at<double>(i,j));
                im.push_back(imgs[k].at<double>(i,j));
            }
            
            sort(vx.begin(), vx.end());
            sort(vy.begin(), vy.end());
            sort(im.begin(), im.end());
            
            medianX.at<double>(i,j) = vx[NUM_IMG/2 - 1];
            medianY.at<double>(i,j) = vy[NUM_IMG/2 - 1];
            
            median.at<double>(i,j) = im[NUM_IMG/2 - 1];
            imax.at<double>(i,j) = im[NUM_IMG - 1]; 
        }
    }
    
    cout << "Solving Poisson problem\n";
    //normalize(median, median, 0, 1, cv::NORM_MINMAX);
    
    Mat medianXX, medianYY;
    Sobel(medianX, medianXX, CV_64F, 1 , 0, 5);
    Sobel(medianY, medianYY, CV_64F, 0 , 1, 5); 
    
    Scalar zero(0.0f);
    Mat u(rows,cols,CV_64F,zero);
    Mat u0(rows,cols,CV_64F,zero);
    //median.copyTo(u0);
    //u0.convertTo(u0, CV_64F);
    //u0.copyTo(u);
    //normalize(u0, u0, 0, 1, cv::NORM_MINMAX); 
    
        /// Create window
    namedWindow( "Original", WINDOW_NORMAL );
    resizeWindow("Original", 600,600);
    namedWindow( "Cleaned", WINDOW_NORMAL );
    resizeWindow("Cleaned", 600,600);
    
    Mat disp;
    normalize(median, disp, 0, 1, cv::NORM_MINMAX);
    imshow("Original", imgs[0]);
    resizeWindow("Original", 600,600);
    
    //cout << "M = "<< endl << " "  << medianXX << endl << endl;
    
    int ITER = 5000;
    for(int k = 0; k < ITER; ++k) {
    	#pragma omp parallel for
        for(int i = 1; i < rows - 1; ++i) {
            for(int j = 1; j < cols - 1; ++j) {
                //u.at<double>(i,j) = (u0.at<double>(i+1,j)+u0.at<double>(i-1,j)+u0.at<double>(i,j+1)+u0.at<double>(i,j-1)
                //-(1.0f/(1.0f))*(medianXX.at<double>(i,j)+medianYY.at<double>(i,j)))/4.0f;
                
                u.at<double>(i,j) = ((2.0f/3.0f)*(u0.at<double>(i+1,j)+u0.at<double>(i-1,j)+u0.at<double>(i,j+1)+u0.at<double>(i,j-1))
                + (1.0f/6.0f)*(u0.at<double>(i+1,j+1)+u0.at<double>(i+1,j-1)+u0.at<double>(i-1,j+1)+u0.at<double>(i-1,j-1))
                -(medianXX.at<double>(i,j) + medianYY.at<double>(i,j)))*(3.0f/10.0f);
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
                

        normalize(u, u, 0, 1, cv::NORM_MINMAX);        
        Mat alfa = imax - u;
    	normalize(alfa, alfa, 0, 1, cv::NORM_MINMAX);
    	for(int i = 0; i < rows; ++i) {
            for(int j = 1; j < cols; ++j) {
            	u.at<double>(i,j) = alfa.at<double>(i,j)*u.at<double>(i,j) + (1.0f -alfa.at<double>(i,j))*imax.at<double>(i,j);
            }
    	}
        

        imshow("Cleaned", u);
        resizeWindow("Cleaned", 600,600);
        waitKey(1);
    }
    
    cout << "Done!\n";  
    
    waitKey(0);
    
    return 0;
}
