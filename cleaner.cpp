#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat resizeKeepAspectRatio(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor)
{
    cv::Mat output;

    double h1 = dstSize.width * (input.rows/(double)input.cols);
    double w2 = dstSize.height * (input.cols/(double)input.rows);
    if( h1 <= dstSize.height) {
        cv::resize( input, output, cv::Size(dstSize.width, h1));
    } else {
        cv::resize( input, output, cv::Size(w2, dstSize.height));
    }

    int top = (dstSize.height-output.rows) / 2;
    int down = (dstSize.height-output.rows+1) / 2;
    int left = (dstSize.width - output.cols) / 2;
    int right = (dstSize.width - output.cols+1) / 2;

    cv::copyMakeBorder(output, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor );

    return output;
}

Mat laplacianMatrix(size_t size, int origRows)
{
    Mat lap(size, size, CV_32F, Scalar(0.0f));

    Mat T(origRows, origRows,CV_32F, Scalar(0.0f));
    for(int i = 0; i < origRows; ++i) {
        T.at<float>(i,i) = -4.0f;

        if(i-1 >= 0)
            T.at<float>(i,i-1) = 1.0f;

        if(i+1 < origRows)
            T.at<float>(i,i+1) = 1.0f;
    }

    Mat I(origRows,origRows,CV_32F, Scalar(0.0f));
    for(int i = 0; i < origRows; ++i)
        I.at<float>(i,i) = 1.0f;

    for(int i = 0; i < origRows; ++i) {
        for(int j = 0; j < origRows; ++j) {

            Rect block(j*(origRows),i*(origRows), origRows, origRows);

            if(i == j)
                T.copyTo(lap(block));

            if(j == i+1 || j == i-1)
               I.copyTo(lap(block));

        }
    }

    return lap;
}

Mat restrictionOp(int N)
{
    int n = sqrt(N);
    int coarse_n = (n-1)/2;
    int coarse_N = coarse_n*coarse_n;

    Mat res(coarse_N, N, CV_32F, Scalar(0.0f));

    int k = 0;
    for(int j = 1; j < n-1; j += 2)  {
        for(int i = 1; i < n-1; i += 2) {
            int fk = j*n + i;

            res.at<float>(k, fk - n -1) = 0.0625;
            res.at<float>(k, fk - n) = 0.125;
            res.at<float>(k, fk - n + 1) = 0.0625;

            res.at<float>(k, fk - 1)     = 0.125;
            res.at<float>(k, fk    )     = 0.25;
            res.at<float>(k, fk + 1)     = 0.125;

            res.at<float>(k, fk + n - 1) = 0.0625;
            res.at<float>(k, fk + n   )  = 0.125;
            res.at<float>(k, fk + n + 1) = 0.0625;

            k++;
        }
    }

    return res;
}

Mat restrictMat(const Mat& mat)
{
    int rows = (mat.rows+1) / 2;
    int cols = (mat.cols+1) / 2;

    Mat res(rows, cols, CV_32F, Scalar(0.0f));

    for(int i = 1; i < rows - 1; ++i) {
        for(int j = 1; j < cols - 1; ++j) {
            res.at<float>(i,j) = (1.0f / 4.0f) * (mat.at<float>(2*i,2*j)) +
                                 (1.0f / 8.0f) * (mat.at<float>(2*i+1,2*j) + mat.at<float>(2*i-1,2*j) + mat.at<float>(2*i,2*j+1) + mat.at<float>(2*i,2*j-1)) +
                                 (1.0f / 16.0f) * (mat.at<float>(2*i+1,2*j+1) + mat.at<float>(2*i-1,2*j-1) + mat.at<float>(2*i-1,2*j+1) + mat.at<float>(2*i+1,2*j-1));
        }
    }


    return res;
}

Mat prolongMat(const Mat& mat)
{
    int rows = mat.rows * 2-1;
    int cols = mat.cols * 2-1;

    Mat res(rows, cols, CV_32F, Scalar(0.0f));

    for(int i = 1; i < rows - 1; ++i) {
        for(int j = 1; j < cols - 1; ++j) {
            if(i%2 == 0 && j%2 == 0)
                res.at<float>(i,j) = 1.0f * mat.at<float>(i/2,j/2);
            else if(i%2 == 0)
                res.at<float>(i,j) = 0.5 * (mat.at<float>(i/2,j/2) + mat.at<float>(i/2,j/2+1));
            else if(j%2 == 0)
                res.at<float>(i,j) = 0.5 * (mat.at<float>(i/2,j/2) + mat.at<float>(i/2+1,j/2));
            else
                res.at<float>(i,j) = 0.25 * (mat.at<float>(i/2,j/2) + mat.at<float>(i/2+1,j/2+1) + mat.at<float>(i/2+1,j/2) + mat.at<float>(i/2,j/2+1));
        }
    }

    return res;
}

Mat residual(const Mat& u, const Mat& f)
{
    int rows = u.rows;
    int cols = u.cols;

    Mat res(rows, cols, CV_32F, Scalar(0.0f));

    float h = 1.0f/(max(rows,cols)-1);

    for(int i = 1; i < rows - 1; ++i)
        for(int j = 1; j < cols - 1; ++j)
            res.at<float>(i, j) = h*h*f.at<float>(i,j) - (u.at<float>(i-1,j) + u.at<float>(i,j-1) + u.at<float>(i+1,j) + u.at<float>(i,j+1) - 4 * u.at<float>(i,j));

    return res;
}

Mat residual(const Mat &L, const Mat& u, const Mat& f)
{
    int rows = u.rows;
    int cols = u.cols;

    Mat res(rows, cols, CV_32F, Scalar(0.0f));

    float h = 1.0f/(max(rows,cols)-1);
    h = 1.0f;

    for(int i = 1; i < rows - 1; ++i) {
        for(int j = 1; j < cols - 1; ++j) {
                        res.at<float>(i,j) = h * h * f.at<float>(i,j);

			res.at<float>(i,j) -= L.at<float>(0,0) * u.at<float>(i-1,j-1);
			res.at<float>(i,j) -= L.at<float>(0,1) * u.at<float>(i-1,j);
			res.at<float>(i,j) -= L.at<float>(0,2) * u.at<float>(i-1,j+1);
			res.at<float>(i,j) -= L.at<float>(1,0) * u.at<float>(i,j-1);
			res.at<float>(i,j) -= L.at<float>(1,1) * u.at<float>(i,j);
			res.at<float>(i,j) -= L.at<float>(1,2) * u.at<float>(i,j+1);
			res.at<float>(i,j) -= L.at<float>(2,0) * u.at<float>(i+1,j-1);
			res.at<float>(i,j) -= L.at<float>(2,1) * u.at<float>(i+1,j);
			res.at<float>(i,j) -= L.at<float>(2,2) * u.at<float>(i+1,j+1);
		}
	}
            
    return res;
}

void smooth(Mat &u, Mat &f, size_t ITERS)
{
    int rows = u.rows;
    int cols = u.cols;

    float h = 1.0f/(max(rows,cols)-1);

    //#pragma omp parallel for
    for(int k = 0; k < ITERS; ++k) {

        for(int i = 1; i < rows - 1; ++i)
            for(int j = 1; j < cols - 1; ++j)
                u.at<float>(i,j) = 0.25 * (u.at<float>(i-1,j) + u.at<float>(i,j-1) + u.at<float>(i+1,j) + u.at<float>(i,j+1) - h*h*f.at<float>(i,j));
    }
}

void smooth(const Mat &L, Mat &u, const Mat &f, size_t ITERS)
{
    int rows = u.rows;
    int cols = u.cols;

    float h = 1.0f/(max(rows,cols)-1);

    float coeff = 1.0f / L.at<float>(1,1);
    h = 1.0f;

    for(int k = 0; k < ITERS; ++k) {

        for(int i = 1; i < rows - 1; ++i) {
            for(int j = 1; j < cols - 1; ++j) {
                                u.at<float>(i,j) = h*h*f.at<float>(i,j);

				u.at<float>(i,j) -= L.at<float>(0,0) * u.at<float>(i-1,j-1);
				u.at<float>(i,j) -= L.at<float>(0,1) * u.at<float>(i-1,j);
				u.at<float>(i,j) -= L.at<float>(0,2) * u.at<float>(i-1,j+1);
				u.at<float>(i,j) -= L.at<float>(1,0) * u.at<float>(i,j-1);
				u.at<float>(i,j) -= L.at<float>(1,2) * u.at<float>(i,j+1);
				u.at<float>(i,j) -= L.at<float>(2,0) * u.at<float>(i+1,j-1);
				u.at<float>(i,j) -= L.at<float>(2,1) * u.at<float>(i+1,j);
				u.at<float>(i,j) -= L.at<float>(2,2) * u.at<float>(i+1,j+1);

				u.at<float>(i,j) *= coeff;
            }
		}
	}
}

void GS(Mat A, Mat b, Mat &x)
{
    int rows = A.rows;
    int cols = A.cols;

    for(int i = 0; i < rows; ++i) {
            x.at<float>(i,0) = b.at<float>(i,0);
            for(int j = 0; j < i; ++j)
                x.at<float>(i,0) += A.at<float>(i,j)*x.at<float>(j,0);
            for(int j = i+1; j < cols; ++j)
                x.at<float>(i,0) += A.at<float>(i,j)*x.at<float>(j,0);
            x.at<float>(i,0) /= A.at<float>(i, i);
    }
}

void solveDirect(Mat &u, Mat &f)
{
    int rows = u.rows;
    int cols = u.cols;

    float h = 1.0f;///(max(rows,cols)-1);

    Mat A = laplacianMatrix((rows-2)*(cols-2), (rows-2));
    Rect internal(1,1,cols-2,rows-2);
    Mat b = h*h* (f(internal).clone().reshape(1,A.rows));
    Mat x = u(internal).clone().reshape(1,A.rows);

    cout << "Solving direct problem of size " << A.size() << endl;
    cout << "Residual before solve: " << norm(b-A*x) << endl;
    solve(A, b, x, DECOMP_SVD);
    cout << "Residual after solve: " << norm(b-A*x) << endl;

    x = x.reshape(1, rows-2);
    b = b.reshape(1, rows-2);
    x.copyTo(u(internal));
    b.copyTo(f(internal));
}

void MG(Mat &u, Mat &f, size_t ITERS_PRE, size_t ITERS_POST, int level)
{
    /*
    if(level >= MAX_LEVEL) {

        cout << "Level " << level << ": Residual norm before direct solve: " << norm(residual(u, f)) << endl;
        solveDirect(u, f);
        cout << "Level " << level << ": Residual norm after direct solve: " << norm(residual(u, f)) << endl;

        return;
    }
    */

    cout << "Level " << level << ": Residual norm before pre-smoothing: " << norm(residual(u, f)) << endl;
    smooth(u, f, ITERS_PRE);
    cout << "Level " << level << ": Residual norm after pre-smoothing: " << norm(residual(u, f)) << endl;

	int MAX_LEVEL = 3;
    if(level>=MAX_LEVEL)
        return;

    Mat res = restrictMat(residual(u, f));
    Mat e(res.rows, res.cols, CV_32F, Scalar(0.0f));

    MG(e, res, ITERS_PRE, ITERS_POST, level + 1);

    u = u + prolongMat(e);
    cout << "Level " << level << ": Residual norm after correction: " << norm(residual(u, f)) << endl;

    smooth(u, f, ITERS_POST);
    cout << "Level " << level << ": Residual norm after post-smoothing: " << norm(residual(u, f)) << endl;
}

void FMG(Mat &u, Mat f, size_t ITERS_PRE, size_t ITERS_POST)
{
    vector<Mat> fs;

    Mat resF = f.clone();
    fs.push_back(resF.clone());

	int MAX_LEVEL = 3;
    for(int i = 1; i < MAX_LEVEL; ++i) {
        resF = restrictMat(resF);
        fs.push_back(resF.clone());
        u = restrictMat(u);
    }

    solveDirect(u, resF);
     u = prolongMat(u);

    for(int i = MAX_LEVEL - 2; i >= 1; ++i) {
        cout << "Sizes: " << u.size() << " " << fs[i].size() << endl;
        MG(u, fs[i], ITERS_PRE, ITERS_POST, i);
        u = prolongMat(u);
    }
}



void MG2(Mat L, Mat &u, Mat f, size_t ITERS_PRE, size_t ITERS_POST, int level, int MAX_LEVEL)
{	
		    		// pre smoothing
	smooth(L, u, f, ITERS_PRE);

        cout << "Level " << level << ": " << "Residual after pre-smoothing: " << norm(residual(L, u, f)) << endl;

	if(level >= MAX_LEVEL) {
		return;
	}

        Mat res = restrictMat(residual(L, u, f));
        Mat e(res.rows, res.cols, CV_32F, Scalar(0.0f));
	
	// restrict L kernel
	Mat R = (Mat_<float>(3,3) << 1.0f/16.0f, 1.0f/8.0f, 1.0f/16.0f,
                                     1.0f/8.0f, 1.0f/4.0f, 1.0f/8.0f,
                                     1.0f/16.0f, 1.0f/8.0f, 1.0f/16.0f);
	Mat P = 4 * R.t();
	Mat Lh = R.mul(L).mul(P);
	cout << Lh << endl;
	//smooth(L, e, res, 100);
	//cout << e << endl;
	//cout << res << endl;


        MG2(0.25*L, e, res, ITERS_PRE, ITERS_POST, level+1, MAX_LEVEL);
	
	u = u + prolongMat(e);

	cout << "Level " << level << ": " << "Residual after correction: " << norm(residual(L, u, f)) << endl;

        // post smoothing
        smooth(L, u, f, ITERS_POST);

	cout << "Level " << level << ": " << "Residual after post-smoothing: " << norm(residual(L, u, f)) << endl;
}

Mat clean(vector<Mat> &imgs, size_t ITERS, size_t ITERS_PRE, size_t ITERS_POST, int MAX_LEVEL)
{
    size_t NUM_IMG = imgs.size();
    int rows = imgs[0].rows;
    int cols = imgs[0].cols;

    cout << "Computing gradients\n";

    vector<Mat> gradX, gradY;
    for(size_t i = 0; i < NUM_IMG; ++i) {
        Mat x,y;
        Sobel(imgs[i], x, CV_32F, 1, 0, 3);
        Sobel(imgs[i], y, CV_32F, 0, 1, 3);
        gradX.push_back(x);
        gradY.push_back(y);
    }

    cout << "Computing medians\n";

    Mat medianX(rows, cols, CV_32F);
    Mat medianY(rows, cols, CV_32F);
    Mat median(rows, cols, CV_32F);

    #pragma omp parallel for
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            vector<float> px, py, p;
            for(int k = 0; k < NUM_IMG; ++k) {
                px.push_back(gradX[k].at<float>(i,j));
                py.push_back(gradY[k].at<float>(i,j));
                p.push_back(imgs[k].at<float>(i,j));
            }

            sort(px.begin(), px.end());
            sort(py.begin(), py.end());
            sort(p.begin(), p.end());

            if(NUM_IMG%2 == 1) {
                medianX.at<float>(i,j) = px[NUM_IMG/2-1];
                medianY.at<float>(i,j) = py[NUM_IMG/2-1];
                median.at<float>(i,j) = p[NUM_IMG/2-1];
            } else {
                medianX.at<float>(i,j) = 0.5 * (px[NUM_IMG/2-1] + px[NUM_IMG/2]);
                medianY.at<float>(i,j) = 0.5 * (py[NUM_IMG/2-1] + py[NUM_IMG/2]);
                median.at<float>(i,j) = 0.5 * (p[NUM_IMG/2-1] + p[NUM_IMG/2]);
            }
        }
    }

    Mat medianXX, medianYY;
    Sobel(medianX, medianXX, CV_32F, 1 , 0, 3);
    Sobel(medianY, medianYY, CV_32F, 0 , 1, 3);

    // Mat u = median.clone();
    Mat u(rows, cols, CV_32F, Scalar(0.0f));
    Mat f = medianXX + medianYY;

    Mat L = 1.0f/3.0f * (Mat_<float>(3,3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);

    cout << "Intial residual norm: " << norm(residual(L, u, f)) << endl;

    for(size_t k = 0; k < ITERS; ++k)
        MG2(L, u, f, ITERS_PRE, ITERS_POST, 1, MAX_LEVEL);

    cout << "Final residual norm: " << norm(residual(L, u, f)) << endl;

    /*
    normalize(u, u, 0, 1, cv::NORM_MINMAX);
    Mat alfa = imax - u;
    normalize(alfa, alfa, 0, 1, cv::NORM_MINMAX);
    for(int i = 0; i < rows; ++i) {
        for(int j = 1; j < cols; ++j) {
            u.at<double>(i,j) = alfa.at<double>(i,j)*u.at<double>(i,j) + (1.0f -alfa.at<double>(i,j))*imax.at<double>(i,j);
        }
    }
    */

    return u;
}

int main()
{
    //const char* img_paths[] = {"22/1.pgm","22/2.pgm","22/3.pgm","22/4.pgm","22/5.pgm","22/6.pgm",
    //"22/7.pgm","22/8.pgm","22/9.pgm","22/10.pgm","22/11.pgm","22/12.pgm","22/13.pgm"};
    //int NUM_IMG = 13;

    int MAX_LEVEL = 10;
    int size = pow(2,MAX_LEVEL)+1;
    Mat f(size,size,CV_32F,Scalar(0.0f));
    Mat u(size,size,CV_32F,Scalar(0.0f));
    for(int i = 1; i < size-1; ++i)
        for(int j = 1; j < size-1; ++j)
            f.at<float>(i,j) = (float)rand()/RAND_MAX -0.5;
    /*
    vector<Mat> As, Rs;
    Mat A = laplacianMatrix(size*size, size);
    As.push_back(A);

    for(int i = 1; i < MAX_LEVEL; ++i) {
        int size = As[i-1].rows;
        cout << size << endl;
        Mat R = restrictionOp(size);
        Rs.push_back(R);

        Mat P = 4*R.t();

        Mat Ar = R * As[i-1] * P;
        cout << Ar.size() << endl;
        As.push_back(Ar);
    }

    x = x.reshape(1, A.rows);
    b = b.reshape(1, A.rows);
    */

    //MG2(x,b,0,0,1,As,Rs);
	
        Mat L = (Mat_<float>(3,3) << 1.0f/4.0f, 1.0f/2.0f, 1.0f/4.0f, 1.0f/2.0f, -3.0f, 1.0f/2.0f, 1.0f/4.0f, 1.0f/2.0f, 1.0f/4.0f);
        //L = (Mat_<float>(3,3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
        L = 1.0f/3.0f * (Mat_<float>(3,3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
        //L = 1.0f/6.0f * (Mat_<float>(3,3) << 1, 4, 1, 4, -20, 4, 1, 4, 1);

    cout << "Initial Residual: " << norm(residual(L, u, f)) << endl;

    //MG(x,b,3,3,1);
    //MG2(L, u, f, 3, 3, 1, MAX_LEVEL);
    smooth(L, u, f, 6);

    cout << "Final Residual: " << norm(residual(L, u, f)) << endl;

    const char* img_paths[] = {"test/1.jpg","test/2.jpg","test/3.jpg","test/4.jpg"};
    int NUM_IMG = 4;
    
    cout << "Loading images\n";

    Mat tmp = imread(img_paths[0]);
    int targetDim = tmp.rows;
    int po2 = 2;
    MAX_LEVEL = 1;
    while(po2 < targetDim) {
        po2 = pow(2, MAX_LEVEL);
        MAX_LEVEL++;
    }

    vector<Mat> imgs;
    for(int i = 0; i < NUM_IMG; ++i) {
        Mat img = imread(img_paths[i]);
        img.convertTo(img, CV_32FC3);
        //normalize(img, img, 0, 1, NORM_MINMAX);
        img = resizeKeepAspectRatio(img, Size(pow(2,MAX_LEVEL)+1,pow(2,MAX_LEVEL)+1), Scalar(0.0f,0.0f,0.0f));
        imgs.push_back(img);
    }

    cout << "Splitting channels\n";

    vector<Mat> imgsR, imgsG, imgsB;
    for(int i = 0; i < NUM_IMG; ++i) {
        Mat tmp[3];
        split(imgs[i], tmp);
        imgsB.push_back(tmp[0]);
        imgsG.push_back(tmp[1]);
        imgsR.push_back(tmp[2]);
    }

    cout << "Cleaning up\n";

    vector<Mat> BGR(3);
    BGR[0] = clean(imgsB, 1, 3, 3,MAX_LEVEL);
    BGR[1] = clean(imgsG, 1, 3, 3, MAX_LEVEL);
    BGR[2] = clean(imgsR, 1, 3, 3, MAX_LEVEL);

    cout << "Saving output\n";

    Mat output;
    merge(BGR, output);

    normalize(output, output, 0, 1, NORM_MINMAX);
    output.convertTo(output, CV_8UC3, 255);

    imwrite("cleaned.png", output);
    
    return 0;
}
