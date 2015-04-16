#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include "Eigen/Dense"

using namespace cv;
using namespace std;
using namespace std;
using namespace Eigen;

///Function for sorting bounded rectangles
bool compareX_rect(const Rect & a, const Rect &b) {
    return a.x <= b.x;
}
bool compareY_rect(const Rect & c, const Rect &d) {
    return c.y <= d.y;
}

class ColorCalibrator{
    vector<Vec3i> dedColors;
    Mat original;
    Mat correctImg;
    vector<Scalar> origColors;
    vector<Rect> rectangles;
    Matrix3f ccm;
    string filenameCCM;
    string filenameWhiteBalance;
    string path;
    Rect whitePatch;
    MatrixXf whiteBalance;
    
    
    
public:
    ColorCalibrator(Mat originalInput,string filenameCCMInput, string filenameWhiteBalanceInput, string path);
    void readDedicatedColors();
    void getPatches();
    void calculateCCM();
    void saveCCM();
    void visualizeResults();
    void performWhiteBalance();
    void saveWhiteBalance();
    void saveRectangles();
};

///Constructor Initialization
ColorCalibrator::ColorCalibrator(Mat originalInput,string filenameCCMInput, string filenameWhiteBalanceInput, string pathInput){
    original = originalInput.clone();
    filenameCCM = filenameCCMInput;
    filenameWhiteBalance = filenameWhiteBalanceInput;
    path = pathInput;
}

///Read the dedicated colours from the file
void ColorCalibrator::readDedicatedColors(){
    ifstream myfile;
    string line;
    string delimiter = ",";
    string temp;
    vector<int> colors;
    
    myfile.open("DedicatedColors.txt");
    if (myfile.is_open())
    {
        while (getline (myfile,line))
        {
            while(line.find(",",0) != string::npos)
            {
                size_t pos = line.find(",",0);
                temp = line.substr(0, pos);
                line.erase(0,pos+1);
                colors.push_back(atoi(temp.c_str()));
            }
            if(!line.empty()){
                colors.push_back(atoi(line.c_str()));
                dedColors.push_back(Vec3i(colors[2],colors[1],colors[0])); //cout << colors[0] << "\t" << colors[1] << "\t" << colors[2] << "\n";   //BGR order
                colors.clear();
            }

        }
        myfile.close();
    }
    
}

///getPatches(): This method automatically calculates the (shrinked) patches from the original image and saves their mean-values in origColors, as well as saves the corresponding rectangles with respect to the original image.

void ColorCalibrator::getPatches(){
    
    Mat src, src_gray;
    //int thresh = 13;
    //int max_thresh = 255;
    RNG rng(12345);
    Matrix3f CCM;
    vector<vector<Point> > contours;
    
    /// Load source image and convert it to gray
    src = original.clone();
    
    if (src.empty()) {
        cout << "Unable to read image";
        waitKey(0);
        exit(0);
    }
    
    /// Convert image to gray and blur it
    cvtColor( src, src_gray, CV_BGR2GRAY );
    //blur( src_gray, src_gray, Size(11,11) );
    GaussianBlur(src_gray, src_gray, Size(7,7), 1.5, 1.5);
    Mat threshold_output;
    
    vector<Vec4i> hierarchy;
    
    double maxValue = 255;
    int adaptiveMethod = ADAPTIVE_THRESH_MEAN_C;
    int thresholdType = CV_THRESH_BINARY;
    int blockSize = 23; //23
    double C = 0;
    Mat binary;
    
    adaptiveThreshold(src_gray, binary, maxValue, adaptiveMethod, thresholdType, blockSize, C);
    
    threshold_output = binary.clone();
    
    ///Perform dilation and erosion
    int dilation_type = MORPH_ELLIPSE;
    int dilation_size = 3;
    
    int erosion_type = MORPH_ELLIPSE;
    int erosion_size = 3;
    
    Mat dilation_elem = getStructuringElement(dilation_type, Size(2* dilation_size +1, 2*dilation_size+1));
    Mat erosion_elem = getStructuringElement(erosion_type, Size(2*erosion_size+1, 2*erosion_size+1));
    
    dilate(threshold_output, threshold_output, dilation_elem);
    erode(threshold_output, threshold_output, erosion_elem);
    
    namedWindow( "rectangle", CV_WINDOW_AUTOSIZE );
    imshow( "rectangle", threshold_output );
    
    /// Find contours
    findContours( threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    ///Get bounding rectangles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect;
    long int areaImg = src.rows / 8 * src.cols / 5;
//    long int areaImg = src.rows / 10 * src.cols / 7;
    long int areaThresh = areaImg;
    
    for( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        Rect actRect = boundingRect( Mat(contours_poly[i]) );
        if (actRect.size().area() > areaThresh)
            boundRect.push_back(actRect);
        
    }
    
    ///Sort the bounded rectangles
    
    //stable_sort( boundRect.begin(), boundRect.end(), compareX_rect );
    stable_sort( boundRect.begin(), boundRect.end(), compareY_rect );
    
    // for(int i=0; i<18; i=i+6 )
    stable_sort( boundRect.begin(), boundRect.begin()+6, compareX_rect );
    stable_sort( boundRect.begin()+6, boundRect.begin()+12, compareX_rect );
    stable_sort( boundRect.begin()+12, boundRect.begin()+18, compareX_rect );
    stable_sort( boundRect.begin()+18, boundRect.end(), compareX_rect );
//    stable_sort( boundRect.begin()+18, boundRect.begin()+24, compareX_rect );
//    stable_sort( boundRect.begin()+24, boundRect.end(), compareX_rect );
   
    
    //make a copy of bounded rectangles for later use
    vector<Rect> boundRectComplete( boundRect.size() );
    boundRectComplete = boundRect;

    
    ///Shrink Rectangles
    for (int i=0; i<boundRect.size(); i++)
    {
        boundRect[i].x += 0.5 * boundRect[i].width/2;
        boundRect[i].y += 0.5 * boundRect[i].height/2;
        boundRect[i].width *= 0.5;
        boundRect[i].height *= 0.5;
    }
    
    whitePatch = boundRect[18];
//    whitePatch = boundRect[8];
    
    //Restricting the CCM calculation to only use few blocks
    //boundRect.resize(18);
    
    //save the rectangles
    rectangles = boundRect;
    
    
    /// Draw contours
    Mat contoursImg = src.clone();
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
    for( int i = 0; i< boundRect.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        //drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, Point() );
        rectangle( contoursImg, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
    }
    
    /// Show in a window
    namedWindow( "Bounding rectangle", CV_WINDOW_AUTOSIZE );
    imshow( "Bounding rectangle", contoursImg );
    
    ///Crop and save the images of each rectangles. Calculate the mean for each block
    ///The value for blue, green and red are stored in blocksMean[i].cv::Matx<double, 4, 1>::operator()(0), blocksMean[i].cv::Matx<double, 4, 1>::operator()(1) and blocksMean[i].cv::Matx<double, 4, 1>::operator()(2) respectively; i is the block number.
    
    String filename;
    Mat srcCopy = original.clone();

    vector<Scalar> blocksMean(boundRect.size());
    for (int i=0; i<boundRect.size(); i++)
    {
        Mat croppedImg;
        srcCopy(boundRect[i]).copyTo(croppedImg);
        filename = to_string(i+1) + ".jpg";
        imwrite(filename, croppedImg);
        blocksMean[i] = mean(croppedImg);
    }
    origColors = blocksMean; //Mean values saved

}

/// calculateCCM(): calculates the color correction matrix

void ColorCalibrator::calculateCCM(){
    
    ///Compute color correction Matrix
    ///Prepare values in the form Ax=b
    MatrixXf A(origColors.size(),3);
    for (int i=0; i<origColors.size(); i++)
    {
        A(i,0) = origColors[i].cv::Matx<double, 4, 1>::operator()(0);  //BGR order
        A(i,1) = origColors[i].cv::Matx<double, 4, 1>::operator()(1);
        A(i,2) = origColors[i].cv::Matx<double, 4, 1>::operator()(2);
    }
    //cout << "Here is the matrix A:\n" << A << endl;
  
    Matrix3f CCMT;
    
    ///Color values for Blue component from the color chart
    VectorXf bBlue(origColors.size());
    for(int i=0; i<origColors.size(); i++){
        bBlue[i] = dedColors[i][0];
    }
    bBlue.resize(origColors.size());
    CCMT.col(0) << A.jacobiSvd(ComputeThinU | ComputeThinV).solve(bBlue);
    
    ///Color values for Green component from the color chart
    VectorXf bGreen(origColors.size());
    for(int i=0; i<origColors.size(); i++){
        bGreen[i] = dedColors[i][1];
    }
    bGreen.resize(origColors.size());
    CCMT.col(1) << A.jacobiSvd(ComputeThinU | ComputeThinV).solve(bGreen);
    
    //Color values for Red component from the color chart
    VectorXf bRed(origColors.size());
    for(int i=0; i<origColors.size(); i++){
        bRed[i] = dedColors[i][2];
    }
    bRed.resize(origColors.size());
    CCMT.col(2) << A.jacobiSvd(ComputeThinU | ComputeThinV).solve(bRed);
    
    cout << "The color correction matrix is\n";
    
    ccm = CCMT.transpose(); //RGB Order
    
    cout << ccm;

}

///saveCCM(): Saves the CCM to a txt file at location path, with name filename.
void ColorCalibrator:: saveCCM(){
    ///Write the ccm values to a file
    ofstream ccmFile;
    string file;
    file = path + filenameCCM;
    ccmFile.open (file);
 //   ccmFile << "Color Correction Matrix\n";
 //   ccmFile << "------------------------------------------------------------------------\n";
    for (int i=0; i<3; i++){
        for(int j=0; j<3; j++)
            ccmFile << ccm(i,j) << ",";
        ccmFile << "\n";
    }
    ccmFile.close();

}

/// visualizeResults():Creates the two overlay images and saves them to .jpg at location path, use filenames: “orig_overlayed_result.jpg”, “result_overlayed_dedicated.jpg”
void ColorCalibrator::visualizeResults(){
    Mat testImgUC = original.clone();
    Mat originalImg = original.clone();
    Mat testImg;
    testImgUC.convertTo(testImg, CV_32FC3, 1/255.0);
    
    for( int x = 0; x < testImg.rows; x++ )
    {
        for( int y = 0; y < testImg.cols; y++ )
        {
            testImg.at<Vec3f>(x,y)[0] = testImg.at<Vec3f>(x,y)[0] * ccm(0,0) +
                                        testImg.at<Vec3f>(x,y)[1] * ccm(0,1) +
                                        testImg.at<Vec3f>(x,y)[2] * ccm(0,2);
            
            testImg.at<Vec3f>(x,y)[1] = testImg.at<Vec3f>(x,y)[0] * ccm(1,0) +
                                        testImg.at<Vec3f>(x,y)[1] * ccm(1,1) +
                                        testImg.at<Vec3f>(x,y)[2] * ccm(1,2);
            
            testImg.at<Vec3f>(x,y)[2] = testImg.at<Vec3f>(x,y)[0] * ccm(2,0) +
                                        testImg.at<Vec3f>(x,y)[1] * ccm(2,1) +
                                        testImg.at<Vec3f>(x,y)[2] * ccm(2,2);
        }
    }
    
    testImg.convertTo(testImgUC, CV_8UC3, 255);
    correctImg = testImgUC.clone(); //Save the corrected Image

    
    Mat dedicatedImg = testImgUC.clone();
    for( int i = 0; i < rectangles.size(); i++ )
    {
        Scalar color = Scalar(dedColors[i][0], dedColors[i][1], dedColors[i][2]);
        Rect rect = Rect(rectangles[i].tl().x, rectangles[i].tl().y, rectangles[i].width, rectangles[i].height);
        rectangle(dedicatedImg , rect, color, CV_FILLED,0,0);
        rectangle(dedicatedImg, rect, Scalar(0,0,0), 2);
        
    }
    
    namedWindow( "Dedicated and Corrected overlay", CV_WINDOW_AUTOSIZE );
    imshow( "Dedicated and Corrected overlay", dedicatedImg );
    imwrite("Dedicated and Corrected overlay.jpg", dedicatedImg);
    
    Mat mask = Mat::zeros(testImgUC.size(), CV_8UC1);
    
    for( int i = 0; i< rectangles.size(); i++ )
    {
        Rect rect = Rect(rectangles[i].tl().x, rectangles[i].tl().y, rectangles[i].width, rectangles[i].height);
        rectangle(mask,rect, Scalar(255), CV_FILLED);
        
    }
    
    testImgUC.copyTo(originalImg, mask);
    
    for( int i = 0; i< rectangles.size(); i++ )
    {
        Rect rect = Rect(rectangles[i].tl().x, rectangles[i].tl().y, rectangles[i].width, rectangles[i].height);
        //Point center = Point((rectangles[i].x + rectangles[i].width)/2, (rectangles[i].y + rectangles[i].height)/2);
        //Rect rect = Rect(center.x, center.y, 75, 75);
        rectangle(originalImg, rect, Scalar(0,0,0), 2);
    }
    
    namedWindow( "Corrected and Orignal overlay", CV_WINDOW_AUTOSIZE );
    imshow( "Corrected and Orignal overlay", originalImg );
    imwrite("Corrected and Orignal overlay.jpg", originalImg);
}

///Peform white balancing on the corrected Image
void ColorCalibrator::performWhiteBalance(){
    
    Mat srcCopy = original.clone();
    Scalar whitePatchMean;
    Mat croppedImg;
    srcCopy(whitePatch).copyTo(croppedImg);
    whitePatchMean = mean(croppedImg);
   
    ///convert the matrix to the form aX = b, where X represents the diagonal matrix for white balancing
    VectorXf xWB;
    
    ///Get the dedicated values of the white patch
    VectorXf bDedicated(3);
    bDedicated[0] = dedColors[18][0];
    bDedicated[1] = dedColors[18][1];
    bDedicated[2] = dedColors[18][2];
    
    VectorXf bCalculated(3);
    bCalculated[0] = whitePatchMean[0];
    bCalculated[1] = whitePatchMean[1];
    bCalculated[2] = whitePatchMean[2];
    
    xWB = bDedicated.cwiseQuotient(bCalculated);
    
    whiteBalance = xWB.asDiagonal();
    cout << "\nMatrix for white balance\n" << whiteBalance;
    
    ///Visualize the result of white balance
    Mat tmpImgUC = correctImg.clone();
    Mat tmpImg;
    tmpImgUC.convertTo(tmpImg, CV_32FC3, 1/255.0);
    
    for( int x = 0; x < tmpImg.rows; x++ )
    {
        for( int y = 0; y < tmpImg.cols; y++ )
        {
            tmpImg.at<Vec3f>(x,y)[0] =  tmpImg.at<Vec3f>(x,y)[0] * whiteBalance(0,0) +
                                        tmpImg.at<Vec3f>(x,y)[1] * whiteBalance(0,1) +
                                        tmpImg.at<Vec3f>(x,y)[2] * whiteBalance(0,2);
            
            tmpImg.at<Vec3f>(x,y)[1] =  tmpImg.at<Vec3f>(x,y)[0] * whiteBalance(1,0) +
                                        tmpImg.at<Vec3f>(x,y)[1] * whiteBalance(1,1) +
                                        tmpImg.at<Vec3f>(x,y)[2] * whiteBalance(1,2);
            
            tmpImg.at<Vec3f>(x,y)[2] =  tmpImg.at<Vec3f>(x,y)[0] * whiteBalance(2,0) +
                                        tmpImg.at<Vec3f>(x,y)[1] * whiteBalance(2,1) +
                                        tmpImg.at<Vec3f>(x,y)[2] * whiteBalance(2,2);
        }
    }
    
    
    tmpImg.convertTo(tmpImgUC, CV_8UC3, 255);
    
    /*
    Mat mask = Mat::zeros(tmpImgUC.size(), CV_8UC1);
    
    for( int i = 0; i< rectangles.size(); i++ )
    {
        Rect rect = Rect(rectangles[i].tl().x, rectangles[i].tl().y, 75, 75);
        //Point center = Point((rectangles[i].x + rectangles[i].width)/2, (rectangles[i].y + rectangles[i].height)/2);
        //Rect rect = Rect(center.x, center.y, 75, 75);
        rectangle(mask,rect, Scalar(255), CV_FILLED);
        
    }
    Mat originaltmp = correctImg.clone();
    tmpImgUC.copyTo(originaltmp, mask);
    
    for( int i = 0; i< rectangles.size(); i++ )
    {
        Rect rect = Rect(rectangles[i].tl().x, rectangles[i].tl().y, 75, 75);
        //Point center = Point((rectangles[i].x + rectangles[i].width)/2, (rectangles[i].y + rectangles[i].height)/2);
        //Rect rect = Rect(center.x, center.y, 75, 75);
        rectangle(originaltmp, rect, Scalar(0,0,0), 2);
    }
     */
     
    namedWindow( "White balance", CV_WINDOW_AUTOSIZE );
    imshow( "White balance", tmpImgUC );
    
}

///saveWhiteBalance(): Save the white balance matrix to a text file at location path, with name filename.
void ColorCalibrator:: saveWhiteBalance(){
    ///Write the white balance matrix values to a file
    ofstream wbFile;
    string file;
    file = path + filenameWhiteBalance;
    wbFile.open (file);
//   wbFile << "White Balance Matrix\n";
//    wbFile << "------------------------------------------------------------------------\n";
    for (int i=0; i<3; i++){
        for(int j=0; j<3; j++)
            if(i == j)
                wbFile << whiteBalance(i,j) << ",";
        wbFile << "\n";
    }
    wbFile.close();
    
}

///Save the Rectangles to a file
void ColorCalibrator:: saveRectangles(){
    ///Write the rectangle values to a file
    ofstream rectFile;
    string file;
    file = "Rectangles.txt";
    rectFile.open (file);
    for (int i=0; i<rectangles.size(); i++){
        rectFile << rectangles[i].x << "," << rectangles[i].y << "," << rectangles[i].width << "," << rectangles[i].height <<"\n";
    }
    rectFile.close();
}

int main(){
    
//    Mat original = imread("colorchartnew.png", 1 );
    Mat original = imread("colorchartRealSmallBlue.jpg", 1 );
    string path = "";
    string filenameCCM = "color correction matrix.txt";
    string filenameWB = "White Balance Matrix.txt";
    ColorCalibrator cc(original,filenameCCM,filenameWB,path);
    cc.readDedicatedColors();
    cc.getPatches();
    cc.calculateCCM();
    cc.saveCCM();
    cc.visualizeResults();
    cc.performWhiteBalance();
    cc.saveWhiteBalance();
    cc.saveRectangles();
    waitKey(0);
    return(0);

}