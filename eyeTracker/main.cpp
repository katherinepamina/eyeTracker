//
//  main.cpp
//  eyeTracker
//
//  Created by Pamina Lin on 3/9/16.
//  Copyright © 2016 Pamina Lin. All rights reserved.
//


#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"

//#include "detectFaces.hpp"

using namespace cv;
using namespace std;

// Constants
const float  kSmoothFaceFactor = 0.005;
const float  kEyeTopFraction = .25;
const float  kEyeSideFraction = .10;
const float  kEyeHeightFraction = .30;
const float  kEyeWidthFraction = .35;
const double kGradientThreshold = 50.0;
const int    kWeightBlurSize = 5;
const float  kWeightDivisor = 1.0;
const int    kScaledDownEyeWidth = 50;

// Global variables
string face_cascade_path = "/Users/paminalin/Developer/eyeTracker/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade; // CascadeClassifier class detects objects in a video stream (load Haar or LBP classifiers or detectMultiScale to perform detection)


// Helper functions (move to other file later)
vector<Rect_ <int> > detectFaces(Mat frame, string cascade_path) {
    vector<Rect_<int> > faces = vector<Rect_<int>>();
    
    // Load the cascade
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(cascade_path)) {
        cout << "Error loading face cascade." << endl;
        return faces;
    }
    
    /*Mat original_copy = frame.clone();  <-- now converting to gray before this function
    Mat gray_copy;
    cvtColor(original_copy, gray_copy, CV_BGR2GRAY);
     */
    face_cascade.detectMultiScale(frame, faces);
    return faces;
    
}


// Make a draw all things function later
void drawFace(Mat &frame, Rect_ <int> face) {
    rectangle(frame, face, CV_RGB(0,255,0), 1);
    return;
}

Mat computeXGradient(Mat mat) {
    Mat gradient(mat.rows, mat.cols, CV_64F);

    for (int j=0; j<mat.rows; j++) {
        const uchar *matRowPtr = mat.ptr<uchar>(j);
        double *gradientRowPtr = gradient.ptr<double>(j);
        
        // Right now, calculating the gradient by taking the difference on x or y side
        
        // Process the first column separately
        gradientRowPtr[0] = matRowPtr[1] - matRowPtr[0];
        
        // Process middle columns
        for (int i=1; i<mat.cols-1; i++) {
            gradientRowPtr[i] = (matRowPtr[i+1] - matRowPtr[i-1])/2.0;
        }
        
        // Also process the last column separately
        gradientRowPtr[mat.cols-1] = matRowPtr[mat.cols-1] - matRowPtr[mat.cols-2];
    }
    
    return gradient;
}

Mat computeYGradient(Mat mat) {
    // Compute the y gradient by taking the gradient of the transpose and then transposing it again
    return computeXGradient(mat.t()).t();
}

Mat computeMagnitudes(Mat mat1, Mat mat2) {
    // check that mat1 and mat2 are the same dimension? will be necessary if using cascade
    Mat mag(mat1.rows, mat1.cols, CV_64F);
    for (int j=0; j<mat1.rows; j++) {
        const double *xPtr = mat1.ptr<double>(j);
        const double *yPtr = mat2.ptr<double>(j);
        double *magPtr = mag.ptr<double>(j);
        for (int i=0; i<mat1.cols; i++) {
            magPtr[i] = sqrt((xPtr[i]*xPtr[i]) + yPtr[i]*yPtr[i]);
        }
    }
    return mag;
}

double computeGradientThreshold(Mat gradient) {
    Scalar stdDev;
    Scalar mean;
    meanStdDev(gradient, mean, stdDev);
    //cout << "mean: " << mean[0] << endl;
    //cout << "stdDev: " << stdDev[0] << endl;
    double stdDevScaled = stdDev[0] / sqrt(gradient.rows*gradient.cols);
    return mean[0] + stdDevScaled*50; // trying 50 for now based on recommendation from article
}

void normalizeMats(Mat &mat1, Mat &mat2) {
    Mat magnitudes = computeMagnitudes(mat1, mat2);
    
    // Get some sort of threshold so if the gradient is under the threshold, just set it to zero
    double gradThreshold = computeGradientThreshold(magnitudes);
    //cout << "threshold: " << gradThreshold << endl;
    for (int j=0; j<mat1.rows; j++) {
        double * mat1Ptr = mat1.ptr<double>(j);
        double * mat2Ptr = mat2.ptr<double>(j);
        double * magPtr = magnitudes.ptr<double>(j);
        
        for (int i=0; i<mat1.cols; i++) {
            double mat1Element = mat1Ptr[i];
            double mat2Element = mat2Ptr[i];
            double mag = magPtr[i];
            if (mag > gradThreshold) {
                mat1Ptr[i] = mat1Element / mag;
                mat2Ptr[i] = mat2Element / mag;
            } else {
                mat1Ptr[i] = 0.0;
                mat2Ptr[i] = 0.0;
            }
        }
    }
    return;
}

void normalizeVector(double &x, double &y) {
    double magnitude = sqrt(x*x + y*y);
    x = x / magnitude;
    y = y / magnitude;
    return;
}


Mat getWeightedImage(Mat image) {
    Mat weight;
    GaussianBlur(image, weight, Size(kWeightBlurSize, kWeightBlurSize), 0, 0);
    
    //blur(image, weight, Size(kWeightBlurSize, kWeightBlurSize));
    for (int j=0; j<weight.rows; j++) {
        unsigned char * rowPtr = weight.ptr<unsigned char>(j); // MAYBE THE PTR DATA TYPE IS NOT RIGHT?
        for (int i=0; i<weight.cols; i++) {
            rowPtr[i] = (255 - rowPtr[i]);
        }
    }
    //imshow("weighted", weight);
    return weight;
}

void scaleDownImage(Mat &src, Mat &dst) {
    float ratio = (float)src.rows/src.cols;
    Size smallerSize = Size(kScaledDownEyeWidth, ((float)kScaledDownEyeWidth)*ratio);
    resize(src, dst, smallerSize);
}

Point unscalePoint(Point p, Rect size) {
    float ratio = (((float)kScaledDownEyeWidth)/size.width);
    int x = round(p.x / ratio);
    int y = round(p.y / ratio);
    return Point(x, y);
}

// where x and y is a Mat element, gradX and gradY is gradient vector at that element
void evaluateAllCenters(int x, int y, const Mat &weight, double gradX, double gradY, Mat &result) {
    for (int j=0; j < result.rows; j++) {
        double * resultPtr = result.ptr<double>(j);
        const unsigned char * weightPtr = weight.ptr<unsigned char>(j);
        for (int i=0; i<result.cols; i++) {
            //cout << "(" << i << ", " << j << ") " << endl;
            // if the current location being tested is the same as the one we're evaluating at, continue
            if (x == i && y == j) {
                continue;
            }
            // otherwise, calculate the direction vectors
            double directionX = x - i;
            double directionY = y - j;
            //cout << "before norm: (" << directionX << ", " << directionY << ") " << endl;
            normalizeVector(directionX, directionY);
            //cout << "after norm: (" << directionX << ", " << directionY << ") " << endl;

            
            // Get the dot product
            //cout << "gradient: (" << gradX << ", " << gradY << ") " << endl;
            double dotProduct = directionX*gradX + directionY*gradY;
            dotProduct = (dotProduct > 0) ? dotProduct : -1 * dotProduct;
            //cout << "dot product: " << dotProduct << endl;
            //double dotProductMag = (dotProduct < 0) ? -1*dotProduct : dotProduct; // <-- shouldn't ned this anymore
            //cout << "dot product mag: " << dotProductMag << endl;
            // Add the weighting
            resultPtr[i] += dotProduct*dotProduct*weightPtr[i]/kWeightDivisor;
            /*if (resultPtr[i] + dotProductMag > 0) {
                //cout << "result before: " << resultPtr[i] << endl;
                resultPtr[i] = dotProductMag + resultPtr[i]; // += not working...?
                //cout << "result: " << resultPtr[i] << endl;
            }
            if (resultPtr[i] >= std::numeric_limits<float>::infinity() || resultPtr[i] < 0) {
                cout << "INF at (" << i << ", " << j << ")" << endl;
            }*/
            //resultPtr[i] += dotProduct*dotProduct;
            //cout << "result i " << resultPtr[i] << endl;
            //cout << endl << endl;
        }
    }
}

Mat getSubImage(Mat image, Rect imageRect, Rect roi) {
    if (imageRect.width > 0 && imageRect.height > 0) {
        int X = roi.x - imageRect.x;
        int Y = roi.y - imageRect.y;
        Rect translatedROI(X, Y, roi.width, roi.height);
        Mat subimage = image(translatedROI);
        if (!subimage.empty()) {
            //imshow("subimage", subimage);
            return subimage;
        } else {
            return image;
        }
    } else {
        return image; // TODO: is this really desired behavior
    }
}


Point findEyeCenter(Mat eyeImageUnscaled, Rect eyeROI, String window) {
    Mat eyeImage;
    //cout << "eyeImage unscaled size (rows, cols) " << eyeImageUnscaled.rows << ", " << eyeImageUnscaled.cols << ") " << endl;
    //imshow("unscaled", eyeImageUnscaled);
    scaleDownImage(eyeImageUnscaled, eyeImage);
    //cout << "eyeImage size (rows, cols) " << eyeImage.rows << ", " << eyeImage.cols << ") " << endl;
    //imshow("scaled", eyeImage);
    // Get the gradients of the eye image
    Mat gradX = computeXGradient(eyeImage);
    Mat gradY = computeYGradient(eyeImage);
    //imshow("gradx", gradX);
    //imshow("grady", gradY);
    
    normalizeMats(gradX, gradY);
    //imshow("gradx", gradX);
    //imshow("grady", gradY);
    
    // Get a "weight" Mat, equal to the inverse gray-scale image
    Mat weight = getWeightedImage(eyeImage);
    //imshow("weight", weight);
    
    // Set up the result Mat
    Mat result = Mat::zeros(eyeImage.rows, eyeImage.cols, CV_64F);
    
    // For each gradient location, evaluate every possible eye center
    for (int j=0; j<eyeImage.rows; j++) {
        const double * gradXPtr = gradX.ptr<double>(j);
        const double * gradYPtr = gradY.ptr<double>(j);
        for (int i=0; i<eyeImage.cols; i++) {
            double gradX = gradXPtr[i];
            double gradY = gradYPtr[i];
            // if the gradient is 0, ignore the point
            if (gradX == 0.0 && gradY == 0.0) {
                continue;
            }
            // otherwise, test all possible centers against this location/gradient
            evaluateAllCenters(i, j, weight, gradX, gradY, result);
        }
    }
    
    // Look for the maximum dot product (should correspond with the center of the circle)
    //double numGradients = (eyeImage.rows > 0) ? eyeImage.rows*eyeImage.cols : 0.00001;
    double numGradients = eyeImage.rows * eyeImage.cols;
    //cout << "numGradients: " << numGradients << endl;
    Mat resultScaled;
    result.convertTo(resultScaled, CV_32F, 1.0/numGradients);
    //resultScaled = result;
    Point maxCenter;
    double maxDotProduct = 0;
    //minMaxLoc(resultScaled, NULL, &maxDotProduct, NULL, &maxCenter);
    
    double currentMax = 0;
    Point currentMaxPoint;
    for (int j=0; j<resultScaled.rows; j++) {
        const float * resultPtr = resultScaled.ptr<float>(j);
        for (int i=0; i<resultScaled.cols; i++) {
            if (resultPtr[i] > currentMax) {
                //cout << "result i: " << resultPtr[i] << endl;
                //cout << "prev max: " << currentMax << endl;
                currentMax = resultPtr[i];
                //cout << "current max: " << currentMax << endl;
                currentMaxPoint.x = i;
                currentMaxPoint.y = j;
            }
        }
    }
    maxCenter = currentMaxPoint;
    maxDotProduct = currentMax;

    //cout << "max dot product " << maxDotProduct << endl;
    //cout << "max center" << maxCenter.x << ", " << maxCenter.y << endl;
    
    //cout << "best center: (" << maxCenter.x << ", " << maxCenter.y << ")" << endl;
    //cout << "eyeRegion: (" << eyeRegion.x << ", "<< eyeRegion.y << ")" << endl;
    //cout << "faceRegion: (" << faceRegion.x << ", " << faceRegion.y << ")" << endl;
    
    // Need to translate eye point back to original coordinate system
    //maxCenter.x += faceRegion.x;
    //maxCenter.y += faceRegion.y;
    Point resultCenter = unscalePoint(maxCenter, eyeROI);
    circle(eyeImage, maxCenter, 3, CV_RGB(0,0,255), -1);
    imshow("eye", eyeImage);
    return resultCenter;
}



void detectEyes(Mat &frame, Rect faceRect) {
    //cout << "detect eyes" << endl;
    Mat face_image = frame(faceRect);
    
    // TODO: replace and use haar cascades
    // find eye regions and draw them
    int width = faceRect.width;
    int height = faceRect.height;
    int eyeRegionTop = height * kEyeTopFraction;
    int eyeRegionSide = width * kEyeSideFraction;
    
    
    int eyeRegionWidth = width * kEyeWidthFraction;
    int eyeRegionHeight = width * kEyeHeightFraction;
    int leftEyeX = faceRect.x + eyeRegionSide;
    int rightEyeX = faceRect.x + 2*eyeRegionSide + eyeRegionWidth;
    int eyeY =  faceRect.y + eyeRegionTop;
    
    Rect leftEyeRegion(leftEyeX, eyeY, eyeRegionWidth, eyeRegionHeight);
    Rect rightEyeRegion(rightEyeX, eyeY, eyeRegionWidth, eyeRegionHeight);
    
    // Get subimages
    Mat leftEyeImage = getSubImage(face_image, faceRect, leftEyeRegion);
    Mat rightEyeImage = getSubImage(face_image, faceRect, rightEyeRegion);

    
    // find eye center
    Point leftPupil = findEyeCenter(leftEyeImage, leftEyeRegion, "left eye");
    Point rightPupil = findEyeCenter(rightEyeImage, rightEyeRegion, "right eye");


    // draw
    rectangle(frame, leftEyeRegion, CV_RGB(0,0,255), 1);
    rectangle(frame, rightEyeRegion, CV_RGB(0,0,255), 1);
    circle(leftEyeImage, leftPupil, 3, CV_RGB(0,0,255), -1);
    circle(rightEyeImage, rightPupil, 3, CV_RGB(0,0,255), -1);
    return;
}

Rect getBiggestFace(vector<Rect_ <int>> faces) {
    int maxSize = 0;
    Rect maxFace;
    for (int i=0; i<faces.size(); i++) {
        Rect face_i = faces.at(i);
        int currentSize = face_i.width + face_i.height;
        if (currentSize > maxSize) {
            maxSize = currentSize;
            maxFace = face_i;
        }
    }
    return maxFace;
}

int main() {
    // Load the cascade
    if (!face_cascade.load(face_cascade_path)) {
        cout << "Error loading face cascade." << endl;
        return -1;
    }
    
    //namedWindow(main_window_name, CV_WINDOW_NORMAL);
    
    VideoCapture cap(-1); // switch back to 0?  which is which
    if (!cap.isOpened()) {
        cout << "Webcame is not open." << endl;
        return -1;
    }
    
    Mat frame;

    while (true) {
        cap.read(frame);
        /*
        if (!frame.empty()) {
            imshow("frame", frame);
        }
        else {
            cout << "No capture frame" << endl;
            break;
        }
         */
        
        //Mat weight;
        
        if (frame.rows > 0 && frame.cols > 0) {
            cvtColor(frame, frame, CV_BGR2GRAY);
        
            vector<Rect_ <int>> faces = detectFaces(frame, face_cascade_path);
            // Get the biggest face
            Rect biggestFace = getBiggestFace(faces);
            if (biggestFace.width > 0 && biggestFace.height > 0) {
                //weight = getWeightedImage(frame);
                //imshow("weighted", weight);
                drawFace(frame, biggestFace);
                detectEyes(frame, biggestFace);
            }
            
            // Smooth image
            // Detect eyes
            // Calculate gradients
            // Calculate direction vectors
            // Calculate dot products
            // Calculate weights/Users/paminalin/Desktop/Screen Shot 2016-03-14 at 3.13.59 PM.png
            // Determine best center
        
            imshow("face_detector", frame);
        
            waitKey(1);
        }
        
        
        
        
        
        
        // from TUTORIAL make matrices
        /*Mat image;
         Mat HSVimage; // hue saturation value
         Mat processedImage;
         
         cap.read(image); // assigns mat image to raw webcam footage
         // convert from rgb to hsv (easier to decipher some things)
         cvtColor(image, HSVimage, CV_BGR2HSV); // convert mat image to hsv under mat HSVimage
         inRange(HSVimage, Scalar(0,0,0), Scalar(100,100,100), processedImage); // processes HSVimage and gets only pixels in scalar range to processedImage
         
         int numPixels = countNonZero(processedImage);
         
         cout << numPixels << endl;
         
         imshow("Original", image);
         imshow("Processed", processedImage);
         */
    }
}
