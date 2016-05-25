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
string eye_cascade_path = "/Users/paminalin/Developer/eyeTracker/haarcascade_eye.xml";
cv::CascadeClassifier face_cascade; // CascadeClassifier class detects objects in a video stream (load Haar or LBP classifiers or detectMultiScale to perform detection)
cv::CascadeClassifier eye_cascade;


// Helper functions (move to other file later)
vector<Rect_ <int> > detectFaces(Mat frame, string cascade_path) {
    vector<Rect_<int> > faces = vector<Rect_<int>>();
    
    /*Mat original_copy = frame.clone();  <-- now converting to gray before this function
    Mat gray_copy;
    cvtColor(original_copy, gray_copy, CV_BGR2GRAY);
     */
    face_cascade.detectMultiScale(frame, faces);
    return faces;
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

// Some helper functions to move between frames of references i.e. moving out of a smaller frame of reference within a larger one)
Point translatePoint(Point p, Rect currentFrame) {
    p.x = p.x + currentFrame.x;
    p.y = p.y + currentFrame.y;
    return p;
}

Rect translateRect(Rect r, Rect from) {
    r.x = r.x + from.x;
    r.y = r.y + from.y;
    return r;
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

double calculateGradientThreshold(Mat gradient) {
    Scalar stdDev;
    Scalar mean;
    meanStdDev(gradient, mean, stdDev);
    //cout << "mean: " << mean[0] << endl;
    //cout << "stdDev: " << stdDev[0] << endl;
    double stdDevScaled = stdDev[0] / sqrt(gradient.rows*gradient.cols);
    return mean[0] + stdDevScaled*50; // trying 50 for now based on recommendation from article
    
    //return mean[0] + stdDev[0]/10.0;
}

void normalizeMats(Mat &mat1, Mat &mat2) {
    Mat magnitudes = computeMagnitudes(mat1, mat2);
    
    // Get some sort of threshold so if the gradient is under the threshold, just set it to zero
    double gradThreshold = calculateGradientThreshold(magnitudes);
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
        unsigned char * rowPtr = weight.ptr<unsigned char>(j);
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
            //double dotProductMag = (dotProduct < 0) ? -1*dotProduct : dotProduct; // <-- shouldn't need this anymore
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
    scaleDownImage(eyeImageUnscaled, eyeImage);
    
    // Get the gradients of the eye image
    Mat gradX = computeXGradient(eyeImage);
    Mat gradY = computeYGradient(eyeImage);
    
    normalizeMats(gradX, gradY);
    
    // Get a "weight" Mat, equal to the inverse gray-scale image
    Mat weight = getWeightedImage(eyeImage);
   
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
    double numGradients = eyeImage.rows * eyeImage.cols;
    Mat resultScaled;
    result.convertTo(resultScaled, CV_32F, 1.0/numGradients);
    Point maxCenter;
    double maxDotProduct = 0;
    
    double currentMax = 0;
    Point currentMaxPoint;
    for (int j=0; j<resultScaled.rows; j++) {
        const float * resultPtr = resultScaled.ptr<float>(j);
        for (int i=0; i<resultScaled.cols; i++) {
            if (resultPtr[i] > currentMax) {
                currentMax = resultPtr[i];
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
    
    // Need to translate eye point back to original coordinate system (biggestFaceRect)
    Point resultCenter = unscalePoint(maxCenter, eyeROI);
    resultCenter.x += eyeROI.x;
    resultCenter.y += eyeROI.y;
    
    //const double * eyePtr = eyeImage.ptr<double>(maxCenter.y);
    //cout << "color at pupil: " << eyePtr[maxCenter.x] << endl;
    return resultCenter;
}

vector<Point> detectCorner(Mat frame, Rect eyeRect) {
    Mat img = frame(eyeRect);
    vector<Point> corners = vector<Point>();
    goodFeaturesToTrack(img, corners, 50, 0.02, 5); // what number makes sense for the "quality level" parameter?  just using 0.02 for now
    for (int i=0; i<corners.size(); i++) {
        corners.at(i) = translatePoint(corners.at(i), eyeRect);
        //circle(frame, corners.at(i), 2, CV_RGB(0,0,255), -1);
    }
    return corners;
}

vector<Rect> getEyeRegionRect(Rect faceRect) {
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
    
    vector<Rect> eyeRects = vector<Rect>();
    eyeRects.push_back(leftEyeRegion);
    eyeRects.push_back(rightEyeRegion);
    return eyeRects;
}

/* 
 this is the area in which we look for eye corners
*/
Rect getFilterArea(Rect eyeRect, Point pupil) {
    Rect filterRect = Rect();
    // These hard-coded dimensions work for my eyes but still need to be tested on others
    filterRect.x = pupil.x - eyeRect.width/3.5;
    filterRect.y = pupil.y - eyeRect.height/12;
    filterRect.height = eyeRect.height/5;
    filterRect.width = eyeRect.width/1.5;

    return filterRect;
}

void detectEyes(Mat &frame, Rect faceRect) {
    //cout << "detect eyes" << endl;
    Mat face_image = frame(faceRect);
    
    // Using haar cascade (commented out bc it ends up giving a lot of false positives)
    /*
    // find eye regions and draw them
    vector<Rect_<int> > eyes = vector<Rect_<int>>();
    eye_cascade.detectMultiScale(face_image, eyes);
    int numEyes = eyes.size();
    cout << "numEyes detected:" << numEyes << endl;
    vector<Mat> eyeVector = vector<Mat>();
    vector<Point> pupilVector = vector<Point>();
    
    // need to translate eye rect to the right frame of reference?
    for (int i=0; i<numEyes; i++) {
        Rect eye = eyes.at(i);
        cout << "(X,Y):" << eyes.at(i).x << ", " << eye.y << endl;
        cout << "faceRect (X,Y): " << faceRect.x << ", " << faceRect.y << endl;
        // Translate eye rect coordinates to face rect frame of reference
        eye.x = eye.x + faceRect.x;
        eye.y = eye.y + faceRect.y;
        cout << "(X,Y):" << eye.x << ", " << eye.y << endl;
        cout << "face_image rows: " << face_image.rows << endl;
        cout << "eye height: " << eye.height << endl;
        Mat eyeImage = getSubImage(face_image, faceRect, eye);
        eyeVector.push_back(eyeImage);
        Point pupil = findEyeCenter(eyeImage, eye, "eye");
        pupilVector.push_back(pupil);
        // draw
        rectangle(frame, eye, CV_RGB(0,0,255), 1);
        circle(eyeImage, pupil, 3, CV_RGB(0,0,255), -1);
    }
    */
    
    // Get the eye regions using a ratio method
    vector<Rect> eyeRects = getEyeRegionRect(faceRect);
    Rect leftEyeRect = eyeRects.at(0);
    Rect rightEyeRect = eyeRects.at(1); // left and right might be flipped
    
    // Get subimages
    Mat leftEyeImage = getSubImage(face_image, faceRect, leftEyeRect);
    Mat rightEyeImage = getSubImage(face_image, faceRect, rightEyeRect);
    
    // find eye center
    Point leftPupil = findEyeCenter(leftEyeImage, leftEyeRect, "left eye");
    Point rightPupil = findEyeCenter(rightEyeImage, rightEyeRect, "right eye");
    
    // find corners
    Rect leftFilterRect = getFilterArea(leftEyeRect, leftPupil);
    Rect rightFilterRect = getFilterArea(rightEyeRect, rightPupil);
    vector<Point> potentialLeftCorners = detectCorner(frame, leftEyeRect);
    vector<Point> potentialRightCorners = detectCorner(frame, rightEyeRect);
    //cout << "num left corners: " << leftCorners.size() << endl;
    //cout << "num right corners: " << rightCorners.size() << endl;
    //cout << "left pupil: " << leftPupil.x << ", " << leftPupil.y << endl;
    // filter through the detected corners to identify two candidate eye corners for each eye
    //look for the rightmost corner for each eye
    Point leftMaxCorner = Point();
    Point leftMinCorner = Point();

    int maxLeftCol = 0;
    int minLeftCol = 999999999; // haha get the actual constant?  should be fine bc images are small
    for (int i=0; i<potentialLeftCorners.size(); i++) {
        Point potential = potentialLeftCorners.at(i);
        if (potential.x > maxLeftCol &&
            potential.x > leftFilterRect.x && potential.x <= leftFilterRect.x + leftFilterRect.width &&
            potential.y > leftFilterRect.y && potential.y <= leftFilterRect.y + leftFilterRect.height) {
            maxLeftCol = potential.x;
            leftMaxCorner = potential;
        }
        else if (potential.x < minLeftCol &&
            potential.x > leftFilterRect.x && potential.x <= leftFilterRect.x + leftFilterRect.width &&
            potential.y > leftFilterRect.y && potential.y <= leftFilterRect.y + leftFilterRect.height) {
            minLeftCol = potential.x;
            leftMinCorner = potential;
        }
    }
    Point rightMaxCorner = Point();
    Point rightMinCorner = Point();
    int maxRightCol = 0;
    int minRightCol = 999999999;
    for (int i=0; i<potentialRightCorners.size(); i++) {
        Point potential = potentialRightCorners.at(i);
        if (potential.x > maxRightCol &&
            potential.x > rightFilterRect.x && potential.x < rightFilterRect.x + rightFilterRect.width &&
            potential.y > rightFilterRect.y && potential.y < rightFilterRect.y + rightFilterRect.height) {
            maxRightCol = potential.x;
            rightMaxCorner = potential;
        }
        else if (potential.x < minRightCol &&
            potential.x > rightFilterRect.x && potential.x <= rightFilterRect.x + rightFilterRect.width &&
            potential.y > rightFilterRect.y && potential.y <= rightFilterRect.y + rightFilterRect.height) {
            minRightCol = potential.x;
            rightMinCorner = potential;
        }
    }

    // draw THINGS ELSEWHERE
    rectangle(frame, leftEyeRect, CV_RGB(0,0,255), 1);
    rectangle(frame, rightEyeRect, CV_RGB(0,0,255), 1);
    circle(frame, leftPupil, 3, CV_RGB(0,0,255), -1);
    circle(frame, rightPupil, 3, CV_RGB(0,0,255), -1);
    circle(frame, leftMinCorner, 2, CV_RGB(0,0,255), -1);
    circle(frame, rightMinCorner, 2, CV_RGB(0,0,255), -1);
    circle(frame, leftMaxCorner, 2, CV_RGB(0,0,255), -1);
    circle(frame, rightMaxCorner, 2, CV_RGB(0,0,255), -1);
    rectangle(frame, leftFilterRect, CV_RGB(0,0,255), 1);
    rectangle(frame, rightFilterRect, CV_RGB(0,0,255), 1);

    return;
}

int main() {
    // Load the cascade
    if (!face_cascade.load(face_cascade_path)) {
        cout << "Error loading face cascade." << endl;
        return -1;
    }
    if (!eye_cascade.load(eye_cascade_path)) {
        cout << "Error loading eye cascade." << endl;
        return -1;
    }
    
    //namedWindow(main_window_name, CV_WINDOW_NORMAL);
    
    VideoCapture cap(-1); // switch back to 0?  which is which
    if (!cap.isOpened()) {
        cout << "Webcam is not open." << endl;
        return -1;
    }
    
    Mat frame;

    while (true) {
        cap.read(frame);
        //Mat weight;
        
        if (frame.rows > 0 && frame.cols > 0) {
            Mat gray;
            cvtColor(frame, gray, CV_BGR2GRAY);
            
            // Convert to HSV to filter out high saturation points
            Mat hsv;
            cvtColor(frame, hsv, CV_BGR2HSV);
            
            // Split into H, S, and V channels
            //Mat hsv_channels[3];
            //split(hsv, hsv_channels);
            //imshow("hsv H", hsv_channels[0]);
            //imshow("hsv S", hsv_channels[1]);
            //imshow("hsv V", hsv_channels[2]);
            
            // Playing around with histograms
            //Mat dst;
            //equalizeHist(hsv_channels[2], dst);
            //imshow("before", hsv_channels[2]);
            //imshow("after", dst);
            
            // Binarize the H channel
            /*Mat binary;
            adaptiveThreshold(hsv_channels[0], binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 701, 0);
            vector<Point> corners = detectCorner(binary);
            cout << "numCorners: " << corners.size();
            for (int i=0; i<corners.size(); i++) {
                circle(binary, corners.at(i),5,100);
            }
            imshow("binary", binary);
             */
            
            
            vector<Rect_ <int>> faces = detectFaces(gray, face_cascade_path);
            // Get the biggest face
            Rect biggestFace = getBiggestFace(faces);
            //cout << "num faces: " << faces.size();
            if (biggestFace.width > 0 && biggestFace.height > 0) {
                //weight = getWeightedImage(frame);
                //imshow("weighted", weight);
                drawFace(gray, biggestFace);
                detectEyes(gray, biggestFace);
            }
            
            
            // GETTING PUPILS
            // Smooth image
            // Detect eyes
            // Calculate gradients
            // Calculate direction vectors
            // Calculate dot products
            // Calculate weights
            // Determine best center
            
            //namedWindow("source_window", CV_WINDOW_AUTOSIZE);
            //namedWindow("equalized_window", CV_WINDOW_AUTOSIZE);
            imshow("face_detector", gray);
            
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
