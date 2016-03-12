//
//  main.cpp
//  eyeTracker
//
//  Created by Pamina Lin on 3/9/16.
//  Copyright © 2016 Pamina Lin. All rights reserved.
//


#include <iostream>
//#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;


// Global variables
string face_cascade_path = "/Users/paminalin/Developer/eyeTracker/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade; // CascadeClassifier class detects objects in a video stream (load Haar or LBP classifiers or detectMultiScale to perform detection)
string main_window_name = "Capture - Face detection";
string face_window_name = "Capture - Face";

int main() {
    // Load the cascade
    if (!face_cascade.load(face_cascade_path)) {
        cout << "Error loading face cascade." << endl;
        return -1;
    }
    
    //namedWindow(main_window_name, CV_WINDOW_NORMAL);
    
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Webcame is not open." << endl;
    }
    
    while (true) {
        Mat frame;
        cap.read(frame);
        
        Mat original = frame.clone();
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        
        vector<Rect_<int> > faces;
        face_cascade.detectMultiScale(gray, faces);
        
        for (int i=0; i<faces.size(); i++) {
            Rect face_i = faces[i];
            Mat face = gray(face_i);
            rectangle(original, face_i, CV_RGB(0,255,0), 1);
        }
        
        imshow("face_detector", original);
        
        char key = (char) waitKey(1);
        
        
         
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
