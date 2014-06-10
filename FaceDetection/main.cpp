//
//  main.cpp
//  FaceDetection
//
//  Created by Mona Liu on 6/6/14.
//
//  Face Detection in c++ using opencv libraries
//  Press Esc to exit


//
//  Created by Cedric Verstraeten on 18/02/14.
//  Copyright (c) 2014 Cedric Verstraeten. All rights reserved.
//

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void detectFace(Mat & frame, Mat crop, Rect area, CascadeClassifier& face_cascade);
Mat motionDifference(Mat prevFrame, Mat currFrame, Mat nextFrame);
int detectMotion(const Mat & motion, Mat & result, Mat & resCrop, int xBeg, int xEnd,
                 int yBeg, int yEnd, int maxDeviation, Rect & motionArea);

// Global static variables
static const string cascadeName = "/Users/Tsuruko/Desktop/School/2014 Spring/CSE190/FaceDetection/faces.xml";
static const string cascadeName1 = "/Users/Tsuruko/Desktop/School/2014 Spring/CSE190/FaceDetection/faces1.xml";
static const string cascadeName2 = "/Users/Tsuruko/Desktop/School/2014 Spring/CSE190/FaceDetection/faces2.xml";

static const int motionThresh = 5;
static const int DELAY = 5; // in mseconds

int main () {
    
    // Load the cascade
    CascadeClassifier faceCascade;
    if (!faceCascade.load(cascadeName1)) {
        printf("--(!)Error loading cascade\n");
        return (-1);
    }
    
    // Set up camera
    CvCapture * camera = cvCaptureFromCAM(CV_CAP_ANY);
    cvSetCaptureProperty(camera, CV_CAP_PROP_FRAME_WIDTH, 1280); // width of viewport of camera
    cvSetCaptureProperty(camera, CV_CAP_PROP_FRAME_HEIGHT, 720); // height of ...
    
    // Take images and convert them to gray
    Mat result, resCrop;
    Mat prevFrame = result = cvQueryFrame(camera);
    Mat currFrame = cvQueryFrame(camera);
    Mat nextFrame = cvQueryFrame(camera);
    cvtColor(currFrame, currFrame, CV_RGB2GRAY);
    cvtColor(prevFrame, prevFrame, CV_RGB2GRAY);
    cvtColor(nextFrame, nextFrame, CV_RGB2GRAY);
    
    // Detect motion in window
    Mat motion;
    Rect motionArea;
    int numChanges, numSequence = 0;
    int xBeg = 10, xEnd = currFrame.cols-11;
    int yBeg = 10, yEnd = currFrame.rows-11;
    
    // Maximum deviation of the image, the higher the value, the more motion is allowed
    int maxDeviation = 20;
    
    //loop constantly displaying livestream until ESC key is hit
    while (true) {
        // Take a new image
        prevFrame = currFrame;
        currFrame = nextFrame;
        nextFrame = cvQueryFrame(camera);
        result = nextFrame;
        cvtColor(nextFrame, nextFrame, CV_RGB2GRAY);
        
        motion = motionDifference(prevFrame, currFrame, nextFrame);
        
        numChanges = detectMotion(motion, result, resCrop, xBeg, xEnd, yBeg, yEnd, maxDeviation, motionArea);
        
        // If a lot of changes happened, we assume something changed.
        if(numChanges>=motionThresh) {
            if(numSequence>0){
                //only detect faces in areas of the frame with movement
                detectFace(result, resCrop, motionArea, faceCascade);
                imshow("Motion and Face Detection", result);
            }
            numSequence++;
        }
        else {
            numSequence = 0;
            cvWaitKey (DELAY);
        }
        int c = waitKey(10);
        if (char(c) == 27) break;
    }
    return 0;
}


// Check if there is motion in the result matrix
// count the number of changes and return.
int detectMotion(const Mat & motion, Mat & result, Mat & resCrop, int xBeg, int xEnd,
                 int yBeg, int yEnd, int maxDeviation, Rect &motionArea) {
    
    Scalar color(0,255,255); // yellow for rectangle

    // calculate the standard deviation
    Scalar mean;
    Scalar stdDev;
    meanStdDev(motion, mean, stdDev);
    
    // ignore extreme changes
    if(stdDev[0] < maxDeviation) {
        int numChanges = 0;
        int xMin = motion.cols;
        int yMin = motion.rows;
        int xMax = 0;
        int yMax = 0;
        // loop over image and detect changes
        for(int j = yBeg; j < yEnd; j+=2) { // height
            for(int i = xBeg; i < xEnd; i+=2) { // width
                // check if at pixel (j,i) intensity is equal to 255
                // this means that the pixel is different in the sequence
                // of images (prevFrame, currFrame, nextFrame)
                if(static_cast<int>(motion.at<uchar>(j,i)) == 255) {
                    numChanges++;
                    if(xMin>i) xMin = i;
                    if(yMin>j) yMin = j;
                    if(xMax<i) xMax = i;
                    if(yMax<j) yMax = j;
                }
            }
        }
        if(numChanges) {
            //check if not out of bounds
            if(xMin-10 > 0) xMin -= 10;
            if(yMin-10 > 0) yMin -= 10;
            if(xMax+10 < result.cols-1) xMax += 10;
            if(yMax+10 < result.rows-1) yMax += 10;
            // draw rectangle round the changed pixel
            Point x(xMin,yMin);
            Point y(xMax,yMax);
            Rect rect(x,y);
            motionArea = rect;
            Mat crop = result(rect);
            crop.copyTo(resCrop);
            rectangle(result,rect,color,1);
            
            //display motion difference image and detected area with motion
            //Mat m = motion;
            //rectangle(m,rect,Scalar(255, 255, 255),1);
            //imshow("motion difference", m);
            
        }
        return numChanges;
    }
    return 0;
}

Mat motionDifference(Mat prevFrame, Mat currFrame, Mat nextFrame) {
    // Calc differences between the images and do AND-operation
    // threshold image, low differences are ignored (ex. contrast change due to sunlight)
    
    Mat difference1;
    Mat difference2;
    Mat result;
    Mat erodeKernel = getStructuringElement(MORPH_RECT, Size(2,2));
    
    //double check that frames are the same size
    Size p = prevFrame.size();
    Size c = currFrame.size();
    Size n = nextFrame.size();
    if (p == n) absdiff(prevFrame, nextFrame, difference1);
    if (c == n) absdiff(nextFrame, currFrame, difference2);
    
    bitwise_and(difference1, difference2, result);
    threshold(result, result, 20, 255, CV_THRESH_BINARY);
    erode(result, result, erodeKernel);
    
    return result;
}

void detectFace(Mat & frame, Mat crop, Rect motionArea, CascadeClassifier &cascade) {
    vector<Rect> faces;
    
    Mat frameGray;
    //frame.copyTo(frameGray);
    crop.copyTo(frameGray);
    
    //cvtColor(frame, frameGray, COLOR_BGR2GRAY);
    cvtColor(crop, frameGray, COLOR_BGR2GRAY);
    equalizeHist(frameGray, frameGray);
    
    // Detect faces
    const float sizeFact = 1.1f;
    const size_t minNeighbors = 2;
    const Size face_size(30, 30);
    cascade.detectMultiScale(frameGray, faces, sizeFact, minNeighbors, CASCADE_SCALE_IMAGE, face_size);
    
    // Set Region of Interest
    Rect roi;
    Rect largest;
    
    for (auto const& face : faces) {
        // Iterate through all current elements (detected faces)
        
        int area_curr = face.width * face.height;
        int area_larg = largest.width * largest.height;
        
        if (area_curr > area_larg) {
            roi = face;
            largest = face;
        }
        else roi = largest;
        
        // Display rectangle on detected faces on live stream
        int offsetFacex = face.x + motionArea.tl().x;
        int offsetFacey = face.y + motionArea.tl().y;
        
        Point pt1(offsetFacex, offsetFacey);
        Point pt2((offsetFacex + face.height), (offsetFacey + face.width));
        Scalar color = Scalar(0, 255, 0);
        const int thickness = 2;
        const int lineType = 8;
        const int shift = 0;
        rectangle(frame, pt1, pt2, color, thickness, lineType, shift);
    }
    
    // Add detect area box and text to frame
    stringstream sstm;
    string text;
    CvPoint loc = cvPoint(30, 30);
    CvScalar color = cvScalar(0, 0, 255);
    const double fontScale = 0.8;
    const int thickness = 1;
    sstm << "Face area size: " << roi.width << "x" << roi.height;
    text = sstm.str();
    putText(frame, text, loc, FONT_HERSHEY_COMPLEX_SMALL, fontScale, color, thickness, CV_AA);
}