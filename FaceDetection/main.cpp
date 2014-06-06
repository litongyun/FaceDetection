//
//  main.cpp
//  FaceDetection
//
//  Created by Mona Liu on 6/6/14.
//
//  Press Esc to exit

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay(Mat frame, CascadeClassifier& face_cascade);

// Global static variables
static const string face_cascade_name = "/Users/Tsuruko/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt.xml";
static const string saveLocation = "/Users/Tsuruko/face_images/";
static const int MaxFileNumber = 5; // Number of file to be saved

// Function main
int main() {
    VideoCapture capture(0);
    if (!capture.isOpened()) return -1;
    
    // Load the cascade
    CascadeClassifier faceCascade;
    if (!faceCascade.load(face_cascade_name)) {
        printf("--(!)Error loading\n");
        return (-1);
    }
    
    // Read the video stream
    Mat frame;
    
    for (;;) {
        capture >> frame;
        
        // Apply the classifier to the frame
        if (!frame.empty()) detectAndDisplay(frame, faceCascade);
        else {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        
        int c = waitKey(10);
        if (char(c) == 27) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

// Function detectAndDisplay
void detectAndDisplay(Mat frame, CascadeClassifier &cascade) {
    
    std::vector<Rect> faces;
    
    Mat frameGray;
    cvtColor(frame, frameGray, COLOR_BGR2GRAY);
    equalizeHist(frameGray, frameGray);
    
    // Detect faces
    const float sizeFact = 1.1f;
    const std::size_t minNeighbors = 2;
    const Size face_size(30, 30);
    cascade.detectMultiScale(frameGray, faces, sizeFact, minNeighbors, CASCADE_SCALE_IMAGE, face_size);
    
    // Set Region of Interest
    cv::Rect roi;
    cv::Rect largest;
    
    int filenum = 0;
    for (auto const& face : faces) {
        // Iterate through all current elements (detected faces)
        
        int area_curr = face.width * face.height;
        int area_larg = largest.width * largest.height;
        
        if (area_curr > area_larg) {
            roi = face;
            largest = face;
        }
        else roi = largest;
        
        //save the image of the detected face to a file
        Mat crop = frame(roi);
        Mat cropGray;
        //Mat res;
        //resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR);
        cvtColor(crop, cropGray, CV_BGR2GRAY);
        
        if (filenum < MaxFileNumber) {
            stringstream ssfn;
            ssfn << saveLocation << filenum << ".png";
            filenum++;
            imwrite(ssfn.str(), cropGray);
        }
    
        // Display detected faces on live stream
        Point pt1(face.x, face.y);
        Point pt2((face.x + face.height), (face.y + face.width));
        Scalar color = Scalar(0, 255, 0);
        const int thickness = 2;
        const int lineType = 8;
        const int shift = 0;
        rectangle(frame, pt1, pt2, color, thickness, lineType, shift);
    }
    
    // Show image
    stringstream sstm;
    string text;
    CvPoint loc = cvPoint(30, 30);
    CvScalar color = cvScalar(0, 0, 255);
    const double fontScale = 0.8;
    const int thickness = 2;
    sstm << "Face area size: " << roi.width << "x" << roi.height;
    text = sstm.str();
    putText(frame, text, loc, FONT_HERSHEY_COMPLEX_SMALL, fontScale, color, thickness, CV_AA);
    imshow("Capture - Face detection", frame);
}