#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"

int main(int argc,char** argv)
{
    // Load Face Detector
    cv::CascadeClassifier faceDetector("/home/ika/Dev/landmarks/src/haarcascade_frontalface_alt2.xml");

    // Create an instance of Facemark Local Binary Fitting
    cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel("/home/ika/Dev/landmarks/src/lbfmodel.yaml");

    // Set up webcam for video capture
    cv::VideoCapture cam(0);
    // Variable to store a video frame and its grayscale
    cv::Mat frame, gray;
    int count=0;
    
    // Read a frame
    while(cam.read(frame))
    {
      
      // Find face
      std::vector<cv::Rect> faces;
      // Convert frame to grayscale because
      // faceDetector requires grayscale image.
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

      // Detect faces
      faceDetector.detectMultiScale(gray, faces);

      //using vector of a vector (2D) in case of more than 1 face in frame
      std::vector< std::vector<cv::Point2f> > landmarks;
      
      // Run landmark detector
      bool success = facemark->fit(frame,faces,landmarks);

      if(success)
      {
        // If successful, render the landmarks on the face
        for(int i = 0; i < landmarks.size(); i++)
        {
          drawLandmarks(frame, landmarks[i]);
        }
      }



      // Display results 
      imshow("Facial Landmark Detection", frame);
      // Exit loop if ESC is pressed
      if (cv::waitKey(1) == 27) break;
      
    }
    return 0;
}
