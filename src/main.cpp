#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include<chrono>
#include<string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "../include/ORBextractor.h"

#define nFeatures     500
#define fScaleFactor  1.2
#define nLevels       8
#define fIniThFAST    20
#define fMinThFAST    7

#define DIST_THRESHOLD 30

#define max(a,b) a>b?a:b

using namespace  std;
using namespace  cv;
using namespace  ORB_SLAM2;
using namespace chrono;

float computerConstrast(const Mat img);

int main ( int argc, char** argv )
{

    Mat image,image2;
    image = imread ( argv[1] ); 
    image2 = imread ( argv[2] );

    // 判断图像文件是否正确读取
    if ( image.data == nullptr || image2.data == nullptr) //数据不存在,可能是文件不存在
    {
        cerr<<"文件不存在."<<endl;
        return 0;
    }
	
    //resize(src_img,image,Size(640,360),0,0,INTER_LINEAR);
    //resize(src_img2,image2,Size(640,360),0,0,INTER_LINEAR);


    //imwrite("test1.png",image);
    //imwrite("test2.png",image2);

    ORBextractor* mpIniORBextractor;
    ORBextractor* mpIniORBextractor2;
    Mat mImGray,mImGray2;
    vector<KeyPoint> mvKeys,mvKeys2;
    Mat mDescriptors,mDescriptors2;
    // 文件顺利读取, 首先输出一些基本信息
    cout<<"图像宽为"<<image.cols<<",高为"<<image.rows<<",通道数为"<<image.channels()<<endl;

    // cv::imshow ( "image", image );      // 用cv::imshow显示图像
    // waitKey(-1);
    //Mat image_l,image_r;

    cvtColor(image, mImGray, CV_BGR2GRAY);
    cvtColor(image2, mImGray2, CV_BGR2GRAY);
   
    //直方图均衡化
    //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.0, cv::Size(3, 3));
    //clahe->apply(mImGray, mImGray);
    //clahe->apply(mImGray2, mImGray2);


    mpIniORBextractor=new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    mpIniORBextractor2=new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    auto start=system_clock::now();   
    (*mpIniORBextractor)(mImGray,cv::Mat(),mvKeys,mDescriptors);
    (*mpIniORBextractor2)(mImGray2,cv::Mat(),mvKeys2,mDescriptors2);

    auto end=system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
   cout<<"ORB提取 time:"<<double(duration.count()) * microseconds::period::num / microseconds::period::den <<endl;
    Mat outimg;
    drawKeypoints(mImGray,mvKeys,outimg,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
  
  //matching

  vector<DMatch> matches;
  BFMatcher matcher = BFMatcher(NORM_HAMMING);
  matcher.match(mDescriptors,mDescriptors2,matches);
  
  //Step6
  int min_dist = 150;
  int max_dist = 0;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<min_dist)
    {
      min_dist = matches[i].distance;
    }
    if(matches[i].distance>max_dist)
    {
      max_dist = matches[i].distance;
    }
  }

  vector<Point2f> points1;
  vector<Point2f> points2;
  vector<DMatch> good_matches;
  int th = max(4 * min_dist,DIST_THRESHOLD);
  cout << min_dist << " " << max_dist << " " << th << endl;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<th)
    {
      good_matches.push_back(matches[i]);
      points1.push_back(mvKeys[matches[i].queryIdx].pt);
      points2.push_back(mvKeys2[matches[i].trainIdx].pt);
    }
  }
  cout << matches.size() << " " << good_matches.size() << endl;
  int ransacReprojThreshold = 5;
  Mat H12;
  H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold );

	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);
	int numInliner = 0;
	vector<char> matchesMask;
	matchesMask.resize(good_matches.size(), 0 );

	for( size_t i1 = 0; i1 < points1.size(); i1++ )  //保存‘内点’
	{
		if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) <= ransacReprojThreshold ) //给内点做标记
		{
			matchesMask[i1] = 1;
			numInliner++;
		}	
	}
	Mat match_img2;   //滤除‘外点’后
	drawMatches(mImGray,mvKeys,mImGray2,mvKeys2,good_matches,match_img2,Scalar(0,0,255),Scalar::all(-1),matchesMask);
  	imwrite("matchingResult.png",match_img2);


    //for(int i=0;i<mvKeys.size();i++)
        //cout<<mvKeys[i].pt.x<<" "<<mvKeys[i].pt.y<<endl;
     ofstream outfile;
     outfile.open("matchingInfo.txt",ios::app);
     outfile << mvKeys.size()<< " " << mvKeys2.size()<< endl;
     outfile << good_matches.size()<< " " << numInliner<< endl;
     outfile.close();
     cout<<"keypoint1 size is :"<<mvKeys.size()<<"   keypoint2 size is :"<<mvKeys2.size()<<endl;
     cout<<"good_matches: " << good_matches.size() << " inlier matches: " << numInliner << endl; 
    //imshow("ORBkeypoint",outimg);
   // imshow("image",image);
    if(mpIniORBextractor)
        delete mpIniORBextractor;
    if(mpIniORBextractor2)
        delete mpIniORBextractor2;
    //waitKey(-1);

    computerConstrast(mImGray);
    computerConstrast(mImGray2);

	//namedWindow( "matches", WINDOW_NORMAL);
	imshow("matches", match_img2);
	waitKey(0);
    return 0;
}

float computerConstrast(const Mat img){
  Scalar mean; 
  Scalar stddev; 

  cv::meanStdDev( img, mean, stddev ); 
  double mean_pxl = mean.val[0]; 
  double stddev_pxl = stddev.val[0]; 
  double sum = 0;
  double constrast;
    int M = img.cols;
    int N = img.rows;
    for(int i = 0; i < N; i ++)
    {
      for(int j = 0; j < M; j++)
      {
        sum += (img.at<uchar>(i,j) - mean_pxl)*(img.at<uchar>(i,j) - mean_pxl);
      }
    }

    constrast = sqrt(sum / M / N);
     ofstream outfile;
     outfile.open("matchingInfo.txt",ios::app);
     outfile << constrast<< " ";
     outfile.close();
    cout << "Image Constrast: " << constrast << "  stddev_pxl: " << stddev_pxl << endl;

}
