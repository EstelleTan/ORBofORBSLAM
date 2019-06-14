/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>


namespace ORB_SLAM2
{

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    /** \brief 计算图像的orb特征及描述，将orb特征分配到一个四叉树当中
        *  目前mask参数是被忽略的，没有实现
        */
    void operator()( cv::InputArray image, cv::InputArray mask,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);
   
    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }
    
    std::vector<cv::Mat> mvImagePyramid;

protected:

    /** \brief 计算图像金字塔
        */
    void ComputePyramid(cv::Mat image);
    /** \brief  对金字塔图像进行角点检测
     * 对影像金字塔中的每一层图像进行特征点的计算。具体的计算过程是将影像格网分割为小区域，每一个小区域独立使用 FAST 角点检测
     *
     * 使用了八叉树（其实是平面上的四叉树）的数据结构来存储提取出的特征点
        */
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    //金字塔中每一层提取出的特征点放在不同的vector<KeyPoint>中
    //该树结构除了根节点其实只实现了3层，最顶层的node数量由图像的横纵比决定（例如2）;下面两层最多产生64个叶子.对于前面提到的特征点数，
    //平均每个分割节点中分布一两个特征点，如果该叶子中含有较多特征点，则选取其中Harris响应值(由OpenCV的KeyPoint.response属性计算的）最大的，其他的抛弃
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);
    std::vector<cv::Point> pattern;//!<用于存放训练的模板

    int nfeatures;//!<最多提取的特征点的数量
    double scaleFactor;//!<金字塔图像之间的尺度参数
    int nlevels;//!<高斯金字塔的层数
    int iniThFAST;//!<默认设置fast角点阈值
    int minThFAST;//!<设置fast角点阈值

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

};

} //namespace ORB_SLAM

#endif

