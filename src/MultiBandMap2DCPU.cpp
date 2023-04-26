/******************************************************************************

  This file is part of Map2DFusion.

  Copyright 2016 (c)  Yong Zhao <zd5945@126.com> http://www.zhaoyong.adv-ci.com

  ----------------------------------------------------------------------------

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.

*******************************************************************************/
#include "MultiBandMap2DCPU.h"

#include <GL/gl.h>
#include <base/Svar/Svar.h>
#include <base/time/Global_Timer.h>
#include <gui/gl/SignalHandle.h>
#include <gui/gl/glHelper.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching/stitcher.hpp>

#define HAS_GOOGLEMAP
#ifdef HAS_GOOGLEMAP
#include <base/Svar/Scommand.h>
#include <hardware/Gps/utils_GPS.h>
#endif

using namespace std;

/**

  __________max
  |    |    |
  |____|____|
  |    |    |
  |____|____|
 min
 */

/**
 * @brief Destructor of MultiBandMap2DCPUEle
 *
 */
// Destructor is called when
// A local (automatic) object with block scope goes out of scope.
// Or an object allocated using the new operator is explicitly deallocated using delete.
MultiBandMap2DCPU::MultiBandMap2DCPUEle::~MultiBandMap2DCPUEle() {
    // Check if texture exists
    if (texName) {
        // Delete the texture
        pi::gl::Signal_Handle::instance().delete_texture(texName);
    }
}

/**
 * @brief Divide src matrix in place by weight matrix, elemnt wise
 *
 * @param weight Weight matrix
 * @param src Source matrix
 * @return true
 * @return false Wrong matirx type
 */
bool MultiBandMap2DCPU::MultiBandMap2DCPUEle::normalizeUsingWeightMap(const cv::Mat &weight, cv::Mat &src) {
    // Check type of input matrixes
    if (!(src.type() == CV_32FC3 && weight.type() == CV_32FC1)) {
        return false;
    }

    // Element wise division of src by weight. Result is sotred in src
    pi::Point3f *srcP = (pi::Point3f *)src.data;
    float *weightP = (float *)weight.data;
    for (float *Pend = weightP + weight.cols * weight.rows; weightP != Pend; weightP++, srcP++) {
        *srcP = (*srcP) / (*weightP + 1e-5);  // avoid null division
    }
    return true;
}

/**
 * @brief Multiply src matrix in place by weight matrix, element wise
 *
 * @param weight Weight matrix
 * @param src Source matrix
 * @return true
 * @return false Wrong matirx type
 */
bool MultiBandMap2DCPU::MultiBandMap2DCPUEle::mulWeightMap(const cv::Mat &weight, cv::Mat &src) {
    // Check type of input matrixes
    if (!(src.type() == CV_32FC3 && weight.type() == CV_32FC1)) {
        return false;
    }

    // Element wise multiplication of src by weight. Result is sotred in src
    pi::Point3f *srcP = (pi::Point3f *)src.data;
    float *weightP = (float *)weight.data;
    for (float *Pend = weightP + weight.cols * weight.rows; weightP != Pend; weightP++, srcP++) {
        *srcP = (*srcP) * (*weightP);
    }
    return true;
}

cv::Mat MultiBandMap2DCPU::MultiBandMap2DCPUEle::blend(const std::vector<SPtr<MultiBandMap2DCPUEle>> &neighbors) {
    // Return empty matrix if laplace pyramid is empty
    if (!pyr_laplace.size())
        return cv::Mat();
    // For non-border pixels, blend with neighbors
    if (neighbors.size() == 9) {
        // blend with neighbors, this obtains better visualization
        int flag = 0;
        for (int i = 0; i < neighbors.size(); i++) {
            flag <<= 1;  // set flag to itsself shifted by one bit to the left
            if (neighbors[i].get() && neighbors[i]->pyr_laplace.size())
                // flag will have bit 0 for neighbors with empty pyr_laplace and 1 otherwise
                flag |= 1;  // bitwise or
        }
        switch (flag) {
            // 0X01FF = 511 (decimal) = 111111111 (binary)
            // In case all 9 neighbors exist and have a non-empty pyr_laplace
            case 0X01FF: {
                vector<cv::Mat> pyr_laplaceClone(pyr_laplace.size());
                for (int i = 0; i < pyr_laplace.size(); i++) {
                    int borderSize = 1 << (pyr_laplace.size() - i - 1);
                    int srcrows = pyr_laplace[i].rows;
                    int dstrows = srcrows + (borderSize << 1);
                    pyr_laplaceClone[i] = cv::Mat(dstrows, dstrows, pyr_laplace[i].type());

                    for (int y = 0; y < 3; y++) {
                        for (int x = 0; x < 3; x++) {
                            const SPtr<MultiBandMap2DCPUEle> &ele = neighbors[3 * y + x];
                            pi::ReadMutex lock(ele->mutexData);
                            if (ele->pyr_laplace[i].empty())
                                continue;
                            cv::Rect src, dst;
                            src.width = dst.width = (x == 1) ? srcrows : borderSize;
                            src.height = dst.height = (y == 1) ? srcrows : borderSize;
                            src.x = (x == 0) ? (srcrows - borderSize) : 0;
                            src.y = (y == 0) ? (srcrows - borderSize) : 0;
                            dst.x = (x == 0) ? 0 : ((x == 1) ? borderSize : (dstrows - borderSize));
                            dst.y = (y == 0) ? 0 : ((y == 1) ? borderSize : (dstrows - borderSize));
                            ele->pyr_laplace[i](src).copyTo(pyr_laplaceClone[i](dst));
                        }
                    }
                }

                cv::detail::restoreImageFromLaplacePyr(pyr_laplaceClone);

                {
                    cv::Mat result;
                    int borderSize = 1 << (pyr_laplace.size() - 1);
                    pyr_laplaceClone[0](cv::Rect(borderSize, borderSize, ELE_PIXELS, ELE_PIXELS)).copyTo(result);
                    return result.setTo(cv::Scalar::all(0), weights[0] == 0);
                }
            } break;
            default:
                break;
        }
    }

    {
        // blend by self
        vector<cv::Mat> pyr_laplaceClone(pyr_laplace.size());
        for (int i = 0; i < pyr_laplace.size(); i++) {
            pyr_laplaceClone[i] = pyr_laplace[i].clone();
        }

        cv::detail::restoreImageFromLaplacePyr(pyr_laplaceClone);

        return pyr_laplaceClone[0].setTo(cv::Scalar::all(0), weights[0] == 0);
    }
}

// this is a bad idea, just for test
/**
 * @brief Function for updating the orthomosaic texture of an Ele
 *
 * @param neighbors Neighbors of the Ele
 * @return true
 * @return false If there is no texture map returned or wrong type
 */
bool MultiBandMap2DCPU::MultiBandMap2DCPUEle::updateTexture(const std::vector<SPtr<MultiBandMap2DCPUEle>> &neighbors) {
    // Blend with the neighbors (if neighbors is not NULL)
    cv::Mat tmp = blend(neighbors);
    uint type = 0;

    // Check new texture map type
    // If there is no texture map returned, fail
    if (tmp.empty()) {
        return false;
    } else if (tmp.type() == CV_16SC3) {
        tmp.convertTo(tmp, CV_8UC3);
        type = GL_UNSIGNED_BYTE;
    } else if (tmp.type() == CV_32FC3) {
        type = GL_FLOAT;
    }

    // Fail if wrong type
    if (!type) {
        return false;
    }

    // Check if the texture of the Ele has been created before
    if (texName == 0)  // Texture has not been created before
    {
        // Create new texture
        glGenTextures(1, &texName);
        glBindTexture(GL_TEXTURE_2D, texName);

        // Apply blended texture map to the texture object
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tmp.cols, tmp.rows, 0, GL_BGR, type, tmp.data);

        // Apply texture magnification function and texture minifying function
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    } else {
        // Update previously created texture with the blended texture map
        glBindTexture(GL_TEXTURE_2D, texName);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tmp.cols, tmp.rows, 0, GL_BGR, type, tmp.data);
    }

    // Update the last texture map and last texture map weights
    SvarWithType<cv::Mat>::instance()["LastTexMat"] = tmp;
    SvarWithType<cv::Mat>::instance()["LastTexMatWeight"] = weights[0].clone();

    Ischanged = false;
    return true;
}

MultiBandMap2DCPU::MultiBandMap2DCPUData::MultiBandMap2DCPUData(double eleSize_, double lengthPixel_, pi::Point3d max_,
        pi::Point3d min_, int w_, int h_, const std::vector<SPtr<MultiBandMap2DCPUEle>> &d_)
    : _eleSize(eleSize_),
      _eleSizeInv(1. / eleSize_),
      _lengthPixel(lengthPixel_),          // Pixel length (Map2D.Resolution)
      _lengthPixelInv(1. / lengthPixel_),  // Inv. of pixel length
      _min(min_),                          // Window minimum coordinates
      _max(max_),                          // WIndow maximum coordinates
      _w(w_),
      _h(h_),
      _data(d_)  // Vector of MultiBandMap2DCPUEle
{
    _gpsOrigin = svar.get_var("GPS.Origin", _gpsOrigin);  // Set GPS origin
}

bool MultiBandMap2DCPU::MultiBandMap2DCPUData::prepare(SPtr<MultiBandMap2DCPUPrepare> prepared) {
    // Check if already prepared
    if (_w || _h) {
        return false;  // already prepared
    }

    // Find the minimum and maximum x, y, z coordinates in the loaded poses
    _max = pi::Point3d(-1e10, -1e10, -1e10);
    _min = -_max;
    for (std::deque<std::pair<cv::Mat, pi::SE3d>>::iterator it = prepared->_frames.begin();
            it != prepared->_frames.end(); it++) {
        pi::SE3d &pose = it->second;
        pi::Point3d &t = pose.get_translation();
        _max.x = t.x > _max.x ? t.x : _max.x;
        _max.y = t.y > _max.y ? t.y : _max.y;
        _max.z = t.z > _max.z ? t.z : _max.z;
        _min.x = t.x < _min.x ? t.x : _min.x;
        _min.y = t.y < _min.y ? t.y : _min.y;
        _min.z = t.z < _min.z ? t.z : _min.z;
    }
    // Make sure all the poses are on one side of the projection plane
    if (_min.z * _max.z <= 0) {
        return false;
    }
    cout << "Box:Min:" << _min << ",Max:" << _max << endl;

    //// Compute the 3D bounding box using the camera parameters
    // Compute max height from the min/max z
    double maxh;
    if (_max.z > 0) {
        maxh = _max.z;
    } else {
        maxh = -_min.z;
    }
    // Compute the vector from 3D point back-projected from the image bottom-right corner to the 3D point back-projected
    // from the image top-left corner
    pi::Point3d line = prepared->UnProject(pi::Point2d(prepared->_camera.w, prepared->_camera.h)) -
                       prepared->UnProject(pi::Point2d(0, 0));
    // Define the radius in metres as half the max height multiplied by the line length.
    double radius = 0.5 * maxh * sqrt(line.x * line.x + line.y * line.y);
    // Define the pixel length as the cell size in metres. Get from configuration or compute automatically.
    _lengthPixel = svar.GetDouble("Map2D.Resolution", 0);
    if (!_lengthPixel) {
        cout << "Auto resolution from max height " << maxh << "m.\n";
        // Automatic pixel length (m) is 2*radius divided by the length of the image hypotenuse in pixels
        _lengthPixel = 2 * radius /
                       sqrt(prepared->_camera.w * prepared->_camera.w + prepared->_camera.h * prepared->_camera.h);
        // Scale the pixel length
        _lengthPixel /= svar.GetDouble("Map2D.Scale", 1);
    }
    cout << "Map2D.Resolution=" << _lengthPixel << endl;
    // Inverse pixel length computed for future calculations
    _lengthPixelInv = 1. / _lengthPixel;
    // Increase the bounds by the radius
    _min = _min - pi::Point3d(radius, radius, 0);
    _max = _max + pi::Point3d(radius, radius, 0);
    // Compute the center
    pi::Point3d center = 0.5 * (_min + _max);
    // Scale and shift the min and max?
    _min = 2 * _min - center;
    _max = 2 * _max - center;
    // Compute the size of each data element in metres
    _eleSize = ELE_PIXELS * _lengthPixel;
    _eleSizeInv = 1. / _eleSize;
    // Save the grid's w, h, max, and resize to w*h
    _w = ceil((_max.x - _min.x) / _eleSize);
    _h = ceil((_max.y - _min.y) / _eleSize);
    _max.x = _min.x + _eleSize * _w;
    _max.y = _min.y + _eleSize * _h;
    _data.resize(_w * _h);

    // Save the GPS origin
    _gpsOrigin = svar.get_var("GPS.Origin", _gpsOrigin);
    return true;
}

/**
 * @brief Constructor for MultiBandMap2DCPU
 *
 * @param thread Threding flag,  true if running in separate thread
 */
MultiBandMap2DCPU::MultiBandMap2DCPU(bool thread)
    : alpha(svar.GetInt("Map2D.Alpha", 0)),                      // GL_ALPHA_TEST flag, true => enable GL_ALPHA_TEST
      _valid(false),                                             // Prepare flag, true if prepared
      _thread(thread),                                           // Threading flag, true if running in separate thread
      _bandNum(svar.GetInt("MultiBandMap2DCPU.BandNumber", 5)),  // Number of bands for the renderFrame
      _highQualityShow(
              svar.GetInt("MultiBandMap2DCPU.HighQualityShow", 1))  // High quality flag, true => blend using neighbors
{
    _bandNum = min(_bandNum,
            static_cast<int>(ceil(log(ELE_PIXELS) / log(2.0))));  // Calculate number of bands for the renderFrame
}

bool MultiBandMap2DCPU::prepare(const pi::SE3d &plane, const PinHoleParameters &camera,
        const std::deque<std::pair<cv::Mat, pi::SE3d>> &frames) {
    // insert frames
    SPtr<MultiBandMap2DCPUPrepare> p(new MultiBandMap2DCPUPrepare);
    SPtr<MultiBandMap2DCPUData> d(new MultiBandMap2DCPUData);

    if (p->prepare(plane, camera, frames))
        if (d->prepare(p)) {
            pi::WriteMutex lock(mutex);
            prepared = p;
            data = d;
            weightImage.release();
            if (_thread && !isRunning())
                start();
            _valid = true;
            return true;
        }
    return false;
}

/**
 * @brief Render new frame or insert new frame in queue
 *
 * @param img New image to be rendered/inserted
 * @param pose Pose corresponding to the image
 * @return true Insertion was successful or rendering was successful (threading)
 * @return false MultiBandMap2DCPU was not prepared
 */
bool MultiBandMap2DCPU::feed(cv::Mat img, const pi::SE3d &pose) {
    // Check if MultiBandMap2DCPU was prepared
    if (!_valid) {
        return false;  // MultiBandMap2DCPU was not prepared
    }

    // Get p and d
    SPtr<MultiBandMap2DCPUPrepare> p;
    SPtr<MultiBandMap2DCPUData> d;
    {
        pi::ReadMutex lock(mutex);
        p = prepared;
        d = data;
    }

    // Create frame pair while converting the pose
    std::pair<cv::Mat, pi::SE3d> frame(img, p->_plane.inverse() * pose);

    // If threding is enabled
    if (_thread) {
        pi::WriteMutex lock(p->mutexFrames);

        // Insert new frame
        p->_frames.push_back(frame);
        // If more than 20 frames in queue remove the oldest one
        if (p->_frames.size() > 20) {
            p->_frames.pop_front();
        }

        return true;
    } else {
        // If threading is not enabled render the frame
        return renderFrame(frame);
    }
}

bool MultiBandMap2DCPU::renderFrame(const std::pair<cv::Mat, pi::SE3d> &frame) {
    // Get the prepared frames (p) and grid data (d). Note that the mutex is used incorrectly. A (shared) pointer is
    // acquired but the data can still be read/written by this and any other thread once the mutex goes out of scope.
    SPtr<MultiBandMap2DCPUPrepare> p;
    SPtr<MultiBandMap2DCPUData> d;
    {
        pi::ReadMutex lock(mutex);
        p = prepared;
        d = data;
    }

    // Check the image data matches expected dimensions and type.
    if (frame.first.cols != p->_camera.w || frame.first.rows != p->_camera.h || frame.first.type() != CV_8UC3) {
        cerr << "MultiBandMap2DCPU::renderFrame: "
                "frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h||frame.first.type()!=CV_8UC3\n";
        return false;
    }

    //// 1. Compute the 3D coordinates of the image corners on the plane, and save the x and y coordinates.
    // Set imgPts to be the 4 corners of the image in pixel coordinates
    std::vector<pi::Point2d> imgPts;
    imgPts.reserve(4);
    imgPts.push_back(pi::Point2d(0, 0));
    imgPts.push_back(pi::Point2d(p->_camera.w, 0));
    imgPts.push_back(pi::Point2d(0, p->_camera.h));
    imgPts.push_back(pi::Point2d(p->_camera.w, p->_camera.h));
    // Set pts to be the projection of the 4 corners onto the plane
    vector<pi::Point2d> pts;
    pts.reserve(imgPts.size());
    pi::Point3d downLook(0, 0, -1);
    if (frame.second.get_translation().z < 0)
        downLook = pi::Point3d(0, 0, 1);
    for (int i = 0; i < imgPts.size(); i++) {
        // Compute axis as p_P'^(corner) = R_P^D * p_D^(corner), the image corner point (with homogeneous scale = 1) in
        // the orientation of the plane frame (P' has the same orientation as the plane frame P).
        pi::Point3d axis = frame.second.get_rotation() * p->UnProject(imgPts[i]);
        // Ensure the drone pose faces the plane
        if (axis.dot(downLook) < 0.4) {
            return false;
        }
        // Project to plane: p_P^(corner) = p_P^D - p_P'^(corner) * (z_P^D / z_P'^(corner))
        axis = frame.second.get_translation() - axis * (frame.second.get_translation().z / axis.z);
        // Save the x and y coordinates of the projected point on the plane
        pts.push_back(pi::Point2d(axis.x, axis.y));
    }

    //// 2. Increase the size of the map if necessary, and compute the indices (and coordinates) of the corner points.
    // Compute the min/max 2D coordinates from the 4 projected corner points on the plane
    double xmin = pts[0].x;
    double xmax = xmin;
    double ymin = pts[0].y;
    double ymax = ymin;
    for (int i = 1; i < pts.size(); i++) {
        if (pts[i].x < xmin)
            xmin = pts[i].x;
        if (pts[i].y < ymin)
            ymin = pts[i].y;
        if (pts[i].x > xmax)
            xmax = pts[i].x;
        if (pts[i].y > ymax)
            ymax = pts[i].y;
    }
    // If the points lie outside of the current bounds (defined in data), expand the size of the map (data)
    if (xmin < d->min().x || xmax > d->max().x || ymin < d->min().y || ymax > d->max().y) {
        // Make sure p still points to the same prepared data. This should only fail if MultiBandMap2DCPU::prepare has
        // been called again, but this only occurs once at the start before this function is called.
        if (p != prepared) {
            return false;
        }
        // Increase the map bounds, exiting if this fails.
        if (!spreadMap(xmin, ymin, xmax, ymax)) {
            return false;
        } else {
            pi::ReadMutex lock(mutex);
            if (p != prepared) {
                return false;
            }
            // Update the data pointer because spread map creates a new data object, and thus d points to the old map.
            d = data;
        }
    }
    // Compute the indices of the elements (in the grid/data) at the corners
    int xminInt = floor((xmin - d->min().x) * d->eleSizeInv());
    int yminInt = floor((ymin - d->min().y) * d->eleSizeInv());
    int xmaxInt = ceil((xmax - d->min().x) * d->eleSizeInv());
    int ymaxInt = ceil((ymax - d->min().y) * d->eleSizeInv());
    // Complain if the indices are out of bounds, which should never happen because the map was resized
    if (xminInt < 0 || yminInt < 0 || xmaxInt > d->w() || ymaxInt > d->h() || xminInt >= xmaxInt ||
            yminInt >= ymaxInt) {
        cerr << "MultiBandMap2DCPU::renderFrame:should never happen!\n";
        return false;
    }
    // Recompute the min/max coordinates from the element indices
    xmin = d->min().x + d->eleSize() * xminInt;
    ymin = d->min().y + d->eleSize() * yminInt;
    xmax = d->min().x + d->eleSize() * xmaxInt;
    ymax = d->min().y + d->eleSize() * ymaxInt;

    //// 3. Prepare weight image (once only)
    // The weightImage is created once (if frames remain the same size), and copied into weight_src.
    cv::Mat weight_src;
    if (weightImage.empty() || weightImage.cols != frame.first.cols || weightImage.rows != frame.first.rows) {
        // Create an image of floats with the same dimensions as the image.
        pi::WriteMutex lock(mutex);
        int w = frame.first.cols;
        int h = frame.first.rows;
        weightImage.create(h, w, CV_32FC1);
        float *p = (float *)weightImage.data;
        // Compute the weight for each pixel as a circular pattern, decaying linearly (by default) from 1 in the centre
        // to 0 at the corners, but with a minimum of 1e-5.
        float x_center = w / 2;
        float y_center = h / 2;
        float dis_max = sqrt(x_center * x_center + y_center * y_center);
        int weightType = svar.GetInt("Map2D.WeightType", 0);
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                float dis = (i - y_center) * (i - y_center) + (j - x_center) * (j - x_center);
                dis = 1 - sqrt(dis) / dis_max;
                if (0 == weightType)
                    *p = dis;
                else
                    *p = dis * dis;
                if (*p <= 1e-5)
                    *p = 1e-5;
                p++;
            }
        weight_src = weightImage.clone();
    } else {
        pi::ReadMutex lock(mutex);
        weight_src = weightImage.clone();
    }

    //// 4. Warp images
    // Copy imgPts to imgPtsCV (2D image coordinates), changing from type pi::Point2d to cv::Point2f
    std::vector<cv::Point2f> imgPtsCV;
    imgPtsCV.reserve(imgPts.size());
    for (int i = 0; i < imgPts.size(); i++) {
        imgPtsCV.push_back(cv::Point2f(imgPts[i].x, imgPts[i].y));
    }
    // Compute destPoints (2D image coordinates) from the 3D coordinates (with z = 0) of the projected corners (pts).
    // The plane is treated as an image, and the x, y 3D coordinates are converted to pixel coordinates by shifting by
    // the min x,y coordinates and then dividing by the length of a pixel (in metres).
    std::vector<cv::Point2f> destPoints;
    destPoints.reserve(imgPtsCV.size());
    for (int i = 0; i < imgPtsCV.size(); i++) {
        destPoints.push_back(
                cv::Point2f((pts[i].x - xmin) * d->lengthPixelInv(), (pts[i].y - ymin) * d->lengthPixelInv()));
    }

    // Compute the warp from the original 2D image corner coordinates to the 2D image coordinates on the plane.
    cv::Mat transmtx = cv::getPerspectiveTransform(imgPtsCV, destPoints);

    // Convert image to 3-channel float (if MultiBandMap2DCPU.ForceFloat == 0) or 3-channel 2-byte signed int.
    cv::Mat img_src;
    if (svar.GetInt("MultiBandMap2DCPU.ForceFloat", 0)) {
        frame.first.convertTo(img_src, CV_32FC3, 1. / 255.);
    } else {
        frame.first.convertTo(img_src, CV_16SC3);
    }

    // Determine the size (in pixels) of warped weight image and RGB image as the number of grid elements between max
    // and min multiplied by the number of pixels per element.
    cv::Mat image_warped((ymaxInt - yminInt) * ELE_s is donePIXELS, (xmaxInt - xminInt) * ELE_PIXELS, img_src.type());
    cv::Mat weight_warped((ymaxInt - yminInt) * ELE_PIXELS, (xmaxInt - xminInt) * ELE_PIXELS, CV_32FC1);

    // Apply the warp to the RGB image and weight image. For the RGB image, use linear interpolation and reflect at the
    // borders. For the weight image, interpolate to the nearest pixel and use default constant border.
    cv::warpPerspective(img_src, image_warped, transmtx, image_warped.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    cv::warpPerspective(weight_src, weight_warped, transmtx, weight_warped.size(), cv::INTER_NEAREST);

    // Display/save the warped images if configured to, waiting until key is pressed.
    if (svar.GetInt("ShowWarped", 0)) {
        cv::imshow("image_warped", image_warped);
        cv::imshow("weight_warped", weight_warped);
        if (svar.GetInt("SaveImageWarped")) {
            cout << "Saving warped image.\n";
            cv::imwrite("image_warped.png", image_warped);
            cv::imwrite("weight_warped.png", weight_warped);
        }
        // Wait forever until key is pressed
        cv::waitKey(0);
    }

    //// 5. Perform the blending of the warped image with the grid (data)
    // Create an N-level Laplacian pyramid of the RGB image where N equals the number of bands K
    std::vector<cv::Mat> pyr_laplace;
    cv::detail::createLaplacePyr(image_warped, _bandNum, pyr_laplace);

    // Create the corresponding weight pyramid, by creating K + 1 pyramidal weight images.
    std::vector<cv::Mat> pyr_weights(_bandNum + 1);
    pyr_weights[0] = weight_warped;
    for (int i = 0; i < _bandNum; ++i) {
        // Apply Gaussian kernel on level i of pyramid and downsample it by rejecting even rows and columns
        cv::pyrDown(pyr_weights[i], pyr_weights[i + 1]);
    }

    pi::timer.enter("MultiBandMap2DCPU::Apply");
    // Iterate over all cells in the maximal rectangular region where image projects onto the plane
    std::vector<SPtr<MultiBandMap2DCPUEle>> dataCopy = d->data();
    for (int x = xminInt; x < xmaxInt; x++)
        for (int y = yminInt; y < ymaxInt; y++) {
            // Get the cell/element patch
            SPtr<MultiBandMap2DCPUEle> ele = dataCopy[y * d->w() + x];
            if (!ele.get()) {
                ele = d->ele(y * d->w() + x);
            }

            // Initialise the element if it hasn't been initialised yet (set size of laplacian pyramid and weights)
            pi::WriteMutex lock(ele->mutexData);
            if (!ele->pyr_laplace.size()) {
                ele->pyr_laplace.resize(_bandNum + 1);
                ele->weights.resize(_bandNum + 1);
            }

            // Iterate over the Laplacian pyramid levels. Start with width/height equal to the patch size (256x256) and
            // halve this size at each level
            int width = ELE_PIXELS, height = ELE_PIXELS;
            for (int i = 0; i <= _bandNum; ++i) {
                if (ele->pyr_laplace[i].empty()) {
                    //// Case 1: Element is new (laplacian pyramid not set)
                    // Create rectangular region of size width*height (256/2^i * 256/2^i) centered on current element
                    cv::Rect rect(width * (x - xminInt), height * (y - yminInt), width, height);
                    // Copy that region of the warped image's laplacian pyramid and warped weight image to element's
                    // laplacian pyramid and weights
                    pyr_laplace[i](rect).copyTo(ele->pyr_laplace[i]);
                    pyr_weights[i](rect).copyTo(ele->weights[i]);
                } else {
                    //// Case 2: Element is not new and blending is required

                    if (pyr_laplace[i].type() == CV_32FC3) {
                        //// Blending procedure for 3-channel float image

                        // org = pixel index of the element origin in the current laplacian pyramid and weights image
                        int org = (x - xminInt) * width + (y - yminInt) * height * pyr_laplace[i].cols;
                        // skip = num pixels to skip when iterating over current laplacian pyramid and weights image
                        int skip = pyr_laplace[i].cols - ele->pyr_laplace[i].cols;

                        // srcL = pointer to a pixel in the current laplacian pyramid image, starting at element origin
                        pi::Point3f *srcL = ((pi::Point3f *)pyr_laplace[i].data) + org;
                        // srcW = pointer to a pixel in the current weights image, starting at element origin
                        float *srcW = ((float *)pyr_weights[i].data) + org;

                        // dstL = pointer to a pixel in the element's current laplacian pyramid image
                        pi::Point3f *dstL = (pi::Point3f *)ele->pyr_laplace[i].data;
                        // dstW = pointer to a pixel in the element's current weights image
                        float *dstW = (float *)ele->weights[i].data;

                        // Iterate over every pixel in the patch (size depends on level, (256/2^i, 256/2^i))
                        for (int eleY = 0; eleY < height; eleY++, srcL += skip, srcW += skip) {
                            for (int eleX = 0; eleX < width; eleX++, srcL++, dstL++, srcW++, dstW++) {
                                // If the weight is higher in the (new) image for this pixel than saved in the element,
                                // then update the laplacian pyramid pixel value and weight value in the element.
                                if ((*srcW) >= (*dstW)) {
                                    *dstL = (*srcL);
                                    *dstW = *srcW;
                                }
                            }
                        }
                    } else if (pyr_laplace[i].type() == CV_16SC3) {
                        //// Same procedure as 3-channel float image but for 3-channel 2-byte signed integers

                        int org = (x - xminInt) * width + (y - yminInt) * height * pyr_laplace[i].cols;
                        int skip = pyr_laplace[i].cols - ele->pyr_laplace[i].cols;

                        pi::Point3_<short> *srcL = ((pi::Point3_<short> *)pyr_laplace[i].data) + org;
                        float *srcW = ((float *)pyr_weights[i].data) + org;

                        pi::Point3_<short> *dstL = (pi::Point3_<short> *)ele->pyr_laplace[i].data;
                        float *dstW = (float *)ele->weights[i].data;

                        for (int eleY = 0; eleY < height; eleY++, srcL += skip, srcW += skip)
                            for (int eleX = 0; eleX < width; eleX++, srcL++, dstL++, srcW++, dstW++) {
                                if ((*srcW) >= (*dstW)) {
                                    *dstL = (*srcL);
                                    *dstW = *srcW;
                                }
                            }
                    }
                }
                // Halve size at each level
                width /= 2;
                height /= 2;
            }
            // Set a flag that the element has changed
            ele->Ischanged = true;
        }
    pi::timer.leave("MultiBandMap2DCPU::Apply");

    return true;
}

/**
 * @brief Increase bounds of the map
 *
 * @param xmin Minimum x corrdinated from the 4 projected corner points on the plane
 * @param ymin Minimum y corrdinated from the 4 projected corner points on the plane
 * @param xmax Maximum x corrdinated from the 4 projected corner points on the plane
 * @param ymax Maximum y corrdinated from the 4 projected corner points on the plane
 * @return true ??? It always succeeds
 * @return false
 */
bool MultiBandMap2DCPU::spreadMap(double xmin, double ymin, double xmax, double ymax) {
    // Start spreadMap timer
    pi::timer.enter("MultiBandMap2DCPU::spreadMap");

    // Get the grid data (d). Note that the mutex is used incorrectly. A (shared) pointer is acquired but
    // the data can still be read/written by this and any other thread once the mutex goes out of scope.
    SPtr<MultiBandMap2DCPUData> d;
    {
        pi::ReadMutex lock(mutex);
        d = data;
    }

    // Compute the indices of the elements (in the grid/data) at the corners
    int xminInt = floor((xmin - d->min().x) * d->eleSizeInv());
    int yminInt = floor((ymin - d->min().y) * d->eleSizeInv());
    int xmaxInt = ceil((xmax - d->min().x) * d->eleSizeInv());
    int ymaxInt = ceil((ymax - d->min().y) * d->eleSizeInv());

    // Determing the increase in width or height required for the map
    xminInt = min(xminInt, 0);
    yminInt = min(yminInt, 0);
    xmaxInt = max(xmaxInt, d->w());
    ymaxInt = max(ymaxInt, d->h());
    int w = xmaxInt - xminInt;
    int h = ymaxInt - yminInt;

    // Compute new dimensions for the map
    pi::Point2d min, max;
    {
        min.x = d->min().x + d->eleSize() * xminInt;
        min.y = d->min().y + d->eleSize() * yminInt;
        max.x = min.x + w * d->eleSize();
        max.y = min.y + h * d->eleSize();
    }

    // Copy old Ele vector to newly resized vector according to new map dimensions
    std::vector<SPtr<MultiBandMap2DCPUEle>> dataOld = d->data();
    std::vector<SPtr<MultiBandMap2DCPUEle>> dataCopy;
    dataCopy.resize(w * h);
    for (int x = 0, xend = d->w(); x < xend; x++) {
        for (int y = 0, yend = d->h(); y < yend; y++) {
            dataCopy[x - xminInt + (y - yminInt) * w] = dataOld[y * d->w() + x];
        }
    }

    // Modify the grid data with the new dimensions while copying the old ELe vector.
    {
        pi::WriteMutex lock(mutex);
        data = SPtr<MultiBandMap2DCPUData>(new MultiBandMap2DCPUData(d->eleSize(), d->lengthPixel(),
                pi::Point3d(max.x, max.y, d->max().z), pi::Point3d(min.x, min.y, d->min().z), w, h, dataCopy));
    }

    // Stop spreadMap timer
    pi::timer.leave("MultiBandMap2DCPU::spreadMap");
    return true;
}

/**
 * @brief Extract frame from queue
 *
 * @param frame Paramter passed as reference where the frame extracted is returned
 * @return true If frame exists
 * @return false If queue is empty
 */
bool MultiBandMap2DCPU::getFrame(std::pair<cv::Mat, pi::SE3d> &frame) {
    // Take Mutexes
    pi::ReadMutex lock(mutex);
    pi::ReadMutex lock1(prepared->mutexFrames);

    // Check if there are frames to be processed
    if (prepared->_frames.size()) {
        frame = prepared->_frames.front();  // Get frame from head of the queue
        prepared->_frames.pop_front();      // Remove frame from the head of the queue
        return true;
    } else {
        return false;  // No frames to be processed
    }
}

/**
 * @brief Thread run function.
 *
 */
void MultiBandMap2DCPU::run() {
    std::pair<cv::Mat, pi::SE3d> frame;

    // Check stopping condition
    while (!shouldStop()) {
        // Check if MultiBandMap2DCPU was prepared
        if (_valid) {
            // Get new frame from queue
            if (getFrame(frame)) {
                // Start renderFrame timer
                pi::timer.enter("MultiBandMap2DCPU::renderFrame");
                // Render the newly extracted frame from the queue
                renderFrame(frame);
                // Stop renderFrame timer
                pi::timer.leave("MultiBandMap2DCPU::renderFrame");
            }
        }
        sleep(10);
    }
}

/**
 * @brief Draw the OpenGL map based on Ele texture (update texture if needed)
 *
 */
void MultiBandMap2DCPU::draw() {
    // Check if MultiBandMap2DCPU was prepared
    if (!_valid)
        return;

    // Get the prepared frames (p) and grid data (d). Note that the mutex is used incorrectly. A (shared) pointer is
    // acquired but the data can still be read/written by this and any other thread once the mutex goes out of scope.
    SPtr<MultiBandMap2DCPUPrepare> p;
    SPtr<MultiBandMap2DCPUData> d;
    {
        pi::ReadMutex lock(mutex);
        p = prepared;
        d = data;
    }

    // Use model view matrix
    glMatrixMode(GL_MODELVIEW);
    // Push new matrix to the top of the stack (Identical to the one below)
    glPushMatrix();
    // Multiply current matrix with the plane
    glMultMatrix(p->_plane);

    //// draw deque frames
    // Draw the pose for each frame
    pi::TicTac ticTac;
    ticTac.Tic();
    {
        // Get all frames
        std::deque<std::pair<cv::Mat, pi::SE3d>> frames = p->getFrames();
        glDisable(GL_LIGHTING);
        glBegin(GL_LINES);

        // For each frame draw the lines for the pose. (XYZ)
        for (std::deque<std::pair<cv::Mat, pi::SE3d>>::iterator it = frames.begin(); it != frames.end(); it++) {
            pi::SE3d &pose = it->second;
            glColor3ub(255, 0, 0);
            glVertex(pose.get_translation());
            glVertex(pose * pi::Point3d(1, 0, 0));
            glColor3ub(0, 255, 0);
            glVertex(pose.get_translation());
            glVertex(pose * pi::Point3d(0, 1, 0));
            glColor3ub(0, 0, 255);
            glVertex(pose.get_translation());
            glVertex(pose * pi::Point3d(0, 0, 1));
        }
        glEnd();
    }

    //// draw global area
    // Draw the border of the map if Map2D.DrawArea is true
    if (svar.GetInt("Map2D.DrawArea")) {
        pi::Point3d _min = d->min();
        pi::Point3d _max = d->max();
        glColor3ub(255, 0, 0);
        glBegin(GL_LINES);
        glVertex3d(_min.x, _min.y, 0);
        glVertex3d(_min.x, _max.y, 0);
        glVertex3d(_min.x, _min.y, 0);
        glVertex3d(_max.x, _min.y, 0);
        glVertex3d(_max.x, _min.y, 0);
        glVertex3d(_max.x, _max.y, 0);
        glVertex3d(_min.x, _max.y, 0);
        glVertex3d(_max.x, _max.y, 0);
        glEnd();
    }

    //// draw textures
    // Enable 2D texturing and blending
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    //    glEnable(GL_LIGHTING);

    // If Map2D.Alpha is set, use Alpha values for blending
    if (alpha) {
        glEnable(GL_ALPHA_TEST);
        glAlphaFunc(GL_GREATER, 0.1f);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    }

    // Get the last binded texture (main texture)
    GLint last_texture_ID;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture_ID);

    // Get Ele vector and map dimensions
    std::vector<SPtr<MultiBandMap2DCPUEle>> dataCopy = d->data();
    int wCopy = d->w(), hCopy = d->h();
    glColor3ub(255, 255, 255);

    // Iterate through every Ele
    for (int x = 0; x < wCopy; x++) {
        for (int y = 0; y < hCopy; y++) {
            // Determine Ele index in vector and its corner coordinates
            int idxData = y * wCopy + x;
            float x0 = d->min().x + x * d->eleSize();
            float y0 = d->min().y + y * d->eleSize();
            float x1 = x0 + d->eleSize();
            float y1 = y0 + d->eleSize();

            // Get Ele from vector
            SPtr<MultiBandMap2DCPUEle> ele = dataCopy[idxData];

            // If Ele does not exist, continue
            if (!ele.get()) {
                continue;
            }

            {
                // Aquire Ele mutex
                pi::ReadMutex lock(ele->mutexData);

                // If one of Laplace Pyramid and Weights matrix have 0 size
                // or they are not equal, then continue
                if (!(ele->pyr_laplace.size() && ele->weights.size() &&
                            ele->pyr_laplace.size() == ele->weights.size())) {
                    continue;
                }

                // Check if Ele has changed during a renderFrame call
                if (ele->Ischanged) {
                    // Start updateTexture timer
                    pi::timer.enter("MultiBandMap2DCPU::updateTexture");
                    bool updated = false, inborder = false;

                    // If high quality map then update texture (blend) with neighbors
                    if (_highQualityShow) {
                        // Get neighbors of the Ele
                        vector<SPtr<MultiBandMap2DCPUEle>> neighbors;
                        neighbors.reserve(9);
                        for (int yi = y - 1; yi <= y + 1; yi++) {
                            for (int xi = x - 1; xi <= x + 1; xi++) {
                                if (yi < 0 || yi >= hCopy || xi < 0 || xi >= wCopy) {
                                    neighbors.push_back(SPtr<MultiBandMap2DCPUEle>());
                                    inborder = true;
                                } else
                                    neighbors.push_back(dataCopy[yi * wCopy + xi]);
                            }
                        }
                        // Update texture with neighbors
                        updated = ele->updateTexture(neighbors);
                    } else {
                        // Update texture without neighbors
                        updated = ele->updateTexture();
                    }
                    // Stop updateTexture timer
                    pi::timer.leave("MultiBandMap2DCPU::updateTexture");

                    // Check if texture update succeeded, there were no neighbors on the
                    // border (if high quality map) and Fuse2Google is true
                    if (updated && !inborder && svar.GetInt("Fuse2Google")) {
                        // Start fuseGoogle timer
                        pi::timer.enter("MultiBandMap2DCPU::fuseGoogle");
                        stringstream cmd;

                        // Calculate world coodintaes for Ele corners (Tl = Top left, Br = Bottom right)
                        pi::Point3d worldTl = p->_plane * pi::Point3d(x0, y0, 0);
                        pi::Point3d worldBr = p->_plane * pi::Point3d(x1, y1, 0);

                        // Calculate gps coordinates for the Ele corners
                        pi::Point3d gpsTl, gpsBr;
                        pi::calcLngLatFromDistance(d->gpsOrigin().x, d->gpsOrigin().y, worldTl.x, worldTl.y, gpsTl.x,
                                gpsTl.y);
                        pi::calcLngLatFromDistance(d->gpsOrigin().x, d->gpsOrigin().y, worldBr.x, worldBr.y, gpsBr.x,
                                gpsBr.y);
                        // cout<<"world:"<<worldBr<<"origin:"<<d->gpsOrigin()<<endl;

                        // Create update command for the MapWidget with gps coordinates.
                        cmd << "Map2DUpdate LastTexMat " << setiosflags(ios::fixed) << setprecision(9) << gpsTl << " "
                            << gpsBr;
                        // cout<<cmd.str()<<endl;

                        // Execute command
                        scommand.Call("MapWidget", cmd.str());

                        // Stop fuseGoogle timer
                        pi::timer.leave("MultiBandMap2DCPU::fuseGoogle");
                    }
                }
            }

            // Check if Ele has texture
            if (ele->texName) {
                // Apply texture to QUAD
                glBindTexture(GL_TEXTURE_2D, ele->texName);
                // Create QUAD with Ele coordinates
                glBegin(GL_QUADS);
                glTexCoord2f(0.0f, 0.0f);
                glVertex3f(x0, y0, 0);
                glTexCoord2f(0.0f, 1.0f);
                glVertex3f(x0, y1, 0);
                glTexCoord2f(1.0f, 1.0f);
                glVertex3f(x1, y1, 0);
                glTexCoord2f(1.0f, 0.0f);
                glVertex3f(x1, y0, 0);
                glEnd();
            }
        }
    }
    // Revert to last binded texture (main texture)
    glBindTexture(GL_TEXTURE_2D, last_texture_ID);
    // Pop the previously pushed matrix
    glPopMatrix();
}

bool MultiBandMap2DCPU::save(const std::string &filename) {
    //// determin minmax
    // Get the prepared frames (p) and grid data (d). Note that the mutex is used incorrectly. A (shared) pointer is
    // acquired but the data can still be read/written by this and any other thread once the mutex goes out of scope.
    SPtr<MultiBandMap2DCPUPrepare> p;
    SPtr<MultiBandMap2DCPUData> d;
    {
        pi::ReadMutex lock(mutex);
        p = prepared;
        d = data;
    }

    // Check Ele matrix dimmensions are non-zero
    if (d->w() == 0 || d->h() == 0) {
        return false;
    }

    // Determine the minimum and maximum size
    // Does not take into consideration unused Eles
    pi::Point2i minInt(1e6, 1e6), maxInt(-1e6, -1e6);
    int contentCount = 0;
    for (int x = 0; x < d->w(); x++) {
        for (int y = 0; y < d->h(); y++) {
            SPtr<MultiBandMap2DCPUEle> ele = d->data()[x + y * d->w()];
            if (!ele.get()) {
                continue;
            }

            {
                pi::ReadMutex lock(ele->mutexData);
                if (!ele->pyr_laplace.size()) {
                    continue;
                }
            }

            contentCount++;
            minInt.x = min(minInt.x, x);
            minInt.y = min(minInt.y, y);
            maxInt.x = max(maxInt.x, x);
            maxInt.y = max(maxInt.y, y);
        }
    }

    // Determine w*h dimmension based on used Eles
    maxInt = maxInt + pi::Point2i(1, 1);
    pi::Point2i wh = maxInt - minInt;

    // Initialize final Laplace pyramids and weights
    vector<cv::Mat> pyr_laplace(_bandNum + 1);
    // vector<cv::Mat> pyr_weights(_bandNum + 1);
    // for (int i = 0; i <= 0; i++) pyr_weights[i] = cv::Mat::zeros(wh.y * ELE_PIXELS, wh.x * ELE_PIXELS, CV_32FC1);
    cv::Mat pyr_weights = cv::Mat::zeros(wh.y * ELE_PIXELS, wh.x * ELE_PIXELS, CV_32FC1);

    for (int x = minInt.x; x < maxInt.x; x++) {
        for (int y = minInt.y; y < maxInt.y; y++) {
            SPtr<MultiBandMap2DCPUEle> ele = d->data()[x + y * d->w()];
            if (!ele.get()) {
                continue;
            }

            {
                pi::ReadMutex lock(ele->mutexData);
                if (!ele->pyr_laplace.size()) {
                    continue;
                }
                int width = ELE_PIXELS, height = ELE_PIXELS;

                for (int i = 0; i <= _bandNum; ++i) {
                    cv::Rect rect(width * (x - minInt.x), height * (y - minInt.y), width, height);
                    if (pyr_laplace[i].empty()) {
                        pyr_laplace[i] = cv::Mat::zeros(wh.y * height, wh.x * width, ele->pyr_laplace[i].type());
                    }
                    ele->pyr_laplace[i].copyTo(pyr_laplace[i](rect));
                    if (i == 0) {
                        // ele->weights[i].copyTo(pyr_weights[i](rect));
                        ele->weights[i].copyTo(pyr_weights(rect));
                    }
                    height >>= 1;
                    width >>= 1;
                }
            }
        }
    }

    cv::detail::restoreImageFromLaplacePyr(pyr_laplace);

    cv::Mat result = pyr_laplace[0];
    if (result.type() == CV_16SC3) {
        result.convertTo(result, CV_8UC3);
    }
    // result.setTo(cv::Scalar::all(svar.GetInt("Result.BackGroundColor")), pyr_weights[0] == 0);
    result.setTo(cv::Scalar::all(svar.GetInt("Result.BackGroundColor")), pyr_weights == 0);
    cv::imwrite(filename, result);
    cout << "Resolution:[" << result.cols << " " << result.rows << "]";
    if (svar.exist("GPS.Origin")) {
        cout << ",_lengthPixel:" << d->lengthPixel() << ",Area:" << contentCount * d->eleSize() * d->eleSize() << endl;
    }
    return true;
}
