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
#ifndef MultiBandMap2DCPU_H
#define MultiBandMap2DCPU_H
#include <base/system/thread/ThreadBase.h>

#include "Map2D.h"

class MultiBandMap2DCPU : public Map2D, public pi::Thread {
    typedef Map2DPrepare MultiBandMap2DCPUPrepare;

    struct MultiBandMap2DCPUEle {
        // Constructor and destructor 
        // Constructor: initializes structure variable with 0 texture and Ischanged set to false
        MultiBandMap2DCPUEle()
            : texName(0),
              Ischanged(false) {}
        ~MultiBandMap2DCPUEle();

        static bool normalizeUsingWeightMap(const cv::Mat &weight, cv::Mat &src);
        static bool mulWeightMap(const cv::Mat &weight, cv::Mat &src);

        cv::Mat blend(
                const std::vector<SPtr<MultiBandMap2DCPUEle>> &neighbors = std::vector<SPtr<MultiBandMap2DCPUEle>>());
        bool updateTexture(
                const std::vector<SPtr<MultiBandMap2DCPUEle>> &neighbors = std::vector<SPtr<MultiBandMap2DCPUEle>>());

        std::vector<cv::Mat> pyr_laplace;
        std::vector<cv::Mat> weights;

        uint texName;
        bool Ischanged;
        pi::MutexRW mutexData;
    };

    struct MultiBandMap2DCPUData  // change when spread and prepare
    {
        MultiBandMap2DCPUData()
            : _w(0),
              _h(0) {}
        MultiBandMap2DCPUData(double eleSize_, double lengthPixel_, pi::Point3d max_, pi::Point3d min_, int w_, int h_,
                const std::vector<SPtr<MultiBandMap2DCPUEle>> &d_);

        bool prepare(SPtr<MultiBandMap2DCPUPrepare> prepared);  // only done Once!

        double eleSize() const {
            return _eleSize;
        }
        double lengthPixel() const {
            return _lengthPixel;
        }
        double eleSizeInv() const {
            return _eleSizeInv;
        }
        double lengthPixelInv() const {
            return _lengthPixelInv;
        }
        const pi::Point3d &gpsOrigin() const {
            return _gpsOrigin;
        }
        const pi::Point3d &min() const {
            return _min;
        }
        const pi::Point3d &max() const {
            return _max;
        }
        const int w() const {
            return _w;
        }
        const int h() const {
            return _h;
        }

        std::vector<SPtr<MultiBandMap2DCPUEle>> data() {
            pi::ReadMutex lock(mutexData);
            return _data;
        }

        SPtr<MultiBandMap2DCPUEle> ele(uint idx) {
            pi::WriteMutex lock(mutexData);
            if (idx > _data.size())
                return SPtr<MultiBandMap2DCPUEle>();
            else if (!_data[idx].get()) {
                _data[idx] = SPtr<MultiBandMap2DCPUEle>(new MultiBandMap2DCPUEle());
            }
            return _data[idx];
        }

    private:
        // IMPORTANT: everything should never changed after prepared!

        // Size of a grid element in metres, such that max - min = eleSize * (width or height)
        double _eleSize, _eleSizeInv;
        // Size of each pixel as eleSize/ELE_PIXELS where ELE_PIXELS = 256. There are 256 pixels per grid element.
        double _lengthPixel, _lengthPixelInv;
        // GPS origin
        pi::Point3d _gpsOrigin;
        // 3D coordinates for min and max bounds of the grid
        pi::Point3d _max, _min;
        // Width and height of _data
        int _w, _h;
        // Element of each grid cell, of size width * height
        std::vector<SPtr<MultiBandMap2DCPUEle>> _data;
        // Mutex for safe multi-threaded access
        pi::MutexRW mutexData;
    };

public:
    MultiBandMap2DCPU(bool thread = true);

    virtual ~MultiBandMap2DCPU() {
        _valid = false;
    }

    virtual bool prepare(const pi::SE3d &plane, const PinHoleParameters &camera,
            const std::deque<CameraFrame> &frames);

    virtual bool feed(cv::Mat img, cv::Mat sem, const pi::SE3d &pose);  // world coordinate

    virtual void draw();

    virtual bool save(const std::string &filename, const std::string &semFilename);

    virtual uint queueSize() {
        if (prepared.get())
            return prepared->queueSize();
        else
            return 0;
    }

    virtual void run();

private:
    bool getFrame(CameraFrame &frame);
    bool renderFrame(const CameraFrame &frame);
    bool spreadMap(double xmin, double ymin, double xmax, double ymax);

    // source
    SPtr<MultiBandMap2DCPUPrepare> prepared;
    SPtr<MultiBandMap2DCPUData> data;
    pi::MutexRW mutex;

    bool _valid, _thread, _changed;
    cv::Mat weightImage;
    int &alpha, _bandNum, &_highQualityShow;
};
#endif  // MULTIBANDMap2DCPU_H
