#ifndef MultiBandMap2DCPUSEM_H
#define MultiBandMap2DCPUSEM_H
#include <base/system/thread/ThreadBase.h>
#include <unordered_map>
#include <cstdint>

#include "Map2D.h"

struct SemLabel {
    std::string name;
    pi::Point3ub colour;
    int priority;
};

class MultiBandMap2DCPUSem : public Map2D, public pi::Thread {
    typedef Map2DPrepare MultiBandMap2DCPUSemPrepare;

    struct MultiBandMap2DCPUSemEle {
        // Constructor and destructor
        // Constructor: initializes structure variable with 0 texture and Ischanged set to false
        MultiBandMap2DCPUSemEle()
            : texName(0),
              Ischanged(false) {}
        ~MultiBandMap2DCPUSemEle();

        static bool normalizeUsingWeightMap(const cv::Mat &weight, cv::Mat &src);
        static bool mulWeightMap(const cv::Mat &weight, cv::Mat &src);

        cv::Mat blend(const std::vector<SPtr<MultiBandMap2DCPUSemEle>> &neighbors =
                              std::vector<SPtr<MultiBandMap2DCPUSemEle>>());
        bool updateTexture(const std::vector<SPtr<MultiBandMap2DCPUSemEle>> &neighbors =
                                   std::vector<SPtr<MultiBandMap2DCPUSemEle>>());

        std::vector<cv::Mat> pyr_laplace;
        std::vector<cv::Mat> weights;
        std::vector<cv::Mat> pyr_sem;
        std::unordered_map<std::uint32_t, std::vector<cv::Mat>> class_scores;

        uint texName;
        bool Ischanged;
        pi::MutexRW mutexData;
    };

    struct MultiBandMap2DCPUSemData  // change when spread and prepare
    {
        MultiBandMap2DCPUSemData()
            : _w(0),
              _h(0) {}
        MultiBandMap2DCPUSemData(double eleSize_, double lengthPixel_, pi::Point3d max_, pi::Point3d min_, int w_,
                int h_, const std::vector<SPtr<MultiBandMap2DCPUSemEle>> &d_);

        bool prepare(SPtr<MultiBandMap2DCPUSemPrepare> prepared);  // only done Once!

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

        std::vector<SPtr<MultiBandMap2DCPUSemEle>> data() {
            pi::ReadMutex lock(mutexData);
            return _data;
        }

        SPtr<MultiBandMap2DCPUSemEle> ele(uint idx) {
            pi::WriteMutex lock(mutexData);
            if (idx > _data.size())
                return SPtr<MultiBandMap2DCPUSemEle>();
            else if (!_data[idx].get()) {
                _data[idx] = SPtr<MultiBandMap2DCPUSemEle>(new MultiBandMap2DCPUSemEle());
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
        std::vector<SPtr<MultiBandMap2DCPUSemEle>> _data;
        // Mutex for safe multi-threaded access
        pi::MutexRW mutexData;
    };

public:
    MultiBandMap2DCPUSem(bool thread = true);

    virtual ~MultiBandMap2DCPUSem() {
        _valid = false;
    }

    virtual bool prepare(const pi::SE3d &plane, const PinHoleParameters &camera, const std::deque<CameraFrame> &frames);

    virtual bool feed(cv::Mat img, cv::Mat sem, const pi::SE3d &pose);  // world coordinate

    virtual void draw();

    virtual bool save(const std::string &filename, const std::string &semFilename);

    inline void setSemanticLabels(const std::vector<SemLabel> &labels_) {
        labels = labels_;
        for (const auto& label : labels) {
            pixelid_to_label.emplace(pixelid(label.colour), label);
        }
    }

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

    std::uint32_t pixelid(const pi::Point3ub& pixel) {
        std::uint32_t id{0};
        for (int i = 0; i < 3; ++i) {
            id = id << 8;
            id |= static_cast<std::uint32_t>(pixel[i]);
        }
        return id;
    }

    // source
    SPtr<MultiBandMap2DCPUSemPrepare> prepared;
    SPtr<MultiBandMap2DCPUSemData> data;
    pi::MutexRW mutex;

    bool _valid, _thread, _changed;
    cv::Mat weightImage;
    int &alpha, _bandNum, &_highQualityShow;
    std::vector<SemLabel> labels;
    std::unordered_map<std::uint32_t, SemLabel> pixelid_to_label;
};

#endif
