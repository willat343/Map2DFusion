#include <base/Svar/Svar.h>
#include <base/Svar/VecParament.h>
#include <base/time/Global_Timer.h>

#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "MainWindow.h"
#include "Map2D.h"

using namespace std;

class TrajectoryLengthCalculator {
public:
    TrajectoryLengthCalculator()
        : length(-1) {}
    ~TrajectoryLengthCalculator() {
        cout << "TrajectoryLength:" << length << endl;
    }

    void feed(pi::Point3d position) {
        if (length < 0) {
            length = 0;
            lastPosition = position;
        } else {
            length += (position - lastPosition).norm();
            lastPosition = position;
        }
    }

private:
    double length;
    pi::Point3d lastPosition;
};

class TestSystem : public pi::Thread, public pi::gl::EventHandle {
public:
    TestSystem() {
        // Create QT GUI main window if QT is used.
        if (svar.GetInt("Win3D.Enable", 1)) {
            mainwindow = SPtr<MainWindow>(new MainWindow(0));
        }
    }

    ~TestSystem() {
        stop(); // Stop pipeline execution

        while (this->isRunning()) 
            sleep(10); // Wait for the piepline to stop
        
        // If map exists save the output to result.png
        if (map.get()) {
            map->save(svar.GetString("Map.File2Save", "result.png")); 
        }
        
        // Destruct map and main window
        map = SPtr<Map2D>();
        mainwindow = SPtr<MainWindow>();
    }

    /**
     * @brief Handle key press for TestMap2DItem
     *          Key "I": feed new frame to the map
     *          Key "P": Pause execution
     *          Key "Esc": Stop Execution
     * @param arg QT Key Event
     * @return false 
     */
    virtual bool KeyPressHandle(void *arg) {
        QKeyEvent *e = (QKeyEvent *)arg;
        switch (e->key()) {
            // Feed new frame to the map
            case Qt::Key_I: {
                CameraFrame frame;
                // Get new frame
                if (obtainFrame(frame)) {
                    pi::timer.enter("Map2D::feed"); // Start timer
                    map->feed(frame.image, frame.pose); // Feed the new frame to the map

                    // Update main window
                    if (mainwindow.get() && tictac.Tac() > 0.033) {
                        tictac.Tic();
                        mainwindow->update();
                    }

                    pi::timer.leave("Map2D::feed"); // End timer
                }
            } break;
            // Pause Execution
            case Qt::Key_P: {
                int &pause = svar.GetInt("Pause");
                pause = !pause;
            } break;
            // Stop Execution
            case Qt::Key_Escape:
                stop();
                break;
            default:
                break;
        }
        return false;
    }

    /**
     * @brief Takes one input image at a time and uses the event listener to get user input
     * 
     * @return int Error code
     */
    int TestMap2DItem() {
        cv::Mat img = cv::imread(svar.GetString("TestMap2DItem.Image", "data/test.png"));
        if (img.empty() || !mainwindow.get()) {
            cerr << "No image or mainwindow found.!\n";
            return -1;
        }
        //        cv::imshow("img",img);
        SvarWithType<cv::Mat>::instance()["LastTexMat"] = img;

        mainwindow->getWin3D()->SetEventHandle(this);
        mainwindow->getWin3D()->setSceneRadius(1000);
        mainwindow->call("show");
        mainwindow->call(
                "MapWidget" +
                svar.GetString(" TestMap2DItem.cmd",
                        " Map2DUpdate LastTexMat 34.257287 108.888931 0 34.253234419307354 108.89463874078366 0"));
    }

    /**
     * @brief 
     * 
     * @param frame  Frame passed as reference (contains image and corresponding pose)
     * @return true  If the execution completed successfully
     * @return false If the image did not load correcly
     */
    bool obtainFrame(CameraFrame &frame) {
        // Read timestamp from line (timestamp is the name of the image)
        string line;
        if (!getline(*in, line)) { // getline reads the next line that hasn't been read yet and *in points to the trajectory file
            return false;
        }
        stringstream ifs(line);
        string imgFileName;
        ifs >> imgFileName; // Reads until the first space in trajectory.txt, which is the timestamp
        imgFileName = datapath + "/rgb/" + imgFileName + ".jpg"; // Get image file path (the image files are names by timestamps)

        pi::timer.enter("obtainFrame"); // Start image read timer
        frame.image = cv::imread(imgFileName); // Read image, frame.first corresponds to the first element of the pair
        pi::timer.leave("obtainFrame"); // End image read timer

        // Image could not be read
        if (frame.image.empty()) {
            return false;
        }
        
        ifs >> frame.pose; // Read poses for the previously read frame, the second element of frame contains the pose
        
        // Feed translation from pose to length calculator if the GPS origin coordinates are set
        if (svar.exist("GPS.Origin")) {
            if (!lengthCalculator.get()) {
                lengthCalculator = SPtr<TrajectoryLengthCalculator>(new TrajectoryLengthCalculator());
            }
            lengthCalculator->feed(frame.pose.get_translation()); // keeps track of trajectory to print total trajectory at the end
        }

        return true;
    }

    /**
     * @brief Runs the main Map2D fusion pipeline on the given dataset
     * 
     * @return int Error code
     *              "-1": Data path is not set
     *              "-2": Plane is not defined
     *              "-3": Cannot open file trajectory.txt
     *              "-4": No frames loaded
     *              "-5": Map2D object failed to create
     *              "-6": Invalid camera parameters
     */
    int testMap2D() {
        cout << "Act=TestMap2D\n";

        // Get Datapath from configuration file (Default: "")
        datapath = svar.GetString("Map2D.DataPath", "");
        if (!datapath.size()) {
            cerr << "Map2D.DataPath is not set!\n";
            return -1;
        }

        // Parse Dataset configuration file
        svar.ParseFile(datapath + "/config.cfg");
        if (!svar.exist("Plane")) {
            cerr << "Plane is not defined!\n";
            return -2;
        }

        // Open trajectory.txt file that contains the poses (Used in obtainFrame)
        if (!in.get()) {
            in = SPtr<ifstream>(new ifstream((datapath + "/trajectory.txt").c_str()));
        }
        if (!in->is_open()) {
            cerr << "Can't open file " << (datapath + "/trajectory.txt") << endl;
            return -3;
        }

        deque<CameraFrame> frames; // Deque containin frames (frame = pair<Image, Pose>)

        // Preaload the queue with #PrepareFrameNum of frames (Default: 10)
        for (int i = 0, iend = svar.GetInt("PrepareFrameNum", 10); i < iend; i++) {
            CameraFrame frame;
            if (!obtainFrame(frame)) { // Obtain frame
                break; // Break if failed to read frame
            }
            frames.push_back(frame); // Push frame to queue
        }
        cout << "Loaded " << frames.size() << " frames.\n";

        // If no frames are loaded return with error code
        if (!frames.size()) {
            cerr << "No frames were loaded!";
            return -4;
        }

        // Create Map2D based on Map2D.Type (Default: TypeGPU) and Map2D.Thread (Default: True)
        map = Map2D::create(svar.GetInt("Map2D.Type", Map2D::TypeGPU), svar.GetInt("Map2D.Thread", true));
        if (!map.get()) {
            cerr << "No map2d created!\n";
            return -5;
        }

        // Read camera intrinsics
        VecParament vecP = svar.get_var("Camera.Paraments", VecParament());
        if (vecP.size() != 6) {
            cerr << "Invalid camera parameters!\n";
            return -6;
        }

        // Prepare (Setup) the map 
        map->prepare(svar.get_var<pi::SE3d>("Plane", pi::SE3d()),
                PinHoleParameters(vecP[0], vecP[1], vecP[2], vecP[3], vecP[4], vecP[5]), frames);

        // Insert map in main window and set event handler
        if (mainwindow.get()) {
            mainwindow->getWin3D()->SetEventHandle(this);
            mainwindow->getWin3D()->insert(map);
            mainwindow->getWin3D()->setSceneRadius(1000);
            mainwindow->call("show");

            if (!svar.exist("GPS.Origin"))
                svar.i["Fuse2Google"] = 0;
            else
                svar.ParseLine("SetCurrentPosition $(GPS.Origin)");
            tictac.Tic();
        } else {
            int &needStop = svar.GetInt("ShouldStop");
            while (!needStop) sleep(20);
        }

        /**
        * If AutoFeedFrames is True (Default: True)
        * Read frames and feed them to the map
        */
        if (svar.GetInt("AutoFeedFrames", 1)) {
            pi::Rate rate(svar.GetInt("Video.fps", 100)); // Frame feed rate

            while (!shouldStop()) {
                // Only feed poses if the queue size is smaller than 2
                if (map->queueSize() < 2) {
                    CameraFrame frame;

                    //Obtain new frame
                    if (!obtainFrame(frame)) {
                        break;
                    }
                    map->feed(frame.image, frame.pose); // Feed new frame to the map
                }

                // Update main window
                if (mainwindow.get() && tictac.Tac() > 0.033) {
                    tictac.Tic();
                    mainwindow->getWin3D()->update();
                }

                rate.sleep(); // Sleep based on the given image feed rate
            }
        }
    }

    /**
     * @brief Run function that is either executed as a separate thread 
     *         or by the main thread
     * 
     */
    virtual void run() {
        string act = svar.GetString("Act", "Default"); // Determine execution type

        if (act == "TestMap2DItem") {
            TestMap2DItem();
        }
        else if (act == "TestMap2D" || act == "Default") {
            testMap2D(); // Execute the incremental Map2DFustion pipeline
        }
        else {
            cout << "No act " << act << "!\n";
        }
    }

    string datapath; // Path to the dataset
    pi::TicTac tictac; // Timer for the update of the main window
    SPtr<MainWindow> mainwindow; // Main QT window
    SPtr<ifstream> in; // Input file stream for image file names and poses
    SPtr<Map2D> map; // Map2D map
    SPtr<TrajectoryLengthCalculator> lengthCalculator; // GPS tranjectory length calculator
};

int main(int argc, char **argv) {
    svar.ParseMain(argc, argv); // Parse config file given as parameter or "Default.cfg"

    // Check if the code should run with QT Application GUI
    if (svar.GetInt("Win3D.Enable", 0)) {
        QApplication app(argc, argv); // Create QT Application Window
        TestSystem sys; // Create the Map2DFusion pipeline
        sys.start(); // Run the Map2DFusion pipeline as a thread
        return app.exec(); // Run the QT Application
    } else {
        TestSystem sys; // Create the Map2DFusion pipeline
        sys.run(); // Run the Map2DFusion pipeline as a thread
    }
    return 0;
}
