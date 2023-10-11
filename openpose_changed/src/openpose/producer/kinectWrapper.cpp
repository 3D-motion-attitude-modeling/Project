#include <openpose/producer/kinectWrapper.hpp>
#include <atomic>
#include <mutex>
#include <opencv2/imgproc/imgproc.hpp> // cv::undistort, cv::initUndistortRectifyMap
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp> // OPEN_CV_IS_4_OR_HIGHER

# define USE_FLIR_CAMERA
# define USE_Kinect_CAMERA

#ifdef OPEN_CV_IS_4_OR_HIGHER
    #include <opencv2/calib3d.hpp> // cv::initUndistortRectifyMap for OpenCV 4
#endif
#ifdef USE_Kinect_CAMERA
    #include <k4a/k4a.h>
#endif

#include <openpose/3d/cameraParameterReader.hpp>

namespace op
{
    #ifdef USE_FLIR_CAMERA

        // 获取全部相机的序列号
        std::vector<std::string> getSerialNumbers(const std::vector<k4a::device> devices,
                                                  const bool sorted)
        {
            try
            {
                // Get strSerialNumbers
                std::vector<std::string> serialNumbers(devices.size());

                for (auto i = 0u; i < serialNumbers.size(); i++)
                {
                    serialNumbers[i] = devices[i].get_serialnum();
                }

                // Sort serial numbers
                if (sorted)
                    std::sort(serialNumbers.begin(), serialNumbers.end());

                // Return result
                return serialNumbers;
            }
            catch (k4a::error &e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return {};
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return {};
            }
        }

        // 打印相机的设备信息
        // int printDeviceInfo(Spinnaker::GenApi::INodeMap &iNodeMap, const unsigned int camNum)

        // 将 Spinnaker SDK 的图像数据转换为 OpenCV 的图像对象
        cv::Mat color_to_opencv(const k4a::image &imagePtr)
        {
            cv::Mat cv_image_with_alpha(imagePtr.get_height_pixels(), imagePtr.get_width_pixels(), CV_8UC4, (void *)imagePtr.get_buffer());
            cv::Mat cv_image_no_alpha;
            cv::cvtColor(cv_image_with_alpha, cv_image_no_alpha, cv::COLOR_BGRA2BGR);
            return cv_image_no_alpha;
        }

        // k4a主从相机的配置函数    
        static k4a_device_configuration_t get_default_config()
        {
            k4a_device_configuration_t camera_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
            camera_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
            camera_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
            camera_config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED; // No need for depth during calibration
            camera_config.camera_fps = K4A_FRAMES_PER_SECOND_15;     // Don't use all USB bandwidth
            camera_config.subordinate_delay_off_master_usec = 0;     // Must be zero for master
            camera_config.synchronized_images_only = true;
            return camera_config;
        }

        // Master customizable settings ———— k4a主从相机的配置函数
        static k4a_device_configuration_t get_master_config()
        {
            k4a_device_configuration_t camera_config = get_default_config();
            camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;

            // Two depth images should be seperated by MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
            // sensor doesn't interfere with the other. To accomplish this the master depth image captures
            // (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) before the color image, and the subordinate camera captures its
            // depth image (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image. This gives us two depth
            // images centered around the color image as closely as possible.
            camera_config.depth_delay_off_color_usec = -static_cast<int32_t>(MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2);
            camera_config.synchronized_images_only = true;
            return camera_config;
        }

        // Subordinate customizable settings ———— k4a主从相机的配置函数
        static k4a_device_configuration_t get_subordinate_config()
        {
            k4a_device_configuration_t camera_config = get_default_config();
            camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;

            // Two depth images should be seperated by MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
            // sensor doesn't interfere with the other. To accomplish this the master depth image captures
            // (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) before the color image, and the subordinate camera captures its
            // depth image (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image. This gives us two depth
            // images centered around the color image as closely as possible.
            camera_config.depth_delay_off_color_usec = MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2;
            return camera_config;
        }

    #else
        const std::string USE_KINECT_CAMERA_ERROR{"OpenPose CMake must be compiled with the `USE_KINECT_CAMERA`"
            " flag in order to use the KINECT camera. Alternatively, disable `--kinect_camera`."};
    #endif

    // PIMPL idiom 结构体的实现
    struct SpinnakerWrapper::Implk4aWrapper
    {
        #ifdef USE_FLIR_CAMERA
            bool mInitialized;
            CameraParameterReader mCameraParameterReader;
            Point<int> mResolution;

            std::vector<std::uint32_t> device_indices;
            std::vector<cv::Mat> mCvMats;
            std::vector<std::string> mSerialNumbers;

            // Camera index
            const int mCameraIndex;

            // Undistortion
            const bool mUndistortImage;
            std::vector<cv::Mat> mRemoveDistortionMaps1;
            std::vector<cv::Mat> mRemoveDistortionMaps2;

            // Thread
            bool mThreadOpened;
            std::vector<k4a::image> mBuffer;
            std::mutex mBufferMutex;
            std::atomic<bool> mCloseThread;
            std::thread mThread;

            // 构造函数
            Implk4aWrapper(const bool undistortImage, const int cameraIndex) :
                mInitialized{false},
                mCameraIndex{cameraIndex},
                mUndistortImage{undistortImage}
            {
            }

            void readAndUndistortImage(const int i, const k4a::image& imagePtr,
                                       const cv::Mat& cameraIntrinsics = cv::Mat(),
                                       const cv::Mat& cameraDistorsions = cv::Mat())
            {
                try
                {
                    // k4a to cv::Mat
                    const auto cvMatDistorted = color_to_opencv(imagePtr);
                    // const auto cvMatDistorted = spinnakerWrapperToCvMat(imagePtr);
                    // Undistort
                    if (mUndistortImage)
                    {
                        // Sanity check
                        if (cameraIntrinsics.empty() || cameraDistorsions.empty())
                            error("Camera intrinsics/distortions were empty.", __LINE__, __FUNCTION__, __FILE__);
                        // // Option a - 80 ms / 3 images
                        // // http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort
                        // cv::undistort(cvMatDistorted, mCvMats[i], cameraIntrinsics, cameraDistorsions);
                        // // In OpenCV 2.4, cv::undistort is exactly equal than cv::initUndistortRectifyMap
                        // (with CV_16SC2) + cv::remap (with LINEAR). I.e., opLog(cv::norm(cvMatMethod1-cvMatMethod2)) = 0.
                        // Option b - 15 ms / 3 images (LINEAR) or 25 ms (CUBIC)
                        // Distortion removal - not required and more expensive (applied to the whole image instead of
                        // only to our interest points)

                        // 矫正需要有两个映射矩阵（x和y方向），这两个矩阵可以通过函数cv::initUndistortRectifyMap()得到。
                        if (mRemoveDistortionMaps1[i].empty() || mRemoveDistortionMaps2[i].empty())
                        {
                            const auto imageSize = cvMatDistorted.size();
                            cv::initUndistortRectifyMap(cameraIntrinsics,
                                                        cameraDistorsions,
                                                        cv::Mat(),
                                                        // cameraIntrinsics instead of cv::getOptimalNewCameraMatrix to
                                                        // avoid black borders
                                                        cameraIntrinsics,
                                                        // #include <opencv2/calib3d/calib3d.hpp> for next line
                                                        // cv::getOptimalNewCameraMatrix(cameraIntrinsics,
                                                        //                               cameraDistorsions,
                                                        //                               imageSize, 1,
                                                        //                               imageSize, 0),
                                                        imageSize,
                                                        CV_16SC2, // Faster, less memory
                                                        // CV_32FC1, // More accurate
                                                        mRemoveDistortionMaps1[i],
                                                        mRemoveDistortionMaps2[i]);
                        }
                        // 图像矫正
                        cv::remap(cvMatDistorted, mCvMats[i],
                                  mRemoveDistortionMaps1[i], mRemoveDistortionMaps2[i],
                                  // cv::INTER_NEAREST);
                                  cv::INTER_LINEAR);
                                  // cv::INTER_CUBIC);
                                  // cv::INTER_LANCZOS4); // Smoother, but we do not need this quality & its >>expensive
                    }
                    // Baseline (do not undistort)
                    else
                        mCvMats[i] = cvMatDistorted.clone();
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }

            void bufferingThread()
            {
                #ifdef USE_FLIR_CAMERA
                    try
                    {
                        mCloseThread = false;
                        // Get cameras - ~0.005 ms (3 cameras)
                        std::vector<k4a::device> cameraPtrs(device_indices.GetSize()); // 为当前连接的相机构造一个指针向量组

                        for (auto i = 0u; i < cameraPtrs.size(); i++)
                            cameraPtrs.at(i) = mCameraList.GetBySerial(mSerialNumbers.at(i)); // Sorted by Serial Number //根据序列号为指针赋值
                            // cameraPtrs.at(i) = mCameraList.GetByIndex(i); // Sorted by however Spinnaker decided

                        while (!mCloseThread)
                        {
                            // Trigger
                            for (auto i = 0u; i < cameraPtrs.size(); i++)
                            {
                                // Retrieve GenICam nodemap
                                auto& iNodeMap = cameraPtrs[i]->GetNodeMap(); //获取每个相机的 GenICam 节点映射

                                Spinnaker::GenApi::CEnumerationPtr ptrAcquisitionMode = iNodeMap.GetNode("AcquisitionMode");

                                const auto result = GrabNextImageByTrigger(iNodeMap); //采集图像
                                if (result != 0)
                                    error("Error in GrabNextImageByTrigger.", __LINE__, __FUNCTION__, __FILE__);
                            }

                            // Get frame
                            std::vector<Spinnaker::ImagePtr> imagePtrs(cameraPtrs.size());
                            for (auto i = 0u; i < cameraPtrs.size(); i++)
                                imagePtrs.at(i) = cameraPtrs.at(i)->GetNextImage();

                            // Move to buffer
                            bool imagesExtracted = true;
                            // 检查每个相机获取的图像是否完整
                            for (auto& imagePtr : imagePtrs)
                            {
                                if (imagePtr->IsIncomplete())
                                {
                                    opLog("Image incomplete with image status " + std::to_string(imagePtr->GetImageStatus())
                                        + "...", Priority::High, __LINE__, __FUNCTION__, __FILE__);
                                    imagesExtracted = false;
                                    break;
                                }
                            }

                            if (imagesExtracted)
                            {
                                std::unique_lock<std::mutex> lock{mBufferMutex};
                                std::swap(mBuffer, imagePtrs); // 确保在访问共享资源（这里是缓冲区 mBuffer）时，不会有其他线程同时进行写入或读取
                                lock.unlock();
                                std::this_thread::sleep_for(std::chrono::microseconds{1});
                            }
                        }
                    }
                    catch (const std::exception& e)
                    {
                        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    }
                #endif
            }

            // This function acquires and displays images from each device. 
            // 从buffer中获取图像，存储在 mCvMats 中，最后转换为 OpenPose 的图像格式
            std::vector<Matrix> acquireImages(
                const std::vector<Matrix>& opCameraIntrinsics,
                const std::vector<Matrix>& opCameraDistorsions,
                const int cameraIndex = -1)
            {
                try
                {   
                    // 将 OpenPose 的相机内参和畸变数据结构转换为 OpenCV 的相应数据结构
                    OP_OP2CVVECTORMAT(cameraIntrinsics, opCameraIntrinsics)
                    OP_OP2CVVECTORMAT(cameraDistorsions, opCameraDistorsions)
                    // std::vector<cv::Mat> cvMats;

                    // Retrieve, convert, and return an image for each camera
                    // In order to work with simultaneous camera streams, nested loops are
                    // needed. It is important that the inner loop be the one iterating
                    // through the cameras; otherwise, all images will be grabbed from a
                    // single camera before grabbing any images from another.

                    // // Get cameras - ~0.005 ms (3 cameras)
                    // std::vector<Spinnaker::CameraPtr> cameraPtrs(cameraList.GetSize());
                    // for (auto i = 0u; i < cameraPtrs.size(); i++)
                    //     cameraPtrs.at(i) = cameraList.GetByIndex(i);

                    // Read raw images - ~0.15 ms (3 cameras)
                    // std::vector<Spinnaker::ImagePtr> imagePtrs(cameraPtrs.size());
                    // for (auto i = 0u; i < cameraPtrs.size(); i++)
                    //     imagePtrs.at(i) = cameraPtrs.at(i)->GetNextImage();
                    std::vector<k4a::image> imagePtrs;
                    // Retrieve frame

                    auto cvMatRetrieved = false;
                    while (!cvMatRetrieved)
                    {
                        // Retrieve frame
                        std::unique_lock<std::mutex> lock{mBufferMutex};
                        if (!mBuffer.empty())
                        {
                            std::swap(imagePtrs, mBuffer);
                            cvMatRetrieved = true;
                        }
                        // No frames available -> sleep & wait
                        else
                        {
                            lock.unlock();
                            std::this_thread::sleep_for(std::chrono::microseconds{5});
                        }
                    }
                    // Getting frames
                    // Retrieve next received image and ensure image completion
                    // Spinnaker::ImagePtr imagePtr = cameraPtrs.at(i)->GetNextImage();

                    // All images completed
                    bool imagesExtracted = true;
                    for (auto& imagePtr : imagePtrs)
                    {
                        if (imagePtr->is_valid())
                        {
                            opLog("Image incomplete with image status " + std::to_string(imagePtr->GetImageStatus())
                                + "...", Priority::High, __LINE__, __FUNCTION__, __FILE__);
                            imagesExtracted = false;
                            break;
                        }
                    }
                    mCvMats.clear();
                    // Convert to cv::Mat
                    
                    if (imagesExtracted)
                    {
                        // // Original image --> BGR uchar image - ~4 ms (3 cameras)
                        // for (auto& imagePtr : imagePtrs)
                        //     imagePtr = spinnakerImagePtrToColor(imagePtr);

                        // Init anti-distortion matrices first time
                        if (mRemoveDistortionMaps1.empty())
                            mRemoveDistortionMaps1.resize(imagePtrs.size());
                        if (mRemoveDistortionMaps2.empty())
                            mRemoveDistortionMaps2.resize(imagePtrs.size());

                        // Multi-thread undistort (slowest function in the class)
                        //     ~7.7msec (3 cameras + multi-thread + (initUndistortRectifyMap + remap) + LINEAR)
                        //     ~23.2msec (3 cameras + multi-thread + (initUndistortRectifyMap + remap) + CUBIC)
                        //     ~35msec (3 cameras + multi-thread + undistort)
                        //     ~59msec (2 cameras + single-thread + undistort)
                        //     ~75msec (3 cameras + single-thread + undistort)
                        mCvMats.resize(imagePtrs.size());
                        // All cameras
                        if (cameraIndex < 0)
                        {
                            // Undistort image
                            if (mUndistortImage)
                            {
                                std::vector<std::thread> threads(imagePtrs.size()-1);
                                for (auto i = 0u; i < threads.size(); i++)
                                {
                                    // Multi-thread option
                                    threads.at(i) = std::thread{&Implk4aWrapper::readAndUndistortImage, this, i,
                                                                imagePtrs.at(i), cameraIntrinsics.at(i),
                                                                cameraDistorsions.at(i)};
                                    // // Single-thread option
                                    // readAndUndistortImage(i, imagePtrs.at(i), cameraIntrinsics.at(i), cameraDistorsions.at(i));
                                }
                                readAndUndistortImage((int)imagePtrs.size()-1, imagePtrs.back(), cameraIntrinsics.back(),
                                                      cameraDistorsions.back());
                                // Close threads
                                for (auto& thread : threads)
                                    if (thread.joinable())
                                        thread.join();
                            }
                            // Do not undistort image
                            else
                            {
                                for (auto i = 0u; i < imagePtrs.size(); i++)
                                    readAndUndistortImage(i, imagePtrs.at(i));
                            }
                        }
                        // Only 1 camera
                        else
                        {
                            // Sanity check
                            if ((unsigned int)cameraIndex >= imagePtrs.size())
                                error("There are only " + std::to_string(imagePtrs.size())
                                      + " cameras, but you asked for the "
                                      + std::to_string(cameraIndex+1) +"-th camera (i.e., `--kinect_camera_index "
                                      + std::to_string(cameraIndex) +"`), which doesn't exist. Note that the index is"
                                      + " 0-based.", __LINE__, __FUNCTION__, __FILE__);
                            // Undistort image
                            if (mUndistortImage)
                                readAndUndistortImage(cameraIndex, imagePtrs.at(cameraIndex), cameraIntrinsics.at(cameraIndex),
                                                      cameraDistorsions.at(cameraIndex));
                            // Do not undistort image
                            else
                                readAndUndistortImage(cameraIndex, imagePtrs.at(cameraIndex));
                            mCvMats = std::vector<cv::Mat>{mCvMats[cameraIndex]};
                        }
                    }

                    OP_CV2OPVECTORMAT(opMats, mCvMats) //数据结构的转换
                    return opMats;
                }
                catch (k4a::error &e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    return {};
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    return {};
                }
            }
        #endif
    };

    SpinnakerWrapper::SpinnakerWrapper(const std::string& cameraParameterPath, const Point<int>& resolution,
                                       const bool undistortImage, const int cameraIndex)
        #ifdef USE_FLIR_CAMERA
            : upImpl{new Implk4aWrapper{undistortImage, cameraIndex}}
        #endif
    {
        #ifdef USE_FLIR_CAMERA
            try
            {
                // Clean previous unclosed builds (e.g., if core dumped in the previous code using the cameras)
                release();

                upImpl->mInitialized = true;

                // Print application build information
                opLog(std::string{ "Application build date: " } + __DATE__ + " " + __TIME__, Priority::High);

                // Retrieve singleton reference to upImpl->mSystemPtr object
                // 在 Spinnaker SDK 中，System 类是整个相机系统的入口点，提供了对相机系统的全局控制。
                upImpl->mSystemPtr = Spinnaker::System::GetInstance();  

                // Retrieve list of cameras from the upImpl->mSystemPtr
                upImpl->mCameraList = upImpl->mSystemPtr->GetCameras();

                const unsigned int numCameras = upImpl->mCameraList.GetSize();

                opLog("Number of cameras detected: " + std::to_string(numCameras), Priority::High);

                // Finish if there are no cameras
                if (numCameras == 0)
                {
                    // Clear camera list before releasing upImpl->mSystemPtr
                    upImpl->mCameraList.Clear();

                    // Release upImpl->mSystemPtr
                    upImpl->mSystemPtr->ReleaseInstance();

                    // opLog("Not enough cameras!\nPress Enter to exit...", Priority::High);
                    // getchar();

                    error("No cameras detected.", __LINE__, __FUNCTION__, __FILE__);
                }
                opLog("Camera system initialized...", Priority::High);

                //
                // Retrieve transport layer nodemaps and print device information for
                // each camera
                //
                // *** NOTES ***
                // This example retrieves information from the transport layer nodemap
                // twice: once to print device information and once to grab the device
                // serial number. Rather than caching the nodemap, each nodemap is
                // retrieved both times as needed.
                //
                opLog("\n*** DEVICE INFORMATION ***\n", Priority::High);

                // 打印设备信息
                for (auto i = 0; i < upImpl->mCameraList.GetSize(); i++)
                {
                    // Select camera
                    auto cameraPtr = upImpl->mCameraList.GetByIndex(i);

                    // Retrieve TL device nodemap
                    auto& iNodeMapTLDevice = cameraPtr->GetTLDeviceNodeMap();

                    // Print device information
                    auto result = printDeviceInfo(iNodeMapTLDevice, i);
                    if (result < 0)
                        error("Result > 0, error " + std::to_string(result) + " occurred...",
                                  __LINE__, __FUNCTION__, __FILE__);
                }

                // 初始化相机，并设置触发模式
                for (auto i = 0; i < upImpl->mCameraList.GetSize(); i++)
                {
                    // Select camera
                    auto cameraPtr = upImpl->mCameraList.GetByIndex(i);

                    // Initialize each camera
                    // You may notice that the steps in this function have more loops with
                    // less steps per loop; this contrasts the acquireImages() function
                    // which has less loops but more steps per loop. This is done for
                    // demonstrative purposes as both work equally well.
                    // Later: Each camera needs to be deinitialized once all images have been
                    // acquired.
                    cameraPtr->Init();

                    // Retrieve GenICam nodemap
                    auto& iNodeMap = cameraPtr->GetNodeMap();

                    // Configure trigger
                    int result = configureTrigger(iNodeMap);
                    if (result < 0)
                        error("Result > 0, error " + std::to_string(result) + " occurred...",
                                  __LINE__, __FUNCTION__, __FILE__);

                    // // Configure chunk data
                    // result = configureChunkData(iNodeMap);
                    // if (result < 0)
                    //     return result;

                    // // Remove buffer --> Always get newest frame
                    // Spinnaker::GenApi::INodeMap& snodeMap = cameraPtr->GetTLStreamNodeMap();
                    // Spinnaker::GenApi::CEnumerationPtr ptrBufferHandlingMode = snodeMap.GetNode(
                    //     "StreamBufferHandlingMode");
                    // if (!Spinnaker::GenApi::IsAvailable(ptrBufferHandlingMode)
                    //     || !Spinnaker::GenApi::IsWritable(ptrBufferHandlingMode))
                    //     error("Unable to change buffer handling mode", __LINE__, __FUNCTION__, __FILE__);

                    // Spinnaker::GenApi::CEnumEntryPtr ptrBufferHandlingModeNewest = ptrBufferHandlingMode->GetEntryByName(
                    //     "NewestFirstOverwrite");
                    // if (!Spinnaker::GenApi::IsAvailable(ptrBufferHandlingModeNewest)
                    //     || !IsReadable(ptrBufferHandlingModeNewest))
                    //     error("Unable to set buffer handling mode to newest (entry 'NewestFirstOverwrite' retrieval)."
                    //               " Aborting...", __LINE__, __FUNCTION__, __FILE__);
                    // int64_t bufferHandlingModeNewest = ptrBufferHandlingModeNewest->GetValue();

                    // ptrBufferHandlingMode->SetIntValue(bufferHandlingModeNewest);
                }

                // Prepare each camera to acquire images
                //
                // *** NOTES ***    
                // For pseudo-simultaneous streaming（伪并行流）, each camera is prepared as if it
                // were just one, but in a loop. Notice that cameras are selected with
                // an index. We demonstrate pseduo-simultaneous streaming because true
                // simultaneous streaming would require multiple process or threads,
                // which is too complex for an example.
                //
                // Serial numbers are the only persistent objects we gather in this
                // example, which is why a std::vector is created.

                //设置相机的采集模式与分辨率
                for (auto i = 0; i < upImpl->mCameraList.GetSize(); i++)
                {
                    // Select camera
                    auto cameraPtr = upImpl->mCameraList.GetByIndex(i);

                    /**************************************************************************************/

                    // Set acquisition mode to continuous
                    // 在这种模式下，相机会持续地采集图像，而不需要外部的触发信号。这与其他模式，比如单帧模式（single frame mode）或者触发模式（trigger mode）相对应。
                    Spinnaker::GenApi::CEnumerationPtr ptrAcquisitionMode = cameraPtr->GetNodeMap().GetNode(
                        "AcquisitionMode");

                    if (!Spinnaker::GenApi::IsAvailable(ptrAcquisitionMode)
                        || !Spinnaker::GenApi::IsWritable(ptrAcquisitionMode))
                        error("Unable to set acquisition mode to continuous (node retrieval; camera "
                              + std::to_string(i) + "). Aborting...", __LINE__, __FUNCTION__, __FILE__);

                    Spinnaker::GenApi::CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName(
                        "Continuous");
                    if (!Spinnaker::GenApi::IsAvailable(ptrAcquisitionModeContinuous)
                        || !Spinnaker::GenApi::IsReadable(ptrAcquisitionModeContinuous))
                        error("Unable to set acquisition mode to continuous (entry 'continuous' retrieval "
                                  + std::to_string(i) + "). Aborting...", __LINE__, __FUNCTION__, __FILE__);

                    const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

                    ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

                    opLog("Camera " + std::to_string(i) + " acquisition mode set to continuous...", Priority::High);

                    /**************************************************************************************/

                    // Set camera resolution
                    // Retrieve GenICam nodemap
                    auto& iNodeMap = cameraPtr->GetNodeMap();
                    // Set offset
                    Spinnaker::GenApi::CIntegerPtr ptrOffsetX = iNodeMap.GetNode("OffsetX");
                    ptrOffsetX->SetValue(0);
                    Spinnaker::GenApi::CIntegerPtr ptrOffsetY = iNodeMap.GetNode("OffsetY");
                    ptrOffsetY->SetValue(0);
                    // Set resolution
                    if (resolution.x > 0 && resolution.y > 0)
                    {
                        // WARNING: setting the obset would require auto-change in the camera matrix parameters
                        // Set offset
                        // const Spinnaker::GenApi::CIntegerPtr ptrWidthMax = iNodeMap.GetNode("WidthMax");
                        // const Spinnaker::GenApi::CIntegerPtr ptrHeightMax = iNodeMap.GetNode("HeightMax");
                        // ptrOffsetX->SetValue((ptrWidthMax->GetValue() - resolution.x) / 2);
                        // ptrOffsetY->SetValue((ptrHeightMax->GetValue() - resolution.y) / 2);
                        // Set Width
                        Spinnaker::GenApi::CIntegerPtr ptrWidth = iNodeMap.GetNode("Width");
                        ptrWidth->SetValue(resolution.x);
                        // Set width
                        Spinnaker::GenApi::CIntegerPtr ptrHeight = iNodeMap.GetNode("Height");
                        ptrHeight->SetValue(resolution.y);
                    }
                    else
                    {
                        // 获取相机的最大分辨率（即最大画幅）
                        const Spinnaker::GenApi::CIntegerPtr ptrWidthMax = iNodeMap.GetNode("WidthMax");
                        const Spinnaker::GenApi::CIntegerPtr ptrHeightMax = iNodeMap.GetNode("HeightMax");
                        // Set Width
                        Spinnaker::GenApi::CIntegerPtr ptrWidth = iNodeMap.GetNode("Width");
                        ptrWidth->SetValue(ptrWidthMax->GetValue());
                        // Set width
                        Spinnaker::GenApi::CIntegerPtr ptrHeight = iNodeMap.GetNode("Height");
                        ptrHeight->SetValue(ptrHeightMax->GetValue());
                        opLog("Choosing maximum resolution for flir camera (" + std::to_string(ptrWidth->GetValue())
                            + " x " + std::to_string(ptrHeight->GetValue()) + ").", Priority::High);
                    }

                    /**************************************************************************************/

                    // Begin acquiring images
                    cameraPtr->BeginAcquisition();

                    opLog("Camera " + std::to_string(i) + " started acquiring images...", Priority::High);
                }

                //打印相机序列号信息
                // Retrieve device serial number for filename
                opLog("\nReading (and sorting by) serial numbers...", Priority::High);
                const bool sorted = true;
                upImpl->mSerialNumbers = getSerialNumbers(upImpl->mCameraList, sorted);
                const auto& serialNumbers = upImpl->mSerialNumbers;
                for (auto i = 0u; i < serialNumbers.size(); i++)
                    opLog("Camera " + std::to_string(i) + " serial number set to "
                        + serialNumbers[i] + "...", Priority::High);
                // 如果只有一个相机
                if (upImpl->mCameraIndex >= 0)
                    opLog("Only using camera index " + std::to_string(upImpl->mCameraIndex) + ", i.e., serial number "
                        + serialNumbers[upImpl->mCameraIndex] + "...", Priority::High);

                // 读取相机参数
                // Read camera parameters from SN
                if (upImpl->mUndistortImage)
                {
                    // If all images required
                    if (upImpl->mCameraIndex < 0)
                        upImpl->mCameraParameterReader.readParameters(cameraParameterPath, serialNumbers);
                    // If only one required
                    else
                    {
                        upImpl->mCameraParameterReader.readParameters(
                            cameraParameterPath,
                            std::vector<std::string>(serialNumbers.size(), serialNumbers.at(upImpl->mCameraIndex)));
                    }
                }

                // Start buffering thread
                upImpl->mThreadOpened = true;
                upImpl->mThread = std::thread{&SpinnakerWrapper::Implk4aWrapper::bufferingThread, this->upImpl};

                // Get resolution
                const auto cvMats = getRawFrames();
                // Sanity check
                if (cvMats.empty())
                    error("Cameras could not be opened.", __LINE__, __FUNCTION__, __FILE__);
                // Get resolution
                upImpl->mResolution = Point<int>{cvMats[0].cols(), cvMats[0].rows()};

                const std::string numberCameras = std::to_string(upImpl->mCameraIndex < 0 ? serialNumbers.size() : 1);
                opLog("\nRunning for " + numberCameras + " out of " + std::to_string(serialNumbers.size())
                    + " camera(s)...\n\n*** IMAGE ACQUISITION ***\n", Priority::High);
            }
            catch (const Spinnaker::Exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        #else
            UNUSED(cameraParameterPath);
            UNUSED(resolution);
            UNUSED(undistortImage);
            UNUSED(cameraIndex);
            error(USE_FLIR_CAMERA_ERROR, __LINE__, __FUNCTION__, __FILE__);
        #endif
    }

    SpinnakerWrapper::~SpinnakerWrapper()
    {
        try
        {
            release();
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<Matrix> SpinnakerWrapper::getRawFrames()
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                try
                {
                    // Sanity check
                    if (upImpl->mUndistortImage &&
                        (unsigned long long) upImpl->mCameraList.GetSize()
                            != upImpl->mCameraParameterReader.getNumberCameras())
                        error("The number of cameras must be the same as the INTRINSICS vector size.",
                          __LINE__, __FUNCTION__, __FILE__);
                    // Return frames
                    return upImpl->acquireImages(upImpl->mCameraParameterReader.getCameraIntrinsics(),
                                                 upImpl->mCameraParameterReader.getCameraDistortions(),
                                                 upImpl->mCameraIndex);
                }
                catch (const Spinnaker::Exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    return {};
                }
            #else
                error(USE_FLIR_CAMERA_ERROR, __LINE__, __FUNCTION__, __FILE__);
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> SpinnakerWrapper::getCameraMatrices() const
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                return upImpl->mCameraParameterReader.getCameraMatrices();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> SpinnakerWrapper::getCameraExtrinsics() const
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                return upImpl->mCameraParameterReader.getCameraExtrinsics();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> SpinnakerWrapper::getCameraIntrinsics() const
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                return upImpl->mCameraParameterReader.getCameraIntrinsics();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    Point<int> SpinnakerWrapper::getResolution() const
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                return upImpl->mResolution;
            #else
                return Point<int>{};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Point<int>{};
        }
    }

    bool SpinnakerWrapper::isOpened() const
    {
        try
        {
            #ifdef USE_FLIR_CAMERA
                return upImpl->mInitialized;
            #else
                return false;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    void SpinnakerWrapper::release()
    {
        #ifdef USE_FLIR_CAMERA
            try
            {
                if (upImpl->mInitialized)
                {
                    // Stop thread
                    // Close and join thread
                    if (upImpl->mThreadOpened)
                    {
                        upImpl->mCloseThread = true;
                        upImpl->mThread.join();
                    }

                    // End acquisition for each camera
                    // Notice that what is usually a one-step process is now two steps
                    // because of the additional step of selecting the camera. It is worth
                    // repeating that camera selection needs to be done once per loop.
                    // It is possible to interact with cameras through the camera list with
                    // GetByIndex(); this is an alternative to retrieving cameras as
                    // Spinnaker::CameraPtr objects that can be quick and easy for small tasks.
                    //
                    for (auto i = 0; i < upImpl->mCameraList.GetSize(); i++)
                    {
                        // Select camera
                        auto cameraPtr = upImpl->mCameraList.GetByIndex(i);

                        cameraPtr->EndAcquisition();

                        // Retrieve GenICam nodemap
                        auto& iNodeMap = cameraPtr->GetNodeMap();

                        // // Disable chunk data
                        // result = disableChunkData(iNodeMap);
                        // // if (result < 0)
                        // //     return result;

                        // Reset trigger
                        auto result = resetTrigger(iNodeMap);
                        if (result < 0)
                            error("Error happened..." + std::to_string(result), __LINE__, __FUNCTION__, __FILE__);

                        // Deinitialize each camera
                        // Each camera must be deinitialized separately by first
                        // selecting the camera and then deinitializing it.
                        cameraPtr->DeInit();
                    }

                    opLog("FLIR (Point-grey) capture completed. Releasing cameras...", Priority::High);

                    // Clear camera list before releasing upImpl->mSystemPtr
                    upImpl->mCameraList.Clear();

                    // Release upImpl->mSystemPtr
                    upImpl->mSystemPtr->ReleaseInstance();

                    // Setting the class as released
                    upImpl->mInitialized = false;

                    opLog("Cameras released! Exiting program.", Priority::High);
                }
                else
                {
                    // Open general system
                    auto systemPtr = Spinnaker::System::GetInstance();
                    auto cameraList = systemPtr->GetCameras();
                    if (cameraList.GetSize() > 0)
                    {

                        for (auto i = 0; i < cameraList.GetSize(); i++)
                        {
                            // Select camera
                            auto cameraPtr = cameraList.GetByIndex(i);
                            // Begin
                            cameraPtr->Init();
                            cameraPtr->BeginAcquisition();
                            // End
                            cameraPtr->EndAcquisition();
                            cameraPtr->DeInit();
                        }
                    }
                    cameraList.Clear();
                    systemPtr->ReleaseInstance();
                }
            }
            catch (const Spinnaker::Exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        #endif
    }
}
