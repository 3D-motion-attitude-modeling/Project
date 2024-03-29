#ifndef OPENPOSE_PRODUCER_FLIR_READER_HPP
#define OPENPOSE_PRODUCER_FLIR_READER_HPP

#include <openpose/core/common.hpp>
#include <openpose/producer/producer.hpp>
#include <openpose/producer/spinnakerWrapper.hpp>

namespace op
{
    /**
     * FlirReader is an abstract class to extract frames from a FLIR stereo-camera system. Its interface imitates the
     * cv::VideoCapture class, so it can be used quite similarly to the cv::VideoCapture class. Thus,
     * it is quite similar to VideoReader and WebcamReader.
     */
    class OP_API FlirReader : public Producer
    {
    public:
        /**
         * Constructor of FlirReader. It opens all the available FLIR cameras
         */

        // cameraParametersPath is like 'models/cameraParameters/flir/'
        // 是否去畸变--undistortImage, 摄像头编号--cameraIndex（-1表示所有摄像头一同同步读取）
        explicit FlirReader(const std::string& cameraParametersPath, const Point<int>& cameraResolution,
                            const bool undistortImage = true, const int cameraIndex = -1);

        virtual ~FlirReader();

        // 获取多摄像头的内外参数，在完成文件生成后可以借用Openpose内部CameraParameterReader来实现
        std::vector<Matrix> getCameraMatrices();

        std::vector<Matrix> getCameraExtrinsics();

        std::vector<Matrix> getCameraIntrinsics();

        std::string getNextFrameName();

        // 详情见producer.hpp
        bool isOpened() const;

        // End acquisition for each camera
        void release();

        double get(const int capProperty);

        void set(const int capProperty, const double value);

    private:
        //辅助实现的类
        SpinnakerWrapper mSpinnakerWrapper;

        // 保留
        Point<int> mResolution;
        unsigned long long mFrameNameCounter;

        // 获取多摄像头中第一个摄像头的帧图像
        Matrix getRawFrame();

        // 获取多摄像头同一时刻的帧图像
        std::vector<Matrix> getRawFrames();

        DELETE_COPY(FlirReader);
    };
}

#endif // OPENPOSE_PRODUCER_FLIR_READER_HPP
