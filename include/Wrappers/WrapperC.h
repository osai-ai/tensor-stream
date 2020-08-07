#pragma once
#include <iostream>
#include "Common.h"
#include "Parser.h"
#include "Decoder.h"
#include "VideoProcessor.h"
/** @defgroup cppAPI C++ API
@brief The list of TensorStream components can be used via C++ interface
@details Here are all the classes, enums, functions described which can be used via C++ to do RTMP/local stream converting to CUDA memory with additional post-processing conversions
@{
*/

class StreamPool {
public:
	std::shared_ptr<Parser> cacheStream(std::string inputFile);
	std::shared_ptr<Parser> getParser(std::string inputFile);
	std::map<std::string, std::shared_ptr<Parser> > getParsers();
	std::shared_ptr<Logger> getLogger();
	int setLogger(std::shared_ptr<Logger> logger);
private:
	std::shared_ptr<Logger> logger;
	std::map<std::string, std::shared_ptr<Parser> > parserArr;
};

/**
Class which allow start decoding process and get Pytorch tensors with post-processed frame data
*/
class TensorStream {
public:
	int addStreamPool(std::shared_ptr<StreamPool> streamPool);

	int resetPipeline(std::string inputFile);
/** Initialization of TensorStream pipeline
 @param[in] inputFile Path to stream should be decoded
 @param[in] maxConsumers Allowed number of simultaneously working consumers
 @param[in] cudaDevice GPU used for execution
 @anchor decoderBuffer
 @param[in] decoderBuffer How many decoded frames should be stored in internal buffer
 @warning decodedBuffer should be less than DPB
 @return Status of execution, one of @ref ::Internal values
*/
	int initPipeline(std::string inputFile, uint8_t maxConsumers = 5, uint8_t cudaDevice = defaultCUDADevice, uint8_t decoderBuffer = 10, FrameRateMode frameRate = FrameRateMode::NATIVE, bool cuda = true, int threads = 1);

/** Get parameters from bitstream
 @return Map with "framerate_num", "framerate_den", "width", "height" values
*/
	std::map<std::string, int> getInitializedParams();

/** Start decoding of bitstream in separate thread
 @return Status of execution, one of @ref ::Internal values
*/
	int startProcessing();

/** Get decoded and post-processed frame. Pixel format can be either float or uint8_t depending on @ref normalization
 @param[in] consumerName Consumer unique ID
 @param[in] index Specify which frame should be read from decoded buffer. Can take values in range [-@ref decoderBuffer, 0]
 @param[in] frameParameters Frame specific parameters, see @ref ::FrameParameters for more information
 @return Decoded frame in CUDA memory and index of decoded frame
*/
	template <class T>
	std::tuple<T*, int> getFrame(std::string consumerName, int index, FrameParameters frameParameters);

/** Get decoded and post-processed frames by they absoulte position in video. Pixel format can be either float or uint8_t depending on @ref normalization
 @param[in] consumerName Consumer unique ID
 @param[in] index Specify batch of frames should be read from stream. Can take values in range [0, video length]
 @param[in] frameParameters Frame specific parameters, see @ref ::FrameParameters for more information
 @return Decoded frames in CUDA memory and indexes of decoded frame
*/
	template <class T>
	std::vector<T*> getFrameAbsolute(std::vector<int> index, FrameParameters frameParameters);
/** Close TensorStream session
*/
	void endProcessing();
/** Enable logs from TensorStream
 @param[in] level Specify output level of logs, see @ref ::LogsLevel for supported values
*/
	void enableLogs(int level);
/** Dump the frame in CUDA memory to hard driver. Pixel format can be either float or uint8_t depending on @ref normalization
 @param[in] frame CUDA memory should be dumped
 @param[in] frameParameters Parameters specific for passed frame, used in @ref TensorStream::getFrame() call
 @param[in] dumpFile File handler
 */
	template <class T>
	int dumpFrame(T* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile);
/** Enable NVTX logs from TensorStream
*/
	void enableNVTX();
/** Allow to skip stage with bitstream analyzing (skip frames, some bitstream conformance checks)
*/
	void skipAnalyzeStage();
/** Set timeout for frame reading (default: -1, means no timeout)
@param[in] value of timeout in ms
*/
	void setTimeout(int timeout);

/** Calculate GOP value which is used for batch loading optimization. Used in batch load mode only.
*/
	int enableBatchOptimization();

	int getTimeout();
	int getDelay();
private:
	int processingLoop();
	std::shared_ptr<StreamPool> streamPool = nullptr;

	std::mutex syncDecoded;
	std::mutex syncRGB;
	std::shared_ptr<Parser> parser;
	std::shared_ptr<Decoder> decoder;
	std::shared_ptr<VideoProcessor> vpp;
	AVPacket* parsed;
	int realTimeDelay = 0;
	double indexToDTSCoeff = 0;
	double DTSToMsCoeff = 0;
	std::pair<int, int> frameRate;
	FrameRateMode frameRateMode;
	bool shouldWork;
	bool skipAnalyze;
	std::vector<std::pair<std::string, AVFrame*> > decodedArr;
	std::vector<std::pair<std::string, AVFrame*> > processedArr;
	std::mutex freeSync;
	std::mutex closeSync;
	bool _cuda;

	std::map<std::string, bool> blockingStatuses;
	std::mutex blockingSync;
	std::condition_variable blockingCV;
	
	std::shared_ptr<Logger> logger;
	uint8_t currentCUDADevice;
};

/** 
@} 
*/