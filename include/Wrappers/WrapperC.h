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

/**
Class which allow start decoding process and get Pytorch tensors with post-processed frame data
*/
class TensorStream {
public:
/** Initialization of TensorStream pipeline
 @param[in] inputFile Path to stream should be decoded
 @anchor decoderBuffer
 @param[in] decoderBuffer How many decoded frames should be stored in internal buffer
 @warning decodedBuffer should be less than DPB
 @return Status of execution, one of @ref ::Internal values
*/
	int initPipeline(std::string inputFile, uint8_t decoderBuffer = 10);

/** Get parameters from bitstream
 @return Map with "framerate_num", "framerate_den", "width", "height" values
*/
	std::map<std::string, int> getInitializedParams();

/** Start decoding of bitstream in separate thread
 @return Status of execution, one of @ref ::Internal values
*/
	int startProcessing();

/** Get decoded and post-processed frame
 @param[in] consumerName Consumer unique ID
 @param[in] index Specify which frame should be read from decoded buffer. Can take values in range [-@ref decoderBuffer, 0]
 @param[in] frameParameters Frame specific parameters, see @ref ::FrameParameters for more information
 @return Decoded frame in CUDA memory and index of decoded frame
*/
	std::tuple<float*, int> getFrame(std::string consumerName, int index, FrameParameters frameParameters);
/** Close TensorStream session
 @param[in] mode Value from @ref ::CloseLevel
*/
	void endProcessing(int mode = HARD);
/** Enable logs from TensorStream
 @param[in] level Specify output level of logs, see @ref ::LogsLevel for supported values
*/
	void enableLogs(int level);
/** Dump the frame in CUDA memory to hard driver
 @param[in] frame CUDA memory should be dumped
 @param[in] frameParameters Parameters specific for passed frame, used in @ref TensorStream::getFrame() call
 @param[in] dumpFile File handler
 */
	int dumpFrame(float* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile);
	int getDelay();
private:
	int processingLoop();
	std::mutex syncDecoded;
	std::mutex syncRGB;
	std::shared_ptr<Parser> parser;
	std::shared_ptr<Decoder> decoder;
	std::shared_ptr<VideoProcessor> vpp;
	AVPacket* parsed;
	int realTimeDelay = 0;
	std::pair<int, int> frameRate;
	bool shouldWork;
	std::vector<std::pair<std::string, AVFrame*> > decodedArr;
	std::vector<std::pair<std::string, AVFrame*> > processedArr;
	std::mutex freeSync;
	std::mutex closeSync;
};

/** 
@} 
*/