#include <iostream>
#ifdef _DEBUG
#undef _DEBUG
#include <torch/extension.h>
#include <THC/THC.h>
#include <ATen/ATen.h>
#if (__linux__)
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#endif
#define _DEBUG
#else
#include <torch/extension.h>
#include <THC/THC.h>
#include <ATen/ATen.h>
#if (__linux__)
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#endif
#endif

#include "Common.h"
#include "Parser.h"
#include "Decoder.h"
#include "VideoProcessor.h"

class TensorStream {
public:
	int initPipeline(std::string inputFile, uint8_t decoderBuffer = 10);
	std::map<std::string, int> getInitializedParams();
	int startProcessing(int cudaDevice = 0);
	std::tuple<at::Tensor, int> getFrame(std::string consumerName, int index, FrameParameters frameParameters);
	void endProcessing();
	void enableLogs(int logsLevel);
	void enableNVTX();
	int dumpFrame(at::Tensor stream, std::string consumerName, FrameParameters frameParameters);
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
	std::vector<at::Tensor> tensors;
	std::vector<std::shared_ptr<uint8_t> > processedFrames;
	std::mutex freeSync;
	std::mutex closeSync;
	std::shared_ptr<Logger> logger;
};