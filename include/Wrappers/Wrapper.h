#pragma once
#include "Common.h"
#include "Parser.h"
#include "Decoder.h"
#include "VideoProcessor.h"

class TensorCommon {
protected:
	int initPipeline(std::string inputFile, uint8_t maxConsumers, uint8_t cudaDevice, uint8_t decoderBuffer);
	template <class T>
	int dumpFrame(T* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile);
	void endProcessing();
	void enableLogs(int level);
	void enableNVTX();

	std::shared_ptr<Parser> _parser;
	std::shared_ptr<Decoder> _decoder;
	std::shared_ptr<VideoProcessor> _vpp;
	std::shared_ptr<Logger> _logger;
	uint8_t _currentCUDADevice;
};

class TensorStreamCommon : public TensorCommon {
protected:
	int initPipeline(std::string inputFile, uint8_t maxConsumers, uint8_t cudaDevice, uint8_t decoderBuffer, FrameRateMode frameRateMode);
	int startProcessing();
	int processingLoop();
	std::map<std::string, int> getInitializedParams();
	template <class T>
	std::tuple<T*, int> getFrame(std::string consumerName, int index, FrameParameters frameParameters);
	void skipAnalyzeStage();
	void setTimeout(int timeout);
	int getTimeout();
	int getDelay();

	int checkGetComplete(std::map<std::string, bool>& blockingStatuses);
private:
	std::mutex _syncDecoded;
	std::mutex _syncRGB;
	AVPacket* _parsed;
	int _realTimeDelay = 0;
	std::pair<int, int> _frameRate;
	FrameRateMode _frameRateMode;
	bool _shouldWork;
	bool _skipAnalyze;
	std::vector<std::pair<std::string, AVFrame*> > _decodedArr;
	std::vector<std::pair<std::string, AVFrame*> > _processedArr;
	std::mutex _freeSync;
	std::mutex _closeSync;

	std::map<std::string, bool> _blockingStatuses;
	std::mutex _blockingSync;
	std::condition_variable _blockingCV;
};