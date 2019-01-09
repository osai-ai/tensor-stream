#include <iostream>
#include "Common.h"
#include "Parser.h"
#include "Decoder.h"
#include "VideoProcessor.h"

class VideoReader {
public:
	int initPipeline(std::string inputFile);
	std::map<std::string, int> getInitializedParams();
	int startProcessing();
	std::tuple<std::shared_ptr<uint8_t>, int> getFrame(std::string consumerName, int index, int pixelFormat, int dstWidth = 0, int dstHeight = 0);
	void endProcessing(int mode = HARD);
	void enableLogs(int _logsLevel);
	int dumpFrame(AVFrame* output, std::shared_ptr<FILE> dumpFile);
private:
	int processingLoop();
	std::mutex syncDecoded;
	std::mutex syncRGB;
	std::shared_ptr<Parser> parser;
	std::shared_ptr<Decoder> decoder;
	std::shared_ptr<VideoProcessor> vpp;
	AVPacket* parsed;
	int realTimeDelay = 0;
	bool shouldWork;
	std::vector<std::pair<std::string, AVFrame*> > decodedArr;
	std::vector<std::pair<std::string, AVFrame*> > processedArr;
	std::vector<std::shared_ptr<uint8_t> > CUDAOutputs;
	std::vector<std::shared_ptr<uint8_t> > processedFrames;
	std::mutex freeSync;
	std::mutex closeSync;
};