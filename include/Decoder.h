#pragma once
#include <map>
#include "Parser.h"

/*
Structure with initialization/reset parameters.
*/
struct DecoderParameters {
	DecoderParameters(std::shared_ptr<Parser> _parser = nullptr, bool _enableDumps = false, unsigned int _bufferDeep = 10) :
		parser(_parser), enableDumps(_enableDumps), bufferDeep(_bufferDeep) {

	}
	std::shared_ptr<Parser> parser;
	bool enableDumps;
	unsigned int bufferDeep;
};

/*
The class takes input from reader, decode frames in NV12 format and return frames in (GPU) CUDA memory
*/
class Decoder {
public:
	Decoder();
	/*
	Initialize decoder with corresponding parameters. Allocate all neccessary resources.
	*/
	int Init(DecoderParameters* input);

	/*
	Asynchronous call, start decoding process. Should be executed in different thread.
	*/
	int Start();

	/*
	Blocked call, returns whether already decoded frame from cache or latest decoded frame which hasn't been reported yet.
	Arguments: 
		int index: index of desired frame.
			Return: last frame if index > 0, frame from cache in case of index < cacheDepth.
	*/
	int GetFrame(int index, std::string consumerName, uint8_t* outputFrame);

	/*
	Close all existing handles, deallocate recources.
	*/
	void Close();
private:
	/*
	It help understand whether allowed or not return frame. If some frame was reported to current consumer and no any new frames were decoded need to wait.
	Parameters: Consumer's name and latest given frame number
	*/
	std::map<std::string, int> consumerStatus;
	/*
	Buffer stores already decoded frames in CUDA memory (frame index can be found in container)
	*/
	std::vector<std::shared_ptr<AVFrame> > framesBuffer;
	/*
	Index of latest decoded frame.
	*/
	unsigned int currentFrame = 0;
	/*
	The map with file descriptors for dumping intermediate frames.
	*/
	std::map<std::string, std::shared_ptr<FILE> > dumpFrame;
	/*
	Internal decoder's state
	*/
	DecoderParameters* state = nullptr;
	/*
	FFmpeg internal stuff
	*/
	AVCodecContext * decoderContext = nullptr;
	AVBufferRef* deviceReference = nullptr;
};