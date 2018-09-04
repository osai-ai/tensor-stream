#pragma once
#include <map>
#include <vector>
#include <memory>

/*should it be under extern clause?*/
#include <libavformat/avformat.h>


/*
Structure with initialization/reset parameters.
*/
struct ParserParameters {
	/*
	Path to input file, no matter where it's placed: remotely or locally.
	*/
	std::string inputFile;
	bool enableDumps = false;
	unsigned int bufferDeep = 10;
};

/*
The class allows to read frames from defined stream.
Buffer with frames is used inside to avoide some internet lags/packet loss/any other issues.
*/
class Parser {
public:
	/*
	Initialization of parser. Initialize work with rtmp, allocate recources.
	*/
	int Init(ParserParameters& input);

	/*
	The main function which read rtmp stream and write result to buffer. Should be executed in different thread.
	*/
	int Start();
	
	/*
	Returns next parsed frame. Frames will be returned as their appeared in bitstream without any loss.
	Arguments: Pointer to AVPacket structure where is demuxed frame will be stored.
	*/
	int Get(std::shared_ptr<AVPacket> outputFrame);

	/*
	Soft re-init of current Parser entity with new parameters.
	*/
	int Reset(ParserParameters& input);

	/*
	Close all existing handles, deallocate recources.
	*/
	int Close();

	/*
	Get input format context. Needed for internal interactions.
	*/
	AVFormatContext* getFormatContext();
private:
	/*
	State of Parser object it was initialized/reseted with.
	*/
	ParserParameters state;
	/*
	The map with file descriptors for dumping intermediate frames.
	*/
	std::map<std::string, std::shared_ptr<FILE> > dumpFrame;
	/*
	Should be initialized during Init.
	Buffer which stores read frames.
	*/
	std::vector<std::shared_ptr<AVPacket> > framesBuffer;
	/*
	Index of latest given frame.
	*/
	unsigned int currentFrame;
	/*
	FFmpeg internal stuff, input file context, contains iterator which allow read frames one by one without pointing to frame's number.
	*/
	AVFormatContext *ifmt_ctx;
};