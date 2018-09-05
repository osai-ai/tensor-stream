#pragma once
#include <map>
#include <vector>
#include <memory>

extern "C"
{
#include <libavformat/avformat.h>
}

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
	void Close();

	/*
	Get input format context. Needed for internal interactions.
	*/
	AVFormatContext* getFormatContext();
	AVStream* getStreamHandle();
private:
	/*
	State of Parser object it was initialized/reseted with.
	*/
	ParserParameters state;
	/*
	Should be initialized during Init.
	Buffer which stores read frames.
	*/
	std::vector<std::shared_ptr<AVPacket> > framesBuffer;
	/*
	Index of latest given frame.
	*/
	unsigned int currentFrame = 0;
	/*
	FFmpeg internal stuff, input file context, contains iterator which allow read frames one by one without pointing to frame's number.
	*/
	AVFormatContext *formatContext = nullptr;
	/*
	Video stream in container. Contains info about codec, etc
	*/
	AVStream * videoStream = nullptr;
	/*
	Is used only in case of dumping to .bin data
	*/
	AVFormatContext *dumpContext = nullptr;
	/*
	Position of video in container
	*/
	int videoIndex = -1;
};