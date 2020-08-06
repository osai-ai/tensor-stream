#pragma once
#include "Common.h"
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
	ParserParameters(std::string _inputFile = "", bool _enableDumps = false) :
		inputFile(_inputFile), enableDumps(_enableDumps) {

	}

	/*
	Path to input file, no matter where it's placed: remotely or locally.
	*/
	std::string inputFile;
	bool enableDumps;
};

class BitReader {
public:
	enum Base {
		NONE,
		DEC,
		HEX
	};
	enum Type {
		RAW,
		GOLOMB,
		SGOLOMB
	};
	BitReader(uint8_t* _byteData, int _dataSize);
	BitReader();
	std::vector<bool> FindNALType();
	std::vector<bool> ReadBits(int number);
	std::vector<bool> ReadGolomb();
	bool SkipBits(int number);
	bool SkipGolomb();
	int Convert(std::vector<bool> value, Type type, Base base);

	int getShiftInBits();
	int getByteIndex();
private:
	uint8_t* byteData;
	int dataSize;
	int byteIndex = 0;
	int shiftInBits = 0;
	bool findNAL();
	std::vector<bool> getVector(int value);
};

/*
The class allows to read frames from defined stream.
*/
class Parser {
public:
	Parser();
	/*
	Initialization of parser. Initialize work with rtmp, allocate recources.
	*/
	int Init(ParserParameters& input, std::shared_ptr<Logger> logger);

	/*
	The main function which read rtmp stream and write result to buffer. Should be executed in different thread.
	*/
	int Read();
	
	int readVideoFrame(std::pair<AVPacket*, bool>& dst);

	/*
	Returns next parsed frame. Frames will be returned as their appeared in bitstream without any loss.
	Arguments: Pointer to AVPacket structure where is demuxed frame will be stored.
	*/
	int Get(AVPacket* outputFrame);

	enum AnalyzeErrors {
		NONE = 0,
		B_POC,
		FRAME_NUM,
		GAPS_FRAME_NUM,
	};

	/*
	Analyze package for possible issues in syntax
	*/
	int Analyze(AVPacket* package);

	/*
	Close all existing handles, deallocate recources.
	*/
	void Close();

	int getWidth();
	int getHeight();

	/*
	Get input format context. Needed for internal interactions.
	*/
	AVFormatContext* getFormatContext();
	AVStream* getStreamHandle();
	int getVideoIndex();
	int getGopSize();
	int setGopSize(int gopSize);
private:
	/*
	*/
	int gopSize = 32;
	/*
	State of Parser object it was initialized/reseted with.
	*/
	ParserParameters state;
	/*
	Latest parsed frame and index indicated if this frame was taken from parser by Get() function
	*/
	std::pair<AVPacket*, bool> lastFrame;
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
	/*
	State of component
	*/
	bool isClosed = true;
	/*
	Frame number for bitstream analyzing, part of H264 bitstream syntax
	*/
	int frameNumValue = -1;
	int POC = 0;
	/*
	Bitstream filter for converting mp4->h264
	*/
	AVBitStreamFilterContext* bitstreamFilter;
	AVPacket* NALu;
	/*
	Instance of Logger class
	*/
	std::shared_ptr<Logger> logger;

	std::chrono::time_point<std::chrono::system_clock> latestFrameTimestamp;
};