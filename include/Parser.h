#pragma once
#include <vector>

/*
The class allows to read frames from defined stream
*/
class Parser {
public:
	Parser();
	/*
	Initialization of parser. Initialize work with rtmp, allocate recources.
	*/
	int Init(std::string _inputFile);
	/*
	The main function which read rtmp stream and return frame.
	*/
	int Read(std::vector<uint8_t>& frame);
	/*
	Close all existing handles, deallocate recources.
	*/
	int Close();
private:
	std::string inputFile;
	/*
	some external ffmpeg stuff
	*/
};