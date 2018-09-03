#pragma once
#include <map>

#include "Parser.h"

/*
The class takes input from reader, decode frames in NV12 format and return frames in (GPU) CUDA memory
*/

class Decoder {
public:
	/*
	Initialize 
	Arguments:
		int cacheDepth:
		shared_ptr<Parser> parser;
	*/
	int Init(/*???*/);

	int Start(/*???*/);

	/*
	Blocked call, returns whether already decoded frame from cache or latest decoded frame which hasn't been reported yet.
	Arguments: 
		int index: index of desired frame.
			Return: last frame if index > 0, frame from cache in case of index < cacheDepth.
	*/
	int GetFrame(int index, std::string consumerName, uint8_t* outputFrame);

	/*
	
	*/
	int Close();
private:
	/*
	It shows allowed or not take frame. If such frame was reported to current consumer and no any new frames were decoded need to wait.
	*/
	std::map<std::string, int> consumerStatus;
};