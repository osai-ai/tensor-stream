#include "Parser.h"
#include <thread>
#include <bitset>
#include <numeric>

BitReader::BitReader(uint8_t* _byteData, int _dataSize) {
	byteData = _byteData;
	dataSize = _dataSize;
}

std::vector<bool> BitReader::getVector(int value) {
	std::vector<bool> result;
	while (value) {
		int remainder = value % 2;
		result.push_back(remainder);
		value /= 2;
	}
	return result;
}

bool BitReader::findNAL() {
	while (byteIndex != dataSize)
	if (Convert(ReadBits(8), BitReader::Base::DEC) == 0) {
		int startCodeCounter = 1;
		while (Convert(ReadBits(8), BitReader::Base::DEC) == 0) {
			startCodeCounter++;
		}
		if (startCodeCounter >= 2 && Convert(ReadBits(8), BitReader::Base::DEC) == 1) {
			return true;
		}
	}
	return false;
}

int BitReader::FindNALType() {
	return 0;
}

std::vector<bool> BitReader::ReadBits(int number) {
	std::vector<bool> result;
	int startIndex = shiftInBits;
	int endIndex   = shiftInBits + number;
	std::vector<bool> value = getVector(byteData[byteIndex]);
	for (int i = startIndex; i < endIndex; i++) {
		//we read we last bit, need to take next byte
		if (i && i % 8 == 0) {
			shiftInBits = 0;
			byteIndex++;
			value = getVector(byteData[byteIndex]);
		}
		result.push_back(value[i % 8]);
	}
	return result;
}

int BitReader::Convert(std::vector<bool> value, Base base) {
	int result = 0;
	switch (base) {
		case Base::DEC: 
		{
			int n = 0;
			for (auto i : value)
			{
				if (i)
				{
					result += pow(2, n++);
				}
			}
			break;
		}
		case Base::HEX:

		break;
	}
	return result;
}

int BitReader::getByteIndex() {
	return byteIndex;
}
int BitReader::getShiftInBits() {
	return shiftInBits;
}

int BitReader::ReadGolomb() {
	return 0;
}

int Parser::Init(ParserParameters& input) {
	state = input;
	int sts = OK;
	//packet_buffer - isn't empty
	sts = avformat_open_input(&formatContext, state.inputFile.c_str(), 0, /*&opts*/0);
	CHECK_STATUS(sts);
	sts = avformat_find_stream_info(formatContext, 0);
	CHECK_STATUS(sts);
	AVCodec* codec;
	videoIndex = av_find_best_stream(formatContext, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
	videoStream = formatContext->streams[videoIndex];
	videoStream->codec->codec = codec;
	if (state.enableDumps) {
		std::string dumpName = "bitstream.h264";
		sts = avformat_alloc_output_context2(&dumpContext, NULL, NULL, dumpName.c_str());
		CHECK_STATUS(sts);
		AVStream * outStream = avformat_new_stream(dumpContext, videoStream->codec->codec);
		sts = avcodec_copy_context(outStream->codec, videoStream->codec);
		CHECK_STATUS(sts);
		//Open output file
		if (!(dumpContext->oformat->flags & AVFMT_NOFILE)) {
			sts = avio_open(&dumpContext->pb, dumpName.c_str(), AVIO_FLAG_WRITE);
			CHECK_STATUS(sts);
		}
		//Write file header
		sts = avformat_write_header(dumpContext, NULL);
	}

	lastFrame = std::make_pair(new AVPacket(), false);
	isClosed = false;
	return sts;
}

int Parser::getWidth() {
	return videoStream->codec->width;
}

int Parser::getHeight() {
	return videoStream->codec->height;
}

int Parser::getVideoIndex() {
	return videoIndex;
}
Parser::Parser() {

}

int Parser::Analyze(AVPacket* package) {
	enum NALTypes {
		UNKNOWN = 0,
		SPS,
		PPS,
		SEI,
		SLICE_I,
		SLICE_P,
		SLICE_B
	} NALType = UNKNOWN;

	BitReader bitReader(lastFrame.first->data, lastFrame.first->size);
	//We need to find SLICE_*
	while (NALType < SLICE_I) {
		NALType = static_cast<NALTypes>(bitReader.FindNALType());
	}
	//here we have position after NAL header
	/*
	while (index < bitSize) {
		int value = (int)(bitData[index]);
		//Start code logic
		if (value == 0) {
			int startCodeCounter = 0;
			startCodeCounter++;
			//we need to analyze at least next byte
			index += 1;
			while (index < bitSize && (int)(bitData[index]) == 0) {
				startCodeCounter++;
			}
			if (startCodeCounter >= 2 && (int)(bitData[index]) == 1) {
				offsetStart = 0;
				continue;
			}
		}
		//the first byte contains info about nal_unit_type
		if (offsetStart == 0) {
			std::bitset<8> valueBin = std::bitset<8>(value);
			std::bitset<8> nalTypeMask = std::bitset<8>("11111"); //u(5) for nal_unit_type
			NALType = static_cast<NALTypes>((valueBin & nalTypeMask).to_ulong());
			offsetStart++;
			index++;
			continue;
		}
		//we are looking for SLICE_* NAL
		if (NALType >= NALTypes::SLICE_I) {
			
		}
		offsetStart++;
	}
	*/
	return OK;
}

int Parser::Read() {
	int sts = OK;
	bool videoFrame = false;
	while (videoFrame == false) {
		sts = av_read_frame(formatContext, lastFrame.first);
		CHECK_STATUS(sts);
		if ((lastFrame.first)->stream_index != videoIndex)
			continue;
		
		videoFrame = true;
		currentFrame++;

		//TODO: critical section + need to uninit?
		lastFrame.second = false;

		if (state.enableDumps) {
#ifdef DEBUG_INFO
			static int count = 0;
#endif
			//in our output file only 1 stream is available with index 0
			lastFrame.first->stream_index = 0;
			sts = av_write_frame(dumpContext, lastFrame.first);
			CHECK_STATUS(sts);
			lastFrame.first->stream_index = videoIndex;
#ifdef DEBUG_INFO
			count++;
			printf("Bitstream %d\n", count);
#endif
		}
	}
	return sts;
}

int Parser::Get(AVPacket* output) {
	if (lastFrame.second == false && lastFrame.first->stream_index == videoIndex) {
		//decoder is responsible for deallocating
		av_packet_ref(output, lastFrame.first);
		av_packet_unref(lastFrame.first);
		lastFrame.second = true;
	}
	else {
		0;
		//need to wait until frame is available
	}
	return OK;
}


AVFormatContext* Parser::getFormatContext() {
	return formatContext;
}

AVStream* Parser::getStreamHandle() {
	return videoStream;
}

void Parser::Close() {
	if (isClosed)
		return;
	avformat_close_input(&formatContext);
	
	if (state.enableDumps) {
		if (dumpContext && !(dumpContext->oformat->flags & AVFMT_NOFILE))
			avio_close(dumpContext->pb);
		avformat_free_context(dumpContext);
	}
	av_packet_unref(lastFrame.first);
	delete lastFrame.first;
	isClosed = true;
}
