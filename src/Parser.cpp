#include "Parser.h"
#include <thread>
#include <bitset>
#include <numeric>

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
//pair<shift, value>
std::pair<int, int> read_ue(uint8_t* data, int& index, int _shift) {
	int zerosAmount = 0;
	int oldIndex = index;
	while (true) {
		int j = _shift;
		std::bitset<8> valueBin = std::bitset<8>(data[index]);
		while (j < 8 && valueBin[j++] == 0) {
			zerosAmount++;
		}
		if (zerosAmount == 0)
			return std::pair<int, int>(1, 0);
		if (zerosAmount == 8) {
			index++;
			continue;
		}
		//TODO: if it's greater than 8, so we need to take new value
		int shift = 0;
		std::vector<bool> result(zerosAmount);
		while (shift <= zerosAmount) {
			shift++; //the next bit after zero shouldn't be counted
			int bit = (_shift + zerosAmount + shift) % 8;
			result.push_back(valueBin[bit]);
		}
		int field = std::accumulate(result.rbegin(), result.rend(), 0, [](int x, int y) { return (x << 1) + y; });
		return std::pair<int, int>(_shift + zerosAmount * 2 + 1, field);
	}
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
	int bitSize = lastFrame.first->size;
	uint8_t* bitData = lastFrame.first->data;
	int offsetStart = 0;
	int index = 0;
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
			/*
			first_mb_in_slice ue(v)
			slice_type ue(v)
			pic_parameter_set_id ue(v)
			if( separate_colour_plane_flag = = 1 )
				colour_plane_id u(2)
			frame_num u(v) <- we are interested in this variable
			*/
			
		}
		offsetStart++;
	}
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
