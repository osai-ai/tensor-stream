#include "Parser.h"
#include "Common.h"

int Parser::Init(ParserParameters* input) {
	state = input;
	int sts = OK;

	sts = avformat_open_input(&formatContext, input->inputFile.c_str(), 0, 0);
	CHECK_STATUS(sts);
	sts = avformat_find_stream_info(formatContext, 0);
	CHECK_STATUS(sts);

	AVCodec* codec;
	sts = av_find_best_stream(formatContext, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
	videoIndex = sts;
	videoStream = formatContext->streams[videoIndex];
	videoStream->codec->codec = codec;

	if (state->enableDumps) {
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

	lastFrame = std::make_pair(std::make_shared<AVPacket>(), false);

	return sts;
}

Parser::Parser() {

}

int Parser::Read() {
	std::shared_ptr<AVPacket> pkt = lastFrame.first;
	bool videoFrame = false;
	int sts = OK;
	while (videoFrame == false) {
		sts = av_read_frame(formatContext, pkt.get());
		CHECK_STATUS(sts);
		if (pkt->stream_index != videoIndex)
			continue;

		videoFrame = true;
		currentFrame++;
		/*
		critical section + need to uninit?
		*/
		lastFrame = std::make_pair(pkt, false);

		if (state->enableDumps) {
			//in our output file only 1 stream is available with index 0
			pkt->stream_index = 0;
			sts = av_write_frame(dumpContext, pkt.get());
			CHECK_STATUS(sts);
			pkt->stream_index = videoIndex;
		}
	}
	return sts;
}

int Parser::Get(std::shared_ptr<AVPacket> output) {
	/*
	Critical section
	*/
	if (lastFrame.second == false) {
		/*
		decoder is responsible for deallocating
		*/
		av_copy_packet(output.get(), lastFrame.first.get());
		lastFrame.second = true;
	}
	else {
		0;
		/*
		need to wait until frame is available
		*/
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
	avformat_close_input(&formatContext);
	
	if (state->enableDumps) {
		if (dumpContext && !(dumpContext->oformat->flags & AVFMT_NOFILE))
			avio_close(dumpContext->pb);
		avformat_free_context(dumpContext);
	}
}