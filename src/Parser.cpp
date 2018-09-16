#include "Parser.h"
#include "Common.h"

int Parser::Init(ParserParameters& input) {
	state = input;
	int sts = OK;
	sts = avformat_open_input(&formatContext, state.inputFile.c_str(), 0, 0);
	CHECK_STATUS(sts);
	sts = avformat_find_stream_info(formatContext, 0);
	CHECK_STATUS(sts);
	AVCodec* codec;
	sts = av_find_best_stream(formatContext, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
	videoIndex = sts;
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
	//to manage data by myself without "smart" logic inside ffmpeg
	return sts;
}

int Parser::getWidth() {
	return videoStream->codec->width;
}

int Parser::getHeight() {
	return videoStream->codec->height;
}

Parser::Parser() {

}

int Parser::Read() {
	int sts = OK;

	bool videoFrame = false;

	while (videoFrame == false) {
#ifdef TRACER
		nvtxNameOsThread(GetCurrentThreadId(), "DECODE_THREAD");
		nvtxRangePush("Read frame");
		nvtxMark("Reading");
#endif
		sts = av_read_frame(formatContext, lastFrame.first);
#ifdef TRACER
		nvtxRangePop();
#endif
		CHECK_STATUS(sts);
		if ((lastFrame.first)->stream_index != videoIndex)
			continue;

		videoFrame = true;
		currentFrame++;

		//critical section + need to uninit?
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
	avformat_close_input(&formatContext);
	
	if (state.enableDumps) {
		if (dumpContext && !(dumpContext->oformat->flags & AVFMT_NOFILE))
			avio_close(dumpContext->pb);
		avformat_free_context(dumpContext);
	}
	av_packet_unref(lastFrame.first);
	delete lastFrame.first;
}