#include "Parser.h"
#include "Common.h"

int Parser::Init(ParserParameters& input) {
	state = input;
	int sts;

	sts = avformat_open_input(&formatContext, input.inputFile.c_str(), 0, 0);
	CHECK_STATUS(sts);
	sts = avformat_find_stream_info(formatContext, 0);
	CHECK_STATUS(sts);

	sts = av_find_best_stream(formatContext, AVMEDIA_TYPE_VIDEO, -1, -1, 0, 0);
	CHECK_STATUS(sts);
	videoIndex = sts;
	videoStream = formatContext->streams[videoIndex];
	
	if (state.enableDumps) {
		std::string dumpName = "bitstream.bin";
		sts = avformat_alloc_output_context2(&dumpContext, NULL, NULL, dumpName.c_str());
		CHECK_STATUS(sts);
		AVCodec * videoCodec;
		int sts = av_find_best_stream(formatContext, AVMEDIA_TYPE_VIDEO, -1, -1, &videoCodec, 0);
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

	return sts;
}

AVFormatContext* Parser::getFormatContext() {
	return formatContext;
}

AVStream* Parser::getStreamHandle() {
	return videoStream;
}

void Parser::Close() {
	avformat_close_input(&formatContext);
	framesBuffer.clear();
	if (state.enableDumps) {
		if (dumpContext && !(dumpContext->oformat->flags & AVFMT_NOFILE))
			avio_close(dumpContext->pb);
		avformat_free_context(dumpContext);
	}
}