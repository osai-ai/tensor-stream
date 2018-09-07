#include "VideoProcessor.h"
#include "Common.h"

void SaveRGB24(AVFrame *avFrame, FILE* dump)
{
	uint8_t *RGB = avFrame->data[0];
	for (uint32_t i = 0; i < avFrame->height; i++) {
		fwrite(RGB, avFrame->width * 3, 1, dump);
		RGB += avFrame->linesize[0];
	}
	fflush(dump);

}

int VideoProcessor::Init(VPPParameters* outputFormat) {
	state = outputFormat;

	if (state->enableDumps) {
		dumpFrame = std::shared_ptr<FILE>(fopen("RGB24.yuv", "wb+"));
	}
	
	return OK;
}

int VideoProcessor::Convert(AVFrame* input, AVFrame* output) {
	/*
	Should decide which method call
	*/
	int sts;
	if (state->dstFourCC == NV12) {
		sts = NV12ToRGB24(input, output);
	}

	if (state->enableDumps) {
		SaveRGB24(output, dumpFrame.get());
	}
	return sts;
}