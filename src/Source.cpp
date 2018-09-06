#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
// basic file operations
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef _DEBUG
#undef _DEBUG
#include <torch/torch.h>
#include <THC/THC.h>
#include <ATen/ATen.h>
#define _DEBUG
#else
#include <torch/torch.h>
#include <THC/THC.h>
#include <ATen/ATen.h>
#endif
#include "memory"
#include "Parser.h"
#include "Decoder.h"
#include "Common.h"

extern "C"
{
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/hwcontext_cuda.h>
}

void kernel_wrapper(int *a);
unsigned char* change_pixels(AVFrame* src, AVFrame* dst,  CUstream stream);
void test_python(float* test);

FILE* fDump;
FILE* fDumpRGB;
unsigned char* RGB;
static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
	const enum AVPixelFormat *pix_fmts)
{
	const enum AVPixelFormat *p;

	for (p = pix_fmts; *p != -1; p++) {
		if (*p == AV_PIX_FMT_CUDA)
			return *p;
	}

	fprintf(stderr, "Failed to get HW surface format.\n");
	return AV_PIX_FMT_NONE;
}
//#define DUMP_DEMUX

void SaveNV12(AVFrame *avFrame)
{
	uint32_t pitchY = avFrame->linesize[0];
	uint32_t pitchUV = avFrame->linesize[1];

	uint8_t *avY = avFrame->data[0];
	uint8_t *avUV = avFrame->data[1];

	for (uint32_t i = 0; i < avFrame->height; i++) {
		fwrite(avY, avFrame->width, 1, fDump);
		avY += pitchY;
	}

	for (uint32_t i = 0; i < avFrame->height / 2; i++) {
		fwrite(avUV, avFrame->width, 1, fDump);
		avUV += pitchUV;
	}
	fflush(fDump);
}

void SaveRGB24(AVFrame *avFrame)
{
	uint8_t *RGB = avFrame->data[0];
	for (uint32_t i = 0; i < avFrame->height; i++) {
		fwrite(RGB, avFrame->width * 3, 1, fDumpRGB);
		RGB += avFrame->linesize[0];
	}
	fflush(fDumpRGB);

}

void printContext() {
	CUcontext test_cu;
	auto cu_err = cuCtxGetCurrent(&test_cu);
	printf("Context %x\n", test_cu);
}

void test(at::Tensor input) {
	printContext();
	float* data = (float*) input.data_ptr();
	test_python(data);
}

at::Tensor load() {
	//CUdeviceptr data_done;
	//cudaError_t err2 = cudaMalloc(reinterpret_cast<void**>(&data_done), 16 * sizeof(float));
	printContext();
	at::Tensor f = torch::CUDA(at::kByte).tensorFromBlob(reinterpret_cast<void*>(RGB), { 3264 * 608 });
	return f;
}
std::shared_ptr<Parser> parser;
std::shared_ptr<Decoder> decoder;
void start(int max_frames) {
	/*avoiding Tensor CUDA lazy initializing for further context attaching*/
	at::Tensor gt_target = at::empty(at::CUDA(at::kByte), { 1 });
	parser = std::make_shared<Parser>();
	decoder = std::make_shared<Decoder>();
	ParserParameters parserArgs = { "rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4" , true };
	int sts = parser->Init(&parserArgs);
	DecoderParameters decoderArgs = { parser, true };

	sts = decoder->Init(&decoderArgs);
	for (int i = 0; i < 500; i ++)
		sts = parser->Read();

	auto test = std::make_shared<AVPacket>();
	parser->Get(test);
	parser->Get(test);
	parser->Close();

	while (sts = av_read_frame(ifmt_ctx, pkt) >= 0 && max_frames > frame_index) {
		/*
		Get an AVPacket containing encoded data for one AVStream, identified by AVPacket.stream_index (Return the next frame of a audio/video stream)
		This function returns what is stored in the file, and does not validate that what is there are valid frames for the decoder.
		It will split what is stored in the file into frames and return one for each call.
		It will not omit invalid data between valid frames so as to give the decoder the maximum information possible for decoding.
		*/
		printContext();
		if (pkt->stream_index != videoindex) {
			continue;
		}
		in_stream = ifmt_ctx->streams[pkt->stream_index];
#ifdef DUMP_DEMUX
		out_stream = ofmt_ctx_v->streams[0];
#endif
		//printf("Write Video Packet. size:%d\tpts:%lld\tdts:%lld\n", pkt.size, pkt.pts, pkt.dts);
#ifdef DUMP_DEMUX
		//in our output file only 1 stream is available with index 0
		pkt.stream_index = 0;
		//Write
		if (av_write_frame(ofmt_ctx_v, &pkt) < 0) {
			printf("Error muxing packet\n");
			goto end;
		}
		pkt.stream_index = videoindex;
#endif
		//TODO: avctx->active_thread_type & FF_THREAD_FRAME
		AVFrame* outFrame = av_frame_alloc();
		printContext();
		int sts = avcodec_send_packet(decoder_ctx, pkt);
		if (sts < 0 || sts == AVERROR(EAGAIN) || sts == AVERROR_EOF) {
			char err[256];
			printf("%s\n", av_make_error_string(err, 256, sts));
			goto end;
		}
		printContext();
		while (sts >= 0) {
			sts = avcodec_receive_frame(decoder_ctx, outFrame);
			printContext();
			if (sts == AVERROR(EAGAIN) || sts == AVERROR_EOF) {
				char err[256];
				printf("%s\n", av_make_error_string(err, 256, sts));
				av_frame_free(&outFrame);
				goto repeat;
			}
			frame_index++;
			std::cout << "frame: " << decoder_ctx->frame_number << "; pix fmt: " << av_get_pix_fmt_name((AVPixelFormat)outFrame->format)
				<< "; width: " << outFrame->width << "; height: " << outFrame->height << "; pict_type: " << outFrame->pict_type << std::endl;
			AVFrame* sw_frame = av_frame_alloc();
			printContext();
			sw_frame->format = AV_PIX_FMT_NV12;

			if (outFrame->format == AV_PIX_FMT_CUDA) {
				if ((sts = av_hwframe_transfer_data(sw_frame, outFrame, 0)) < 0) {
					fprintf(stderr, "Error transferring the data to system memory\n");
					goto end;
				}
			}

			sts = av_frame_copy_props(sw_frame, outFrame);
			if (sts < 0) {
				av_frame_unref(sw_frame);
				goto end;
			}
			SaveNV12(sw_frame);

			/*need to allocate RGB output frame and dump it via own kernel*/
			AVFrame* rgbFrame = av_frame_alloc();
			rgbFrame->width = outFrame->width;
			rgbFrame->height = outFrame->height;
			rgbFrame->format = AV_PIX_FMT_RGB24;
			sts = av_frame_get_buffer(rgbFrame, 32);
			if (sts < 0)
				goto end;
			printContext();
			//cuCtxPushCurrent(device_hwctx->cuda_ctx); //TODO: why can't take ffmpeg memory without it?
			cudaPointerAttributes attrib;
			cudaPointerAttributes attrib2;
			printContext();
			cudaFree(RGB);
			RGB = change_pixels(outFrame, rgbFrame, device_hwctx->stream);
			printContext();
			//we should use the same stream as ffmpeg for copying data from vid to sys due to conflicts
			//because we must wait until operation competion
			sts = cuStreamSynchronize(device_hwctx->stream);
			printContext();
			//cuCtxPopCurrent(&device_hwctx->cuda_ctx);
			SaveRGB24(rgbFrame);
		}

		av_frame_free(&outFrame);
		av_packet_unref(pkt);

		frame_index++;
	}
	printf("Read frame returned no package\n");
	char error[256];
	printf("%s\n", av_make_error_string(error, 256, sts));

	if (sts < 0) {
		char error[256];
		printf("%s\n", av_make_error_string(error, 256, sts));
		goto end;
	}
#ifdef DUMP_DEMUX
	//Flush buffered data
	av_write_trailer(ofmt_ctx_v);
#endif
end:
	avformat_close_input(&ifmt_ctx);
#ifdef DUMP_DEMUX
	/* close output */
	if (ofmt_ctx_v && !(ofmt_ctx_v->oformat->flags & AVFMT_NOFILE))
		avio_close(ofmt_ctx_v->pb);

	avformat_free_context(ofmt_ctx_v);

	if (sts < 0 && sts != AVERROR_EOF) {
		printf("Error occurred.\n");
		return -1;
	}
	fclose(output_file);
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("load", &load, "load data");
	m.def("get", &start, "get data");
	m.def("test", &test, "change");
}

int main()
{
	start(60);

	return 0;
}
