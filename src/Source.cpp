#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
// basic file operations
#include <iostream>
#include <fstream>

extern "C"
{
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
}

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
#define DUMP_DEMUX

int main()
{
#ifdef DUMP_DEMUX_OWN
	FILE * dump = fopen("sample_own.h264", "wb");
#endif
	FILE* output_file = fopen("raw.yuv", "w+");
	//no need to allocate if we don't want to add some callbacks or information (e.g. WxH) for raw input
	AVFormatContext *ifmt_ctx = NULL, *ofmt_ctx_v = NULL;

	const char *in_filename = "rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4";
#ifdef DUMP_DEMUX
	const char *out_filename_v = "sample.h264";
#endif
	av_register_all();

	//Open input file and export information from file header (if exists) to AVFormatContext variable
	int sts = avformat_open_input(&ifmt_ctx, in_filename, 0, 0);
	if (sts < 0) {
		printf("Could not open input file.");
		goto end;
	}

	sts = avformat_find_stream_info(ifmt_ctx, 0);
	//if no header, information will be obtained by these call by decoding several frames
	if (sts < 0) {
		printf("Failed to retrieve input stream information");
		goto end;
	}

	int videoindex = -1;
	AVCodec * pVideoCodec;
	sts = av_find_best_stream(ifmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &pVideoCodec, 0);
	if (sts < 0) {
		fprintf(stderr, "Cannot find a video stream in the input file\n");
		return -1;
	}
	videoindex = sts;
	enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
	//AV_PIX_FMT_CUDA
	/*
	//List of available HW devices
	enum AVHWDeviceType type = AV_HWDEVICE_TYPE_NONE;
	while ((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE)
	fprintf(stderr, " %s", av_hwdevice_get_type_name(type));
	*/
	/*
	//Get pix_fmt for approriate HW device
	static enum AVPixelFormat hw_pix_fmt;
	for (int i = 0;; i++) {
		const AVCodecHWConfig *config = avcodec_get_hw_config(pVideoCodec, i);
		if (!config) {
			fprintf(stderr, "Decoder %s does not support device type %s.\n",
				pVideoCodec->name, av_hwdevice_get_type_name(type));
			return -1;
		}
		if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
			config->device_type == type) {
			hw_pix_fmt = config->pix_fmt;
			break;
		}
	}
	*/

	AVCodecContext * decoder_ctx;
	if (!(decoder_ctx = avcodec_alloc_context3(pVideoCodec)))
		return AVERROR(ENOMEM);

	if (avcodec_parameters_to_context(decoder_ctx, ifmt_ctx->streams[videoindex]->codecpar) < 0)
		return -1;

	decoder_ctx->get_format = get_hw_format;
	AVBufferRef *hw_device_ctx = NULL;
	if ((sts = av_hwdevice_ctx_create(&hw_device_ctx, type,
		NULL, NULL, 0)) < 0) {
		fprintf(stderr, "Failed to create specified HW device.\n");
		return sts;
	}
	decoder_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

	if ((sts = avcodec_open2(decoder_ctx, pVideoCodec, NULL)) < 0) {
		fprintf(stderr, "Failed to open codec \n");
		return -1;
	}

#ifdef DUMP_DEMUX 
	//Output
	avformat_alloc_output_context2(&ofmt_ctx_v, NULL, NULL, out_filename_v);
	if (!ofmt_ctx_v) {
		printf("Could not create output context\n");
		sts = AVERROR_UNKNOWN;
		goto end;
	}

	AVStream * outStream = avformat_new_stream(ofmt_ctx_v, pVideoCodec);
	if (avcodec_copy_context(outStream->codec, decoder_ctx) < 0) {
		printf("Failed to copy context from input to output stream codec context\n");
		goto end;
	}

	//Open output file
	if (!(ofmt_ctx_v->oformat->flags & AVFMT_NOFILE)) {
		if (avio_open(&ofmt_ctx_v->pb, out_filename_v, AVIO_FLAG_WRITE) < 0) {
			printf("Could not open output file '%s'", out_filename_v);
			goto end;
		}
	}

	//Write file header
	if (avformat_write_header(ofmt_ctx_v, NULL) < 0) {
		printf("Error occurred when opening video output file\n");
		goto end;
	}
#endif
	//current frame index
	int frame_index = 0;
	/*
	Encoded package. For video, it should typically contain one compressed frame
	*/
repeat:
	AVPacket pkt;
	AVStream *in_stream;
#ifdef DUMP_DEMUX
	AVStream *out_stream;
#endif
	while (sts = av_read_frame(ifmt_ctx, &pkt) >= 0) {
		/*
		Get an AVPacket containing encoded data for one AVStream, identified by AVPacket.stream_index (Return the next frame of a audio/video stream)
		This function returns what is stored in the file, and does not validate that what is there are valid frames for the decoder.
		It will split what is stored in the file into frames and return one for each call.
		It will not omit invalid data between valid frames so as to give the decoder the maximum information possible for decoding.
		*/
		if (pkt.stream_index != videoindex) {
			continue;
		}
		in_stream = ifmt_ctx->streams[pkt.stream_index];
#ifdef DUMP_DEMUX
		out_stream = ofmt_ctx_v->streams[0];
#endif
		printf("Write Video Packet. size:%d\tpts:%lld\tdts:%lld\n", pkt.size, pkt.pts, pkt.dts);
		/*
		Convert PTS/DTS
		The timing information (AVPacket.pts, AVPacket.dts and AVPacket.duration) is in AVStream.time_base units, 
		i.e. it has to be multiplied by the timebase to convert them to seconds.
		PTS/DTS are needed to resolve decode/display order issue - PTS timestamp for display, DTS - timestamp for decode, they are equal in case
		of not B frames
		*/
#ifdef DUMP_DEMUX
		pkt.pts = av_rescale_q_rnd(pkt.pts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
		pkt.dts = av_rescale_q_rnd(pkt.dts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
		pkt.duration = av_rescale_q_rnd(pkt.duration, in_stream->time_base, out_stream->time_base, AV_ROUND_NEAR_INF);
#endif

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
#ifdef DUMP_DEMUX_OWN
		int written = fwrite(pkt.data, sizeof(uint8_t), pkt.size, dump);
		if (written != pkt.size) {
			printf("Error during dumping\n");
			break;
		}
#endif
#if defined(DUMP_DEMUX) || defined(DUMP_DEMUX_OWN) 
		printf("Write %8d frames to output file\n", frame_index);
		frame_index++;
#endif

		//TODO: avctx->active_thread_type & FF_THREAD_FRAME
		AVFrame* outFrame = av_frame_alloc();
		int sts = avcodec_send_packet(decoder_ctx, &pkt);
		if (sts < 0 || sts == AVERROR(EAGAIN) || sts == AVERROR_EOF) {
			char err[256];
			printf("%s\n", av_make_error_string(err, 256, sts));
			goto end;
		}
		while (sts >= 0) {
			sts = avcodec_receive_frame(decoder_ctx, outFrame);
			if (sts == AVERROR(EAGAIN) || sts == AVERROR_EOF) {
				char err[256];
				printf("%s\n", av_make_error_string(err, 256, sts));
				av_frame_free(&outFrame);
				goto repeat;
			}
			std::cout << "frame: " << decoder_ctx->frame_number << "; pix fmt: " << av_get_pix_fmt_name((AVPixelFormat) outFrame->format) << std::endl;
			AVFrame* sw_frame = av_frame_alloc();
			if (outFrame->format == AV_PIX_FMT_CUDA) {
				/* retrieve data from GPU to CPU */
				if ((sts = av_hwframe_transfer_data(sw_frame, outFrame, 0)) < 0) {
					fprintf(stderr, "Error transferring the data to system memory\n");
					goto end;
				}
			}
			int size = av_image_get_buffer_size((AVPixelFormat) sw_frame->format, sw_frame->width,
				sw_frame->height, 1);
			uint8_t* buffer = (uint8_t*) av_malloc(size);
			if (!buffer) {
				fprintf(stderr, "Can not alloc buffer\n");
				sts = AVERROR(ENOMEM);
				goto end;
			}
			sts = av_image_copy_to_buffer(buffer, size,
				(const uint8_t * const *)sw_frame->data,
				(const int *)sw_frame->linesize, (AVPixelFormat)sw_frame->format,
				sw_frame->width, sw_frame->height, 1);
			if (sts < 0) {
				fprintf(stderr, "Can not copy image to buffer\n");
				goto end;
			}
			if ((sts = fwrite(buffer, 1, size, output_file)) < 0) {
				fprintf(stderr, "Failed to dump raw data.\n");
				goto end;
			}
		}

		av_frame_free(&outFrame);
		av_packet_unref(&pkt);
	} 
	if (sts < 0) {
		char err[256];
		printf("%s\n", av_make_error_string(err, 256, sts));
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
#ifdef DUMP_DEMUX_OWN
	fclose(dump);
#endif

	return 0;
}