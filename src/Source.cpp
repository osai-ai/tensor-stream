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
}

int main()
{
	//no need to allocate if we don't want to add some callbacks or information (e.g. WxH) for raw input
	AVFormatContext *ifmt_ctx = NULL, *ofmt_ctx_v = NULL;

	const char *in_filename = "rtmp://b.sportlevel.com:22881/mylive/389fb14108f868b2dde44907aaf55622488059da?sign=OTc4Ojk0Nzc5OjE5MDg2NzM0OTU6MTUzNDUyMTg5Mzo4YThkMmNiM2FkMDdmMTg4M2YwYjgxOWMxZWQ3NTY0ZQ==";//"rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4";
	const char *out_filename_v = "sample.h264";

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

	//Output
	avformat_alloc_output_context2(&ofmt_ctx_v, NULL, NULL, out_filename_v);
	if (!ofmt_ctx_v) {
		printf("Could not create output context\n");
		sts = AVERROR_UNKNOWN;
		goto end;
	}
	
	int videoindex = -1;
	for (int i = 0; i < ifmt_ctx->nb_streams; i++) {
		//Create output AVStream according to input AVStream
		AVStream *in_stream = ifmt_ctx->streams[i];
		AVStream *out_stream = NULL;

		if (ifmt_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			videoindex = i;
			out_stream = avformat_new_stream(ofmt_ctx_v, in_stream->codec->codec);
		}
		else {
			continue;
		}

		if (!out_stream) {
			printf("Failed allocating output stream\n");
			sts = AVERROR_UNKNOWN;
			goto end;
		}
		//Copy the settings of AVCodecContext
		if (avcodec_copy_context(out_stream->codec, in_stream->codec) < 0) {
			printf("Failed to copy context from input to output stream codec context\n");
			goto end;
		}
		out_stream->codec->codec_tag = 0;

		if (ofmt_ctx_v->oformat->flags & AVFMT_GLOBALHEADER)
			out_stream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
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

	//current frame index
	int frame_index = 0;
	/*
	Encoded package. For video, it should typically contain one compressed frame
	*/
	AVPacket pkt;
	while (1) {
		AVStream *in_stream, *out_stream;
		/*
		Get an AVPacket containing encoded data for one AVStream, identified by AVPacket.stream_index (Return the next frame of a audio/video stream)
		This function returns what is stored in the file, and does not validate that what is there are valid frames for the decoder.
		It will split what is stored in the file into frames and return one for each call.
		It will not omit invalid data between valid frames so as to give the decoder the maximum information possible for decoding.
		*/
		sts = av_read_frame(ifmt_ctx, &pkt);
		if (sts < 0) {
			char err[256];
			printf("%s\n", av_make_error_string(err, 256, sts));
			break;
		}
		in_stream = ifmt_ctx->streams[pkt.stream_index];


		if (pkt.stream_index == videoindex) {
			out_stream = ofmt_ctx_v->streams[0];
			printf("Write Video Packet. size:%d\tpts:%lld\n", pkt.size, pkt.pts);
		}
		else {
			continue;
		}
		/*
		Convert PTS/DTS
		The timing information (AVPacket.pts, AVPacket.dts and AVPacket.duration) is in AVStream.time_base units, 
		i.e. it has to be multiplied by the timebase to convert them to seconds.
		PTS/DTS are needed to resolve decode/display order issue - PTS timestamp for display, DTS - timestamp for decode, they are equal in case
		of not B frames
		*/
		pkt.pts = av_rescale_q_rnd(pkt.pts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
		pkt.dts = av_rescale_q_rnd(pkt.dts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
		pkt.duration = av_rescale_q_rnd(pkt.duration, in_stream->time_base, out_stream->time_base, AV_ROUND_NEAR_INF);
		//in our output file only 1 stream is available with index 0
		pkt.stream_index = 0;
		//Write
		if (av_write_frame(ofmt_ctx_v, &pkt) < 0) {
			printf("Error muxing packet\n");
			break;
		}

		printf("Write %8d frames to output file\n", frame_index);
		av_packet_unref(&pkt);
		frame_index++;
	}
	//Flush buffered data
	av_write_trailer(ofmt_ctx_v);
end:
	avformat_close_input(&ifmt_ctx);
	/* close output */
	if (ofmt_ctx_v && !(ofmt_ctx_v->oformat->flags & AVFMT_NOFILE))
		avio_close(ofmt_ctx_v->pb);

	avformat_free_context(ofmt_ctx_v);

	if (sts < 0 && sts != AVERROR_EOF) {
		printf("Error occurred.\n");
		return -1;
	}
	return 0;
}