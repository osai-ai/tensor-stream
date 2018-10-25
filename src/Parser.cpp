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
	do {
		int remainder = value % 2;
		result.insert(result.begin(), remainder);
		value /= 2;
	} while (value);
	//need to allign data
	while (result.size() != 8) {
		result.insert(result.begin(), 0);
	}
	return result;
}

bool BitReader::findNAL() {
	int value;
	while (byteIndex != dataSize) {
		if ((value = Convert(ReadBits(8), BitReader::Base::DEC)) == 0) {
			int startCodeCounter = 1;
			while ((value = Convert(ReadBits(8), BitReader::Base::DEC)) == 0) {
				startCodeCounter++;
			}
			if (startCodeCounter >= 2 && value == 1) {
				return true;
			}
		}
	}
	return false;
}

std::vector<bool> BitReader::FindNALType() {
	std::vector<bool> nal_unit_type;
	if (findNAL()) {
		SkipBits(1); //forbidden_zero_bit
		SkipBits(2); //nal_ref_idc
		nal_unit_type = ReadBits(5);
	}
	return nal_unit_type;
}

bool BitReader::SkipBits(int number) {
	int bytes = (shiftInBits + number) / 8;
	if (byteIndex + bytes >= dataSize)
		return false;
	byteIndex += bytes;
	shiftInBits = (shiftInBits + number) % 8;
	return true;
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
		//result.insert(result.begin(), value[i % 8]);
		result.push_back(value[i % 8]);
		shiftInBits++;
	}
	if (shiftInBits == 8) {
		shiftInBits = 0;
		byteIndex++;
	}
	return result;
}

int BitReader::Convert(std::vector<bool> value, Base base) {
	int result = 0;
	switch (base) {
		case Base::DEC: 
		{
			int n = 0;
			for (int i = value.size() - 1; i > 0; i--)
			{
				if (value[i])
				{
					result += pow(2, n);
				}
				n++;
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

std::vector<bool> BitReader::ReadGolomb() {
	int zerosNumber = 0;
	while (Convert(ReadBits(1), Base::DEC) == 0) {
		zerosNumber++;
	}
	return ReadBits(zerosNumber);
}


bool BitReader::SkipGolomb() {
	int zerosNumber = 0;
	while (Convert(ReadBits(1), Base::DEC) == 0) {
		zerosNumber++;
	}
	return SkipBits(zerosNumber);
}

int Parser::Analyze(AVPacket* package) {
	enum NALTypes {
		UNKNOWN = 0,
		SPS = 7,
		PPS = 8,
		SEI = 6,
		SLICE_IDR = 5,
		SLICE_NOT_IDR = 1
	} NALType = UNKNOWN;
/*	for (int i = 0; i < package->size; i++) {
		std::cout << std::hex << (int) package->data[i] << std::flush;
	}
*/
	BitReader bitReader(package->data, package->size);
	int separate_colour_plane_flag = 0; //should be parsed from SPS for frame_num
	int log2_max_frame_num_minus4 = 0; //should be parsed from SPS because it's size of frame_num
	//We need to find SLICE_*
	while (NALType != SLICE_IDR && NALType != SLICE_NOT_IDR) {
		NALType = static_cast<NALTypes>(bitReader.Convert(bitReader.FindNALType(), BitReader::Base::DEC));
		//we have to find log2_max_frame_num_minus4
		if (NALType == SPS) {
			int profile_idc = bitReader.Convert(bitReader.ReadBits(8), BitReader::Base::DEC);
			bitReader.SkipBits(8); //reserved
			bitReader.SkipBits(8); //level_idc
			bitReader.SkipGolomb(); //seq_parameter_set_id
			if (profile_idc == 100 || profile_idc == 110 ||
				profile_idc == 122 || profile_idc == 244 || profile_idc == 44 ||
				profile_idc == 83 || profile_idc == 86 || profile_idc == 118 ||
				profile_idc == 128 || profile_idc == 138 || profile_idc == 139 ||
				profile_idc == 134 || profile_idc == 135) {
				int chroma_format_idc = bitReader.Convert(bitReader.ReadGolomb(), BitReader::Base::DEC);
				if (chroma_format_idc == 3)
					separate_colour_plane_flag = bitReader.Convert(bitReader.FindNALType(), BitReader::Base::DEC);
				bitReader.SkipGolomb(); //bit_depth_luma_minus8
				bitReader.SkipGolomb(); //bit_depth_chroma_minus8
				bitReader.SkipBits(1); //qpprime_y_zero_transform_bypass_flag
				int seq_scaling_matrix_present_flag = bitReader.Convert(bitReader.ReadBits(1), BitReader::Base::DEC);
				if (seq_scaling_matrix_present_flag) {
					for (int i = 0; i < ((chroma_format_idc != 3) ? 8 : 12); i++)
						bitReader.SkipBits(1); //seq_scaling_list_present_flag[i]
				}
				log2_max_frame_num_minus4 = bitReader.Convert(bitReader.ReadGolomb(), BitReader::Base::DEC);
			}
		}
	}
	if (NALType == SLICE_IDR || NALType == SLICE_NOT_IDR) {
		//here we have position after NAL header
		int first_mb_in_slice = bitReader.Convert(bitReader.ReadGolomb(), BitReader::Base::DEC);
		bitReader.SkipGolomb(); //slice_type
		bitReader.SkipGolomb(); //pic_parameter_set_id
		if (separate_colour_plane_flag == 1)
			bitReader.SkipBits(2);
		int frame_num = bitReader.Convert(bitReader.ReadBits(log2_max_frame_num_minus4 + 4), BitReader::Base::DEC);
		if (frame_num >= frameNumValue + 1)
			printf("FRAME ERROR\n");
		frameNumValue = frame_num;
	}
	return OK;
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
