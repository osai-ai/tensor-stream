#include "Parser.h"
#include <thread>
#include <bitset>
#include <numeric>

BitReader::BitReader(uint8_t* _byteData, int _dataSize) {
	byteData = _byteData;
	dataSize = _dataSize;
}

BitReader::BitReader() {
	byteData = nullptr;
	dataSize = 0;
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
	if (shiftInBits != 0) {
		shiftInBits = 0;
		byteIndex++;
	}
	while (byteIndex != dataSize) {
		if ((value = Convert(ReadBits(8), Type::RAW, BitReader::Base::DEC)) == 0) {
			int startCodeCounter = 1;
			while ((value = Convert(ReadBits(8), Type::RAW, BitReader::Base::DEC)) == 0) {
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
	//getVector returns vector where most significant bit is placed to zero index (just read from memory bits and push back to vector), next cycle 
	//re-order vector as it should be (the less significant bit is placed to zero index)
	std::vector<bool> value = getVector(byteData[byteIndex]);
	for (int i = startIndex; i < endIndex; i++) {
		//we read we last bit, need to take next byte
		if (i && i % 8 == 0) {
			shiftInBits = 0;
			byteIndex++;
			value = getVector(byteData[byteIndex]);
		}
		result.insert(result.begin(), value[i % 8]);
		shiftInBits++;
	}
	if (shiftInBits == 8) {
		shiftInBits = 0;
		byteIndex++;
	}
	return result;
}

int BitReader::Convert(std::vector<bool> value, Type type, Base base) {
	int result = 0;
	switch (base) {
		case Base::DEC: 
		{
			int n = 0;
			for (int i = 0; i < value.size(); i++)
			{
				if (value[i])
				{
					result += pow(2, n);
				}
				n++;
			}
			if (type == Type::GOLOMB) {
				result = pow(2, value.size()) - 1 + result;
			} else if (type == Type::SGOLOMB) {
				result = pow(2, value.size()) - 1 + result;
				result = pow(-1, result + 1) * ceil(result / 2);
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
	while (Convert(ReadBits(1), Type::RAW, Base::DEC) == 0) {
		zerosNumber++;
	}
	return ReadBits(zerosNumber);
}


bool BitReader::SkipGolomb() {
	int zerosNumber = 0;
	while (Convert(ReadBits(1), Type::RAW, Base::DEC) == 0) {
		zerosNumber++;
	}
	return SkipBits(zerosNumber);
}

int Parser::Analyze(AVPacket* package) {
	PUSH_RANGE("Parser::Analyze", NVTXColors::AQUA);
	enum NALTypes {
		UNKNOWN = 0,
		SPS = 7,
		PPS = 8,
		SEI = 6,
		SLICE_IDR = 5,
		SLICE_NOT_IDR = 1
	} NALType = UNKNOWN;
	int errorBitstream = AnalyzeErrors::NONE;
	av_bitstream_filter_filter(bitstreamFilter, formatContext->streams[videoIndex]->codec, NULL, &NALu->data, &NALu->size, package->data, package->size, 0);
	//content in package is already in h264 format, so no need to do mp4->h264 conversion
	if (NALu->data == nullptr) {
		NALu->data = package->data;
		NALu->size = package->size;
	}
	BitReader bitReader(NALu->data, NALu->size);
	//should be saved as SPS parameters
	static int separate_colour_plane_flag = 0; //should be parsed from SPS for frame_num
	static int log2_max_frame_num_minus4 = 0; //should be parsed from SPS because it's size of frame_num
	static int pic_order_cnt_type = 0;
	static int frame_mbs_only_flag = 0;
	static int log2_max_pic_order_cnt_lsb_minus4 = 0;
	static int gaps_in_frame_num_value_allowed_flag = 0;
	//We need to find SLICE_*
	while (NALType != SLICE_IDR && NALType != SLICE_NOT_IDR) {
		NALType = static_cast<NALTypes>(bitReader.Convert(bitReader.FindNALType(), BitReader::Type::RAW, BitReader::Base::DEC));
		if (NALType == UNKNOWN)
			return VREADER_REPEAT;
		//we have to find log2_max_frame_num_minus4
		if (NALType == SPS) {
			int profile_idc = bitReader.Convert(bitReader.ReadBits(8), BitReader::Type::RAW, BitReader::Base::DEC);
			bitReader.SkipBits(8); //reserved
			int level_idc = bitReader.Convert(bitReader.ReadBits(8), BitReader::Type::RAW, BitReader::Base::DEC); //level_idc
			int seq_parameter_set_id = bitReader.Convert(bitReader.ReadGolomb(), BitReader::Type::GOLOMB, BitReader::Base::DEC); //seq_parameter_set_id
			if (profile_idc == 100 || profile_idc == 110 ||
				profile_idc == 122 || profile_idc == 244 || profile_idc == 44 ||
				profile_idc == 83 || profile_idc == 86 || profile_idc == 118 ||
				profile_idc == 128 || profile_idc == 138 || profile_idc == 139 ||
				profile_idc == 134 || profile_idc == 135) {
				int chroma_format_idc = bitReader.Convert(bitReader.ReadGolomb(), BitReader::Type::GOLOMB, BitReader::Base::DEC);
				if (chroma_format_idc == 3)
					separate_colour_plane_flag = bitReader.Convert(bitReader.FindNALType(), BitReader::Type::RAW, BitReader::Base::DEC);
				bitReader.SkipGolomb(); //bit_depth_luma_minus8
				bitReader.SkipGolomb(); //bit_depth_chroma_minus8
				bitReader.SkipBits(1); //qpprime_y_zero_transform_bypass_flag
				int seq_scaling_matrix_present_flag = bitReader.Convert(bitReader.ReadBits(1), BitReader::Type::RAW, BitReader::Base::DEC);
				if (seq_scaling_matrix_present_flag) {
					for (int i = 0; i < ((chroma_format_idc != 3) ? 8 : 12); i++)
						bitReader.SkipBits(1); //seq_scaling_list_present_flag[i]
				}
			}
			else {
				LOG_VALUE(std::string("[PARSING] Bitstream doesn't conform to the Main profile ") + std::to_string(profile_idc), LogsLevel::LOW);
			}
			log2_max_frame_num_minus4 = bitReader.Convert(bitReader.ReadGolomb(), BitReader::Type::GOLOMB, BitReader::Base::DEC);
			pic_order_cnt_type = bitReader.Convert(bitReader.ReadGolomb(), BitReader::Type::GOLOMB, BitReader::Base::DEC);
			if (pic_order_cnt_type == 0) {
				log2_max_pic_order_cnt_lsb_minus4 = bitReader.Convert(bitReader.ReadGolomb(), BitReader::Type::GOLOMB, BitReader::Base::DEC);
			}
			else if (pic_order_cnt_type == 1) {
				bitReader.SkipBits(1); //delta_pic_order_always_zero_flag
				bitReader.SkipGolomb(); //offset_for_non_ref_pic
				bitReader.SkipGolomb(); //offset_for_top_to_bottom_field
				int num_ref_frames_in_pic_order_cnt_cycle = bitReader.Convert(bitReader.ReadGolomb(), BitReader::Type::GOLOMB, BitReader::Base::DEC);
				for (int i = 0; i < num_ref_frames_in_pic_order_cnt_cycle; i++)
					bitReader.SkipGolomb(); //offset_for_ref_frame
			}
			bitReader.SkipGolomb(); //max_num_ref_frames
			gaps_in_frame_num_value_allowed_flag = bitReader.Convert(bitReader.ReadBits(1), BitReader::Type::RAW, BitReader::Base::DEC);
			//it's very rare scenario with pretty tricky handling logic, so for now message with warning is throwing
			if (gaps_in_frame_num_value_allowed_flag) {
				LOG_VALUE(std::string("[PARSING] Field gaps_in_frame_num_value_allowed_flag is unexpected != 0"), LogsLevel::LOW);
				errorBitstream = errorBitstream | AnalyzeErrors::GAPS_FRAME_NUM;
			}
			bitReader.SkipGolomb(); //pic_width_in_mbs_minus1
			bitReader.SkipGolomb(); //pic_height_in_map_units_minus1
			frame_mbs_only_flag = bitReader.Convert(bitReader.ReadBits(1), BitReader::Type::RAW, BitReader::Base::DEC);
		}
	}
	if (NALType == SLICE_IDR || NALType == SLICE_NOT_IDR) {
		//here we have position after NAL header
		int first_mb_in_slice = bitReader.Convert(bitReader.ReadGolomb(), BitReader::Type::GOLOMB, BitReader::Base::DEC);
		//we want analyze only first slice in frame because from frame drop perspective there is no difference between slices
		//btw we should hit only first slice due to return after 1 slice
		if (first_mb_in_slice)
			return VREADER_OK;
		int slice_type = bitReader.Convert(bitReader.ReadGolomb(), BitReader::Type::GOLOMB, BitReader::Base::DEC);
		bitReader.SkipGolomb(); //pic_parameter_set_id
		if (separate_colour_plane_flag == 1)
			bitReader.SkipBits(2);
		int frame_num = bitReader.Convert(bitReader.ReadBits(log2_max_frame_num_minus4 + 4), BitReader::Type::RAW, BitReader::Base::DEC);
		if (!frame_mbs_only_flag) {
			int field_pic_flag = bitReader.Convert(bitReader.ReadBits(1), BitReader::Type::RAW, BitReader::Base::DEC);
			if (field_pic_flag)
				bitReader.SkipBits(1); //bottom_field_flag
		}
		int idrPicFlag = ((NALType == SLICE_IDR) ? 1 : 0);
		if (idrPicFlag) {
			bitReader.SkipGolomb(); //idr_pic_id
		}
		//we expect frame_num == 0 at the start of GOP (for any IDR)
		//also frame_num has maximum size
		if (idrPicFlag || frameNumValue == pow(2, log2_max_frame_num_minus4 + 4) - 1) {
			frameNumValue = -1;
		}
		int pic_order_cnt_lsb = 0;
		if (pic_order_cnt_type == 0) {
			pic_order_cnt_lsb = bitReader.Convert(bitReader.ReadBits(log2_max_pic_order_cnt_lsb_minus4 + 4), BitReader::Type::RAW, BitReader::Base::DEC);
		}
		if (POC == pow(2, log2_max_pic_order_cnt_lsb_minus4 + 4) - 1) {
			POC = 0;
		}
		if (gaps_in_frame_num_value_allowed_flag == 0) {
			if (frame_num == frameNumValue) {
				if (pic_order_cnt_lsb <= POC) {
					LOG_VALUE(std::string("[PARSING] B-slice incorrect POC. Current POC: ") + std::to_string(pic_order_cnt_lsb)
						+ std::string(" previous POC: ") + std::to_string(POC), LogsLevel::LOW);
					errorBitstream = errorBitstream | AnalyzeErrors::B_POC;
				}
			}
			else if (frame_num != frameNumValue + 1) {
				LOG_VALUE(std::string("[PARSING] frame_num is incorrect. Current frame_num: ") + std::to_string(frame_num)
					+ std::string(" previous frame_num: ") + std::to_string(frameNumValue), LogsLevel::LOW);
				errorBitstream = errorBitstream | AnalyzeErrors::FRAME_NUM;
			}
			
		}

		frameNumValue = frame_num;
		POC = pic_order_cnt_lsb;
	}

	av_freep(&NALu->data);
	av_packet_free_side_data(NALu);
	av_free_packet(NALu);
	return errorBitstream;
}

int interruptCallback(void *ctx) {
	//TODO: In the newest FFmpeg version this callback is called during context destroying, it can be bug but need keep in mind
	if (timeoutFrame < 0)
		return 0;
	AVFormatContext* formatContext = reinterpret_cast<AVFormatContext*>(ctx);
	if (formatContext->opaque == nullptr)
		return 0;

	std::chrono::time_point<std::chrono::system_clock> frameTime = *(std::chrono::time_point<std::chrono::system_clock>*)formatContext->opaque;
	std::chrono::time_point<std::chrono::system_clock> currentTime = std::chrono::system_clock::now();
	int duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - frameTime).count();
	if (duration > timeoutFrame)
		return -1;
	// do something
	return 0;
}

int Parser::Init(ParserParameters& input, std::shared_ptr<Logger> logger) {
	PUSH_RANGE("Parser::Init", NVTXColors::AQUA);
	state = input;
	int sts = VREADER_OK;
	this->logger = logger;
	//packet_buffer - isn't empty
	AVDictionary *opts = 0;
	av_dict_set(&opts, "rtsp_transport", "tcp", 0);
	formatContext = avformat_alloc_context();
	const AVIOInterruptCB intCallback = { interruptCallback, formatContext };
	formatContext->interrupt_callback = intCallback;
	latestFrameTimestamp = std::chrono::system_clock::now();
	formatContext->opaque = &latestFrameTimestamp;
	sts = avformat_open_input(&formatContext, state.inputFile.c_str(), 0, &opts);
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
	NALu = new AVPacket();
	av_init_packet(NALu);

	bitstreamFilter = av_bitstream_filter_init("h264_mp4toannexb");

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

int Parser::readVideoFrame(std::pair<AVPacket*, bool>& dst) {
	int sts = VREADER_OK;
	bool videoFrame = false;
	while (!videoFrame) {
		sts = av_read_frame(formatContext, dst.first);
		CHECK_STATUS(sts);
		if (dst.first->stream_index != videoIndex) {
			av_packet_unref(dst.first);
			continue;
		}
		else {
			videoFrame = true;
		}
	}
	dst.second = false;
	return sts;
}

//no need any sync due to executing in 1 thread only
int Parser::Read() {
	PUSH_RANGE("Parser::Read", NVTXColors::AQUA);
	int sts = VREADER_OK;
	//if no frames specified, just read the last one and add to begin of the buffer
	sts = readVideoFrame(lastFrame);
	latestFrameTimestamp = std::chrono::system_clock::now();
	formatContext->opaque = &latestFrameTimestamp;
	CHECK_STATUS(sts);
	currentFrame++;

	if (state.enableDumps) {
		//we avoid frames which was read but haven't been used anymore later
		//in our output file only 1 stream is available with index == 0 (it's video track)
		lastFrame.first->stream_index = 0;
		sts = av_write_frame(dumpContext, lastFrame.first);
		CHECK_STATUS(sts);
		lastFrame.first->stream_index = videoIndex;
	}

	return sts;
}


//no need any sync due to executing in 1 thread only
int Parser::Get(AVPacket* output) {
	PUSH_RANGE("Parser::Get", NVTXColors::AQUA);
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
	return VREADER_OK;
}


AVFormatContext* Parser::getFormatContext() {
	return formatContext;
}

AVStream* Parser::getStreamHandle() {
	return videoStream;
}

void Parser::Close() {
	PUSH_RANGE("Parser::Close", NVTXColors::AQUA);
	if (isClosed)
		return;
	av_bitstream_filter_close(bitstreamFilter);
	avformat_close_input(&formatContext);
	
	if (state.enableDumps) {
		if (dumpContext && !(dumpContext->oformat->flags & AVFMT_NOFILE))
			avio_close(dumpContext->pb);
		avformat_free_context(dumpContext);
	}
	av_packet_unref(lastFrame.first);
	delete lastFrame.first;

	av_packet_unref(NALu);
	delete NALu;

	isClosed = true;
}
