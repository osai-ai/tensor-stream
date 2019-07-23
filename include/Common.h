#pragma once
#include "cuda.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include "nvToolsExt.h"

/** @addtogroup cppAPI
@{ 
*/

/** Enum with error codes can be return from TensorStream
*/
enum Internal {
	VREADER_ERROR = -3, /**< Unknown error appeared */
	VREADER_UNSUPPORTED = -2, /**< Requested functionality is unsupported */
	VREADER_REPEAT = -1, /**< Need to repeat last request */
	VREADER_OK = 0 /**< No errors */
};

/** Class with list of modes for logs output
 @details Used in @ref TensorStream::enableLogs() function
*/
enum LogsLevel {
	NONE, /**< No logs are needed */
	LOW, /**< Print the indexes of processed frames */
	MEDIUM, /**< Print also frame processing duration */
	HIGH /**< Print also the detailed information about functions in callstack */
};

/** Class with possible C++ extension module close options
 @details Used in @ref TensorStream::endProcessing() function
*/
enum CloseLevel {
	HARD = 1, /**< Close all opened handlers, free resources */
	SOFT /**< Close all opened handlers except logs file handler, free resources */
};
/**
@}
*/

extern std::mutex logsMutex;

class Logger {
public:
	void initialize(LogsLevel logsLevel, std::string logName = "logs.txt");
	std::string logFileName;
	std::ofstream logsFile;
	LogsLevel logsLevel = LogsLevel::NONE;
	bool enableNVTX = false;
	~Logger();
};

enum NVTXColors {
	GREEN = 0xff00ff00,
	BLUE = 0xff0000ff,
	YELLOW = 0xffffff00,
	PURPLE = 0xffff00ff,
	AQUA = 0xff00ffff,
	RED = 0xffff0000,
	WHITE = 0xffffffff
};

class NVTXTracer {
public:
	void trace(const char* name, NVTXColors colorID) {
		nvtxEventAttributes_t eventAttrib = { 0 };
		eventAttrib.version = NVTX_VERSION;
		eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		eventAttrib.colorType = NVTX_COLOR_ARGB;
		eventAttrib.color = colorID;
		eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
		eventAttrib.message.ascii = name;
		nvtxRangePushEx(&eventAttrib);
	}
	~NVTXTracer() {
		nvtxRangePop();
	}
};

//NVTXTracer should be outside of "if" because it's RAII object
#define PUSH_RANGE(name, colorID) \
	NVTXTracer tracer; \
	if (logger && logger->enableNVTX) \
	{ \
		tracer.trace(name, colorID); \
	} \

#define CHECK_STATUS(status) \
	if (status != 0) { \
		std::cout << "TID: " << std::this_thread::get_id() << " "; \
		std::cout << "Error status != 0, status: " << (status) << "\n" << std::flush; \
		std::cout << "TID: " << std::this_thread::get_id() << " "; \
		std::cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << "\n" << std::flush; \
		return status; \
	} \

#define CHECK_STATUS_THROW(status) \
	if (status != 0) { \
		std::cout << "TID: " << std::this_thread::get_id() << " "; \
		std::cout << "Error status != 0, status: " << (status) << "\n" << std::flush; \
		std::cout << "TID: " << std::this_thread::get_id() << " "; \
		std::cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << "\n" << std::flush; \
		throw std::runtime_error(std::to_string(status)); \
	} \

#define LOG_VALUE(messageIn, neededLevel) \
	{ \
		std::unique_lock<std::mutex> locker(logsMutex); \
		if (logger && logger->logsLevel && std::abs(logger->logsLevel) >= std::abs(neededLevel)) \
		{ \
			std::string finalMessage = messageIn + std::string("\n"); \
			if (logger->logsLevel < 0) \
				std::cout << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
			else if (logger->logsFile.is_open()) \
				logger->logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
		} \
	} \

#define START_LOG_FUNCTION(messageIn) \
	{ \
		std::chrono::high_resolution_clock::time_point startFunc; \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logger && logger->logsLevel) \
			{ \
				std::string finalMessage = messageIn + std::string(" +\n"); \
				if (logger->logsLevel < 0) \
					std::cout << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				else if (logger->logsFile.is_open()) \
					logger->logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				if (std::abs(logger->logsLevel) >= MEDIUM) \
					startFunc = std::chrono::high_resolution_clock::now(); \
			} \
		} \

#define END_LOG_FUNCTION(messageOut) \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logger && logger->logsLevel) { \
				std::string finalMessage; \
				if (std::abs(logger->logsLevel) >= MEDIUM) { \
					int timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startFunc).count(); \
					std::string time = std::to_string(timeMs); \
					finalMessage = messageOut + std::string(" -\nFunction time: ") + time + std::string("ms\n\n"); \
				} else { \
					finalMessage = messageOut + std::string(" -\n\n"); \
				} \
				if (logger->logsLevel < 0) \
					std::cout << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				else if (logger->logsFile.is_open()) \
					logger->logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
			} \
		} \
	}

#define START_LOG_BLOCK(messageIn) \
	{ \
		std::chrono::high_resolution_clock::time_point start; \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logger && std::abs(logger->logsLevel) >= HIGH) \
			{ \
				std::string finalMessage = messageIn + std::string(" +\n"); \
				if (logger->logsLevel < 0) \
					std::cout << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				else if (logger->logsFile.is_open()) \
					logger->logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				start = std::chrono::high_resolution_clock::now(); \
			} \
		} \

#define END_LOG_BLOCK(messageOut) \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logger && std::abs(logger->logsLevel) >= HIGH) { \
				std::string finalMessage; \
				std::string time = \
				std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count()); \
				finalMessage = messageOut + std::string(" -\ntime: ") + time + std::string(" ms\n"); \
				if (logger->logsLevel < 0) \
					std::cout << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				else if (logger->logsFile.is_open()) \
					logger->logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
			} \
		} \
	}

#define SET_CUDA_DEVICE() \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			auto sts = cudaSetDevice(currentCUDADevice); \
			CHECK_STATUS(sts); \
		} \

#define SET_CUDA_DEVICE_THROW() \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			auto sts = cudaSetDevice(currentCUDADevice); \
			CHECK_STATUS_THROW(sts); \
		} \

const int defaultCUDADevice = 0;
const int frameRateConstraints = 240;

template <class T>
T findFree(std::string consumerName, std::vector<std::pair<std::string, T> >& entities) {
	for (auto& item : entities) {
		if (item.first == consumerName) {
			return item.second;
		}
		else if (item.first == "empty") {
			item.first = consumerName;
			return item.second;
		}
	}
	return nullptr;
}