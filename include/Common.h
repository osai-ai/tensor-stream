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
	HIGH, /**< Print also the detailed information about functions in callstack */
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

const std::string logFileName = "logs.txt";

extern std::ofstream logsFile;
extern LogsLevel logsLevel;
extern std::mutex logsMutex;

#define LOG_VALUE(messageIn) \
	{ \
		std::unique_lock<std::mutex> locker(logsMutex); \
		if (logsLevel) \
		{ \
			std::string finalMessage = messageIn + std::string("\n"); \
			if (logsLevel < 0) \
				std::cout << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
			else if (logsFile.is_open()) \
				logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
		} \
	} \

#define START_LOG_FUNCTION(messageIn) \
	{ \
		std::chrono::high_resolution_clock::time_point startFunc; \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logsLevel) \
			{ \
				std::string finalMessage = messageIn + std::string(" +\n"); \
				if (logsLevel < 0) \
					std::cout << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				else if (logsFile.is_open()) \
					logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				if (std::abs(logsLevel) >= MEDIUM) \
					startFunc = std::chrono::high_resolution_clock::now(); \
			} \
		} \

#define END_LOG_FUNCTION(messageOut) \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logsLevel) { \
				std::string finalMessage; \
				if (std::abs(logsLevel) >= MEDIUM) { \
					int timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startFunc).count(); \
					std::string time = std::to_string(timeMs); \
					finalMessage = messageOut + std::string(" -\nFunction time: ") + time + std::string("ms\n\n"); \
				} else { \
					finalMessage = messageOut + std::string(" -\n\n"); \
				} \
				if (logsLevel < 0) \
					std::cout << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				else if (logsFile.is_open()) \
					logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
			} \
		} \
	}

#define START_LOG_BLOCK(messageIn) \
	{ \
		std::chrono::high_resolution_clock::time_point start; \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (std::abs(logsLevel) >= HIGH) \
			{ \
				std::string finalMessage = messageIn + std::string(" +\n"); \
				if (logsLevel < 0) \
					std::cout << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				else if (logsFile.is_open()) \
					logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				start = std::chrono::high_resolution_clock::now(); \
			} \
		} \

#define END_LOG_BLOCK(messageOut) \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (std::abs(logsLevel) >= HIGH) { \
				std::string finalMessage; \
				std::string time = \
				std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count()); \
				finalMessage = messageOut + std::string(" -\ntime: ") + time + std::string(" ms\n"); \
				if (logsLevel < 0) \
					std::cout << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				else if (logsFile.is_open()) \
					logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
			} \
		} \
	}
	
const int maxConsumers = 5;
const int frameRateConstraints = 120;

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