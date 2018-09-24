#pragma once
#include "cuda.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <iostream>
#include <fstream>

#define TIMINGS

enum {
	REPEAT = -1,
	OK = 0
};

#define CHECK_STATUS(status) if (status != 0) return status;

enum LogsLevel {
	NONE,
	LOW, //start and end with frame indexes
	MEDIUM, //LOW + times
	HIGH, //whole pipeline with times + notices if latency is greater
};

const std::string logFileName = "logs.txt";

static std::ofstream logsFile;
static LogsLevel logsLevel = NONE;
static std::mutex logsMutex;

#define START_LOG_FUNCTION(messageIn) \
	{ \
		clock_t startFunc; \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logsLevel && logsFile.is_open()) \
			{ \
				std::string finalMessage = messageIn + std::string(" +\n"); \
				logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				if (logsLevel >= MEDIUM) \
					startFunc = clock(); \
			} \
		} \

#define END_LOG_FUNCTION(messageOut) \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logsLevel && logsFile.is_open()) { \
				std::string finalMessage; \
				if (logsLevel >= MEDIUM) { \
					int timeMs = clock() - startFunc; \
					std::string time = std::to_string(timeMs); \
					if (timeMs > (realTimeDelay + realTimeDelay / 4)) { \
						finalMessage = messageOut + std::string(" -\nWARNING: Function time: ") + time + std::string("ms\n\n"); \
					} \
					finalMessage = messageOut + std::string(" -\nFunction time: ") + time + std::string("ms\n\n"); \
				} else { \
					finalMessage = messageOut + std::string(" -\n\n"); \
				} \
				logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
			} \
		} \
	}

#define START_LOG_BLOCK(messageIn) \
	{ \
		clock_t start; \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logsLevel >= HIGH && logsFile.is_open()) \
			{ \
				std::string finalMessage = messageIn + std::string(" +\n"); \
				logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
				start = clock(); \
			} \
		} \

#define END_LOG_BLOCK(messageOut) \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logsLevel >= HIGH && logsFile.is_open()) { \
				std::string finalMessage; \
				std::string time = std::to_string(clock() - start); \
				finalMessage = messageOut + std::string(" -\ntime: ") + time + std::string("ms\n"); \
				logsFile << "TID: " << std::this_thread::get_id() << " " << finalMessage << std::flush; \
			} \
		} \
	}
	
const int maxConsumers = 5;

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