#include "Common.h"

std::mutex logsMutex;

void Logger::initialize(LogsLevel logsLevel, std::string logName) {
	this->logsLevel = logsLevel;
	logFileName = logName;
	if (!logsFile.is_open() && logsLevel > 0)
		logsFile.open(logFileName);
}

Logger::~Logger() {
	if (logsFile.is_open()) {
		logsFile.close();
	}
}