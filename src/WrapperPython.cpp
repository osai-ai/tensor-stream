#include "Wrapper.h"

static VideoReader reader;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("init", [](std::string rtmp) -> int {
		return reader.initPipeline(rtmp);
	});

	m.def("getPars", []() -> std::map<std::string, int> {
		return reader.getInitializedParams();
	});

	m.def("start", [](void) {
		py::gil_scoped_release release;
		return reader.startProcessing();
	});

	m.def("get", [](std::string name, int delay, int pixelFormat, int dstWidth, int dstHeight) {
		py::gil_scoped_release release;
		return reader.getFrame(name, delay, pixelFormat, dstWidth, dstHeight);
	});

	m.def("dump", [](at::Tensor stream, std::string consumerName) {
		py::gil_scoped_release release;
		AVFrame output;
		output.opaque = stream.data_ptr();
		output.width = stream.size(1);
		output.height = stream.size(0);
		output.channels = stream.size(2);
		std::string dumpName = consumerName + std::string(".yuv");
		std::shared_ptr<FILE> dumpFrame = std::shared_ptr<FILE>(fopen(dumpName.c_str(), "ab+"), std::fclose);
		return reader.dumpFrame(&output, dumpFrame);
	});

	m.def("enableLogs", [](int logsLevel) {
		reader.enableLogs(logsLevel);
	});

	m.def("close", [](int mode) {
		reader.endProcessing(mode);
	});
}