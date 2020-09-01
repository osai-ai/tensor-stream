#include "WrapperC.h"
#include <experimental/filesystem> //C++17 
#include "SDL.h"
#undef main

TensorStream reader;
SDL_Texture* bmp;
SDL_Renderer* renderer;

int dstWidth = 1920;
int dstHeight = 1080;

void get_cycle(FrameParameters frameParameters, std::map<std::string, std::string> executionParameters) {
	try {
		int frames = std::atoi(executionParameters["frames"].c_str());
		if (!frames)
			return;

		for (int i = 0; i < frames; i++) {
			auto result = reader.getFrame<unsigned char>(executionParameters["name"], { std::atoi(executionParameters["delay"].c_str()) }, frameParameters);
			uint8_t* resultCPU = new uint8_t[(int)(dstWidth * dstHeight * 1.5)];
			int sts = cudaMemcpy(resultCPU, std::get<0>(result), sizeof(uint8_t) * dstWidth * dstHeight * 1.5, cudaMemcpyDeviceToHost);
			{

				SDL_UpdateTexture(bmp, NULL, resultCPU, dstWidth);

				SDL_RenderClear(renderer);
				SDL_RenderCopy(renderer, bmp, NULL, NULL);
				SDL_RenderPresent(renderer);

			}
			cudaFree(std::get<0>(result));
		}
	}
	catch (std::runtime_error e) {
		return;
	}
}

int main() {
	{
		SDL_Init(SDL_INIT_VIDEO);
		SDL_Window * screen = SDL_CreateWindow("Testing..", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, dstWidth, dstHeight, SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI);
		renderer = SDL_CreateRenderer(screen, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE);
		bmp = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_NV12, SDL_TEXTUREACCESS_STREAMING, dstWidth, dstHeight);
	}
	//reader.enableLogs(HIGH);
	//reader.enableNVTX();
	int sts = VREADER_OK;
	int initNumber = 10;

	while (initNumber--) {
		sts = reader.initPipeline("rtmp://streaming.sportlevel.com/relay/playerlpch2", 5, 0, 5, FrameRateMode::NATIVE);
		if (sts != VREADER_OK)
			reader.endProcessing();
		else
			break;
	}

	reader.skipAnalyzeStage();
	CHECK_STATUS(sts);
	std::thread pipeline([] { reader.startProcessing(); });
	ColorOptions colorOptions = { FourCC::NV12 };
	colorOptions.planesPos = Planes::PLANAR;
	colorOptions.normalization = false;
	ResizeOptions resizeOptions = { dstWidth, dstHeight };
	FrameParameters frameParameters = { resizeOptions, colorOptions };

	std::map<std::string, std::string> executionParameters = { {"name", "first"}, {"delay", "0"}, {"frames", "500000"} };
	std::thread get(get_cycle, frameParameters, executionParameters);
	get.join();
	reader.endProcessing();
	pipeline.join();
	return 0;
}
