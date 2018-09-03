#pragma once

enum FourCC {
	Y800,
	RGB24,
	NV12
};

/*
The class is used for conversion decoded frame to desired FourCC.
*/
class VPP {
	/*
	Pass raw pointer and input FourCC or use some ffmpeg container? Need to prototype decoder first?
	*/
	int Convert(FourCC input, FourCC output);
};