#include <gtest/gtest.h>
#include "VideoProcessor.h"

TEST(VPP_Init, WithoutDumps) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(false), 0);
}