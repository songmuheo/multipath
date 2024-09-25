// config.h

#ifndef CONFIG_H
#define CONFIG_H
#include <string>

#define BUFFER_SIZE 65536
// Fixed jitter latency(us)
#define DELAY 33000
#define FPS 0.033

const std::string BASE_FILEPATH = "/home/songmu/";

const std::string CLIENT_FILEPATH = BASE_FILEPATH + "Multipath/cpp/results/client/2024_09_24_20_22/";
const std::string SERVER_FILEPATH = BASE_FILEPATH + "Multipath/cpp/results/server/2024_09_24_20_22/";

// bin files이 저장되어 있는 filepath
const std::string BINS_FILEPATH = SERVER_FILEPATH + "bins/";
// 생성된 frames를 저장할 filepath
const std::string FRAMES_OUT_FILEPATH = SERVER_FILEPATH + "frames/";


#endif // CONFIG_H
