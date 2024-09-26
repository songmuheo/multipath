// config.h

#ifndef CONFIG_H
#define CONFIG_H
#include <string>

#define BUFFER_SIZE 65536
#define FPS 30
#define DELAY_33 33000
#define DELAY_50 50000
#define DELAY_100 100000
#define NODELAY 9999999

const int DELAYS[] = {DELAY_33, DELAY_50, DELAY_100, NODELAY};
const std::string DELAY_LABELS[] = {"33", "50", "100", "no_delay"};


const std::string BASE_FILEPATH = "/home/songmu/";

const std::string CLIENT_FILEPATH = BASE_FILEPATH + "Multipath/cpp/results/client/2024_09_25_15_09/";
const std::string SERVER_FILEPATH = BASE_FILEPATH + "Multipath/cpp/results/server/2024_09_25_15_07/";

// bin files이 저장되어 있는 filepath
const std::string BINS_FILEPATH = SERVER_FILEPATH + "bins/";
// 생성된 frames를 저장할 filepath
const std::string FRAMES_OUT_FILEPATH = SERVER_FILEPATH + "frames/";
// logging csv file이 저장되어 있는 filepath
const std::string CSV_FILEPATH = SERVER_FILEPATH + "logs/";

#endif // CONFIG_H
