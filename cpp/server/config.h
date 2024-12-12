// server/config.h

#ifndef CONFIG_H
#define CONFIG_H

#include <string>

const char* SERVER_IP = "121.128.220.205";

const int SERVER_PORT1 = 12345; //lg
const int SERVER_PORT2 = 12346; //kt

const int BUFFER_SIZE = 65536;

const std::string BASE_FILEPATH = "/home/songmu/";

// Path to save bin files(received packet)
const std::string FILEPATH_TO_SAVE_RAW_PACKET = BASE_FILEPATH + "Multipath/cpp/results/server";

const std::string FILEPATH_FRAME = BASE_FILEPATH + "Multipath/cpp/results/server/frames";
const std::string FILEPATH_LOG = BASE_FILEPATH + "Multipath/cpp/results/server/logs";

const uint64_t PLAY_DELAY_MS = 50;


#endif // CONFIG_H