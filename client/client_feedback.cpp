// client_feedback.cpp
#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>
#include <vector>
#include <future>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <arpa/inet.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <unistd.h>
#include <filesystem>
#include <cerrno>
#include <cstring>
#include "config.h"  // WIDTH, HEIGHT, FPS, SERVER_IP, SERVER_PORT 등 매크로 정의

// OpenSSL
#include <openssl/hmac.h>
#include <openssl/evp.h>

// FFmpeg
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

// pjnath, pjlib
extern "C" {
#include <pjlib.h>
#include <pjlib-util.h>
#include <pjnath.h>
}

using namespace std;
namespace fs = std::filesystem;

class BufferedLogger {
public:
    BufferedLogger(const string& filepath) {
        log_stream.open(filepath, ios::out | ios::app);
        if (!log_stream.is_open()) {
            throw runtime_error("Failed to open log file: " + filepath);
        }
    }
    ~BufferedLogger() {
        flush();
        log_stream.close();
    }
    void log(const string& msg) {
        lock_guard<mutex> lock(m);
        log_stream << msg << "\n";
    }
    void flush() {
        lock_guard<mutex> lock(m);
        log_stream.flush();
    }
private:
    ofstream log_stream;
    mutex m;
};

struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};

std::string generate_turn_username(const std::string& identifier, uint32_t validSeconds) {
    uint64_t expiration = chrono::duration_cast<chrono::seconds>(
        chrono::system_clock::now().time_since_epoch()).count() + validSeconds;
    return to_string(expiration) + ":" + identifier;
}
// std::string compute_turn_password(const std::string& data, const std::string& secret) {
//     unsigned char hmac_result[EVP_MAX_MD_SIZE] = {0};
//     unsigned int hmac_length = 0;
//     HMAC(EVP_sha1(), secret.c_str(), secret.size(),
//          reinterpret_cast<const unsigned char*>(data.c_str()), data.size(),
//          hmac_result, &hmac_length);
//     ostringstream oss;
//     for(unsigned i=0; i<hmac_length;i++){
//         oss<< hex<< setw(2)<< setfill('0')<< (int)hmac_result[i];
//     }
//     return oss.str();
// }
std::string compute_turn_password_bin(const std::string& data, const std::string& secret) {
    unsigned char hmac_result[EVP_MAX_MD_SIZE];
    unsigned int hmac_length = 0;
    memset(hmac_result, 0, sizeof(hmac_result));

    HMAC(EVP_sha1(),
         secret.c_str(), secret.size(),
         reinterpret_cast<const unsigned char*>(data.c_str()), data.size(),
         hmac_result, &hmac_length);

    // raw binary
    return std::string(reinterpret_cast<const char*>(hmac_result), hmac_length);
}

// TURN 콜백
static pj_bool_t on_data_sent_cb(pj_turn_sock *sock, pj_ssize_t sent)
{
    if(sent<0) {
        pj_status_t err= (pj_status_t)(-sent);
        PJ_LOG(1,("TURN","Data send err: %d", err));
    } else {
        PJ_LOG(4,("TURN","Data sent: %zd bytes",(ssize_t)sent));
    }
    return PJ_TRUE;
}
static void on_rx_data_cb(pj_turn_sock *turn_sock,
                          void *pkt, unsigned pkt_len,
                          const pj_sockaddr_t *peer_addr,
                          unsigned addr_len)
{
    PJ_LOG(4,("TURN","on_rx_data: got %u bytes", pkt_len));
}
static pj_status_t on_connection_attempt_cb(pj_turn_sock *turn_sock,
                                            pj_uint32_t conn_id,
                                            const pj_sockaddr_t *peer_addr,
                                            unsigned addr_len)
{
    PJ_LOG(3,("TURN","TCP connection attempt (RFC6062)"));
    return PJ_SUCCESS;
}
static void on_connection_status_cb(pj_turn_sock *turn_sock,
                                    pj_status_t status,
                                    pj_uint32_t conn_id,
                                    const pj_sockaddr_t *peer_addr,
                                    unsigned addr_len)
{
    PJ_LOG(3,("TURN","Connection status: %d",status));
}
static void on_state_cb(pj_turn_sock *turn_sock,
                        pj_turn_state_t old_state,
                        pj_turn_state_t new_state)
{
    PJ_LOG(3,("TURN","state changed: %d->%d",old_state,new_state));
    if(new_state== PJ_TURN_STATE_READY){
        pj_turn_session_info info;
        if(pj_turn_sock_get_info(turn_sock,&info)== PJ_SUCCESS){
            char relay_ip[64];
            if(info.relay_addr.ipv4.sin_family== PJ_AF_INET){
                pj_inet_ntop(PJ_AF_INET,
                             &info.relay_addr.ipv4.sin_addr,
                             relay_ip,sizeof(relay_ip));
                pj_uint16_t rport= pj_ntohs(info.relay_addr.ipv4.sin_port);
                PJ_LOG(3,("TURN","Relay(IPv4):%s:%d", relay_ip,rport));
            } else {
                char bufv6[64];
                pj_inet_ntop(PJ_AF_INET6,
                             &info.relay_addr.ipv6.sin6_addr,
                             bufv6,sizeof(bufv6));
                pj_uint16_t rport= pj_ntohs(info.relay_addr.ipv6.sin6_port);
                PJ_LOG(3,("TURN","Relay(IPv6):%s:%d", bufv6,rport));
            }
        }
    }
}

static pj_turn_sock_cb turn_callbacks= {
    &on_rx_data_cb,
    &on_data_sent_cb,
    &on_state_cb,
    &on_connection_attempt_cb,
    &on_connection_status_cb
};

class VideoStreamer {
public:
    // 생성자
    VideoStreamer(): frame_counter(0), sequence_number(0){
        create_and_set_output_folders();

        string logfile= logs_folder+"/packet_log.csv";
        logger= make_unique<BufferedLogger>(logfile);
        logger->log("sequence_number,pts,size,timestamp_frame,timestamp_sending,encoding_latency,frame_type");

        codec= avcodec_find_encoder_by_name("libx264");
        if(!codec) throw runtime_error("Codec 'libx264' not found");

        codec_ctx.reset(avcodec_alloc_context3(codec));
        if(!codec_ctx) throw runtime_error("Could not alloc ctx");
        codec_ctx->width= WIDTH;
        codec_ctx->height= HEIGHT;
        codec_ctx->time_base= {1,FPS};
        codec_ctx->framerate= {FPS,1};
        codec_ctx->pix_fmt= AV_PIX_FMT_YUV420P;
        codec_ctx->max_b_frames=0;
        codec_ctx->gop_size=10;

        AVDictionary* opt= nullptr;
        av_dict_set(&opt, "preset","veryfast",0);
        av_dict_set(&opt,"tune","zerolatency",0);
        av_dict_set(&opt,"crf","26",0);
        string xparam= "keyint=10:min-keyint=10:scenecut=0:bframes=0:"
                       "force-cfr=1:rc-lookahead=0:ref=1:sliced-threads=0:"
                       "aq-mode=1:trellis=0:psy-rd=1.0:1.0";
        av_dict_set(&opt,"x264-params", xparam.c_str(),0);
        if(avcodec_open2(codec_ctx.get(), codec,&opt)<0){
            throw runtime_error("Could not open codec");
        }
        av_dict_free(&opt);

        frame.reset(av_frame_alloc());
        if(!frame) throw runtime_error("Could not alloc frame");
        frame->format= codec_ctx->pix_fmt;
        frame->width= codec_ctx->width;
        frame->height= codec_ctx->height;
        if(av_frame_get_buffer(frame.get(),32)<0)
            throw runtime_error("Could not alloc frame data");

        pkt.reset(av_packet_alloc());
        if(!pkt) throw runtime_error("Could not alloc AVPacket");

        sws_ctx= sws_getContext(WIDTH,HEIGHT, AV_PIX_FMT_YUYV422,
                                WIDTH,HEIGHT, AV_PIX_FMT_YUV420P,
                                SWS_BILINEAR, nullptr,nullptr,nullptr);
        if(!sws_ctx) throw runtime_error("sws ctx fail");

        // UDP 소켓
        sockfd1= create_socket_and_bind(INTERFACE1_IP, INTERFACE1_NAME);
        sockfd2= create_socket_and_bind(INTERFACE2_IP, INTERFACE2_NAME);

        servaddr1= create_sockaddr(SERVER_IP, SERVER_PORT);
        servaddr2= create_sockaddr(SERVER_IP, SERVER_PORT+1);
    }

    ~VideoStreamer(){
        if(sockfd1>=0) close(sockfd1);
        if(sockfd2>=0) close(sockfd2);
        if(sws_ctx) sws_freeContext(sws_ctx);
    }

    void stream(rs2::video_frame &color_frame, uint64_t ts){
        frame->pts= frame_counter++;

        uint8_t* yuyv= (uint8_t*)color_frame.get_data();
        const uint8_t* src_slices[1]= {yuyv};
        int src_stride[1]= {2*WIDTH};

        if(sws_scale(sws_ctx, src_slices, src_stride,
                     0,HEIGHT, frame->data, frame->linesize)<0){
            cerr<<"Error in sws_scale\n";
            return;
        }
        // (옵션) PNG저장
        encode_and_send_frame(ts);
    }

private:
    void create_and_set_output_folders(){
        auto now= chrono::system_clock::now();
        time_t tnow= chrono::system_clock::to_time_t(now);
        tm local_t= *localtime(&tnow);

        char folder_name[100];
        strftime(folder_name,sizeof(folder_name), "%Y_%m_%d_%H_%M", &local_t);

        string base= SAVE_FILEPATH + string(folder_name);
        fs::create_directories(base);

        frames_folder= base+"/frames";
        logs_folder= base+"/logs";
        fs::create_directories(frames_folder);
        fs::create_directories(logs_folder);
    }

    sockaddr_in create_sockaddr(const char* ip, int port){
        sockaddr_in addr={};
        addr.sin_family= AF_INET;
        addr.sin_port= htons(port);
        addr.sin_addr.s_addr= inet_addr(ip);
        return addr;
    }

    int create_socket_and_bind(const char* ip, const char* if_name){
        int s= socket(AF_INET,SOCK_DGRAM,0);
        if(s<0) throw runtime_error("socket fail");
        if(setsockopt(s, SOL_SOCKET, SO_BINDTODEVICE,
                      if_name, strlen(if_name))<0){
            close(s);
            throw runtime_error("SO_BINDTODEVICE fail: "+ string(if_name));
        }
        sockaddr_in bindaddr= create_sockaddr(ip,0);
        if(bind(s,(struct sockaddr*)&bindaddr,sizeof(bindaddr))<0){
            close(s);
            throw runtime_error("Bind fail on "+string(ip));
        }
        return s;
    }

    void encode_and_send_frame(uint64_t ts){
        if(avcodec_send_frame(codec_ctx.get(), frame.get())<0){
            cerr<<"Error sending frame\n";
            return;
        }
        int ret;
        while((ret= avcodec_receive_packet(codec_ctx.get(), pkt.get()))==0){
            PacketHeader hdr;
            hdr.timestamp_frame= ts;
            hdr.timestamp_sending= chrono::duration_cast<chrono::microseconds>(
                chrono::system_clock::now().time_since_epoch()).count();
            hdr.sequence_number= sequence_number++;

            double enc_lat= (hdr.timestamp_sending - hdr.timestamp_frame)/1000.0;
            string frame_type= (pkt->flags & AV_PKT_FLAG_KEY)?"I-frame":"P-frame";

            vector<uint8_t> data(sizeof(PacketHeader)+ pkt->size);
            memcpy(data.data(), &hdr, sizeof(PacketHeader));
            memcpy(data.data()+ sizeof(PacketHeader), pkt->data, pkt->size);

            ostringstream logMsg;
            logMsg<< hdr.sequence_number<<","<< frame->pts<<","<< pkt->size<<","
                  << hdr.timestamp_frame<<","<< hdr.timestamp_sending<<","
                  << enc_lat<<","<< frame_type;
            logger->log(logMsg.str());

            auto t1= async(launch::async, [this, &data](){
                if(sendto(sockfd1, data.data(), data.size(),0,
                          (struct sockaddr*)&servaddr1,sizeof(servaddr1))<0){
                    cerr<<"Send error(if1): "<<strerror(errno)<<"\n";
                }
            });
            auto t2= async(launch::async, [this, &data](){
                if(sendto(sockfd2, data.data(), data.size(),0,
                          (struct sockaddr*)&servaddr2,sizeof(servaddr2))<0){
                    cerr<<"Send error(if2): "<<strerror(errno)<<"\n";
                }
            });
            t1.get(); t2.get();
            av_packet_unref(pkt.get());
        }
        if(ret!=AVERROR(EAGAIN) && ret!= AVERROR_EOF){
            cerr<<"Error receiving encoded packet\n";
        }
    }

    const AVCodec* codec;
    unique_ptr<AVCodecContext, void(*)(AVCodecContext*)> codec_ctx{
        nullptr,[](AVCodecContext* p){ avcodec_free_context(&p); }
    };
    unique_ptr<AVFrame, void(*)(AVFrame*)> frame{
        nullptr,[](AVFrame* p){ av_frame_free(&p); }
    };
    unique_ptr<AVPacket, void(*)(AVPacket*)> pkt{
        nullptr,[](AVPacket* p){ av_packet_free(&p); }
    };
    SwsContext* sws_ctx= nullptr;

    int sockfd1=-1, sockfd2=-1;
    sockaddr_in servaddr1, servaddr2;

    unique_ptr<BufferedLogger> logger;
    string frames_folder, logs_folder;

    atomic<int> frame_counter;
    atomic<int> sequence_number;
};

//
// TURN 관련
//
static pj_caching_pool g_cp;
static pj_pool_t *g_pool= nullptr;
static pj_ioqueue_t *g_ioqueue= nullptr;
static pj_timer_heap_t *g_timer_heap= nullptr;
static pj_turn_sock *turn_sock= nullptr;

atomic<bool> turn_running(true);

static void turn_ack_receiver_thread()
{
    pj_status_t status;

    // 1) pjlib init
    status= pj_init();
    if(status!=PJ_SUCCESS){
        PJ_LOG(1,("TAG","pj_init err"));
        return;
    }
    status= pjlib_util_init();
    if(status!=PJ_SUCCESS){
        PJ_LOG(1,("TAG","pjlib_util_init err"));
        pj_shutdown();
        return;
    }

    // 2) caching pool
    pj_caching_pool_init(&g_cp, NULL, 0);
    g_pool= pj_pool_create(&g_cp.factory, "turn_pool", 4000, 4000, NULL);

    // 3) ioqueue/timer
    status= pj_ioqueue_create(g_pool, 32, &g_ioqueue);
    if(status!=PJ_SUCCESS){
        PJ_LOG(1,("TAG","pj_ioqueue_create err %d", status));
        // cleanup
        pj_pool_release(g_pool);
        pj_caching_pool_destroy(&g_cp);
        pj_shutdown();
        return;
    }
    status= pj_timer_heap_create(g_pool, 32, &g_timer_heap);
    if(status!=PJ_SUCCESS){
        PJ_LOG(1,("TAG","pj_timer_heap_create err %d", status));
        // cleanup
        pj_ioqueue_destroy(g_ioqueue);
        pj_pool_release(g_pool);
        pj_caching_pool_destroy(&g_cp);
        pj_shutdown();
        return;
    }

    // 4) stun config
    pj_stun_config stun_cfg;
    pj_stun_config_init(&stun_cfg, &g_cp.factory, 0, g_ioqueue, g_timer_heap);

    pj_turn_sock_cfg tcfg;
    pj_turn_sock_cfg_default(&tcfg);

    status= pj_turn_sock_create(
        &stun_cfg,
        PJ_AF_INET,
        PJ_TURN_TP_UDP,
        &turn_callbacks,
        &tcfg,
        NULL,
        &turn_sock
    );
    if(status!=PJ_SUCCESS){
        PJ_LOG(1,("TAG","pj_turn_sock_create err %d", status));
        pj_ioqueue_destroy(g_ioqueue);
        pj_pool_release(g_pool);
        pj_caching_pool_destroy(&g_cp);
        pj_shutdown();
        return;
    }

    // 인증
    pj_str_t turnServer= pj_str(const_cast<char*>(TURN_SERVER_IP));
    pj_stun_auth_cred auth_cred;
    pj_bzero(&auth_cred, sizeof(auth_cred));
    auth_cred.type= PJ_STUN_AUTH_CRED_STATIC;

    uint64_t expiration = chrono::duration_cast<chrono::seconds>(
                          chrono::system_clock::now().time_since_epoch()
                      ).count() + validSeconds;

    string ephemeral_username= generate_turn_username(TURN_IDENTIFIER, TURN_VALID_SECONDS);
    // string ephemeral_password= compute_turn_password(
    //     ephemeral_username + ":" + TURN_REALM,
    //     TURN_SECRET
    // );
    std::string ephemeral_password_bin = compute_turn_password_bin(
    ephemeral_username + ":" + TURN_REALM,
    TURN_SECRET
    );
    // 
    uint64_t now_epoch = chrono::duration_cast<chrono::seconds>(
                        chrono::system_clock::now().time_since_epoch()
                    ).count();
    std::cerr << "[DEBUG] local epoch now     : " << now_epoch << std::endl;
    std::cerr << "[DEBUG] ephemeral_username : " << ephemeral_username << std::endl;
    std::cerr << "[DEBUG] ephemeral_password_bin.size()=" << ephemeral_password_bin.size() << "\n";

    std::cerr << "[DEBUG] expiration         : " << expiration << " (+" << validSeconds << " from now)\n";
    std::cerr << "[DEBUG] ephemeral_password_bin(hex)=";
    for (unsigned char c : ephemeral_password_bin) {
        std::cerr << std::hex << std::setw(2) << std::setfill('0')
                << (unsigned int)c;
    }
    std::cerr << std::dec << "\n"; // dec로 복원
    // 

    auth_cred.data.static_cred.username = pj_str(const_cast<char*>(ephemeral_username.c_str()));
    // auth_cred.data.static_cred.data= pj_str(const_cast<char*>(ephemeral_password.c_str()));
    auth_cred.data.static_cred.data_type= PJ_STUN_PASSWD_PLAIN;
    auth_cred.data.static_cred.data.ptr =
    const_cast<char*>(ephemeral_password_bin.data());
    auth_cred.data.static_cred.data.slen =
    (pj_ssize_t)ephemeral_password_bin.size();

    status= pj_turn_sock_alloc(
        turn_sock,
        &turnServer,
        TURN_SERVER_PORT,
        NULL,
        &auth_cred,
        NULL
    );
    if(status!=PJ_SUCCESS){
        PJ_LOG(1,("TAG","pj_turn_sock_alloc err %d", status));
        pj_turn_sock_destroy(turn_sock);
        pj_ioqueue_destroy(g_ioqueue);
        pj_pool_release(g_pool);
        pj_caching_pool_destroy(&g_cp);
        pj_shutdown();
        return;
    }

    PJ_LOG(3,("TURN","ack receiver started."));

    // 이 스레드는 allocate 후 바로 종료할 수도 있음. 
    // 혹은 여기서 while(turn_running) { poll } 식으로 돌릴 수도 있다.
    // 예) 여기서 sleep/loop
    while(turn_running.load()) {
        pj_time_val delay= {0,10};
        pj_ioqueue_poll(g_ioqueue, &delay);
        pj_timer_heap_poll(g_timer_heap, NULL);
        pj_thread_sleep(10);
    }

    // clean
    if(turn_sock){
        pj_turn_sock_destroy(turn_sock);
        turn_sock= nullptr;
    }
    if(g_ioqueue) {
        pj_ioqueue_destroy(g_ioqueue);
        g_ioqueue= nullptr;
    }
    if(g_pool){
        pj_pool_release(g_pool);
        g_pool= nullptr;
    }
    pj_caching_pool_destroy(&g_cp);
    pj_shutdown();
}

void client_stream(VideoStreamer &streamer, rs2::pipeline &pipe, atomic<bool> &running){
    while(running.load()){
        rs2::frameset frames= pipe.wait_for_frames();
        rs2::video_frame color_frame= frames.get_color_frame();
        if(!color_frame) continue;

        uint64_t ts= chrono::duration_cast<chrono::microseconds>(
            chrono::system_clock::now().time_since_epoch()).count();

        streamer.stream(color_frame, ts);
    }
}

int main() {
    try{
        // RealSense
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_YUYV, FPS);
        pipe.start(cfg);

        VideoStreamer streamer;

        atomic<bool> running(true);
        thread client_thread(client_stream, ref(streamer), ref(pipe), ref(running));

        // TURN
        thread turn_thread(turn_ack_receiver_thread);

        cout<<"Press Enter to stop streaming..."<<endl;
        cin.get();

        running.store(false);
        turn_running.store(false);

        client_thread.join();
        turn_thread.join();

    } catch(const exception &e){
        cerr<<"Error: "<< e.what() <<endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
