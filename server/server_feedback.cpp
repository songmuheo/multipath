// server_turn.cpp
#include <iostream>
#include <string>
#include <atomic>
#include <pjlib.h>
#include <pjlib-util.h>
#include <pjnath.h>
#include "common_turn_utils.h"

static std::atomic<bool> g_running{true};

// 수신 콜백
static void on_rx_data(pj_turn_sock *sock,
                       void *user_data,
                       unsigned int size,
                       const pj_sockaddr_t *src_addr,
                       unsigned int addr_len)
{
    if (size > 0) {
        // 간단히 수신 데이터를 문자열로 보고 출력
        std::string recv_str((char*)user_data, size);
        std::cout << "[SERVER] Received data: " << recv_str << "\n";

        // 예: "ACK"를 보내거나, 혹은 "ACK: + recv_str" 등
        std::string ack_str = "ACK from SERVER: " + recv_str;

        // src_addr가 "클라이언트 relay 주소"임. 여기에 다시 전송
        pj_status_t status = pj_turn_sock_sendto(sock,
                                                 ack_str.data(),
                                                 (pj_size_t)ack_str.size(),
                                                 0,
                                                 src_addr,
                                                 addr_len);
        if (status != PJ_SUCCESS) {
            std::cerr << "[SERVER] Failed to send ACK via TURN\n";
        }
    }
}

// TURN 소켓 콜백 구조체
static pj_turn_sock_cb g_turn_cb;

int main()
{
    // ---------------------
    // 1) PJLIB / pjnath 초기화
    // ---------------------
    pj_status_t status;
    status = pj_init();                if (status != PJ_SUCCESS) { std::cerr << "pj_init() failed\n"; return 1; }
    status = pjlib_util_init();        if (status != PJ_SUCCESS) { std::cerr << "pjlib_util_init() failed\n"; return 1; }

    pj_caching_pool cp;
    pj_caching_pool_init(&cp, NULL, 0);

    pj_pool_t *pool = pj_pool_create(&cp.factory, "srv_pool", 4000, 4000, NULL);
    if (!pool) {
        std::cerr << "Failed to create pool\n";
        return 1;
    }

    pj_ioqueue_t *ioqueue = NULL;
    status = pj_ioqueue_create(pool, 64, &ioqueue);
    if (status != PJ_SUCCESS) {
        std::cerr << "pj_ioqueue_create() error\n";
        return 1;
    }

    pj_timer_heap_t *timer_heap = NULL;
    status = pj_timer_heap_create(pool, 100, &timer_heap);
    if (status != PJ_SUCCESS) {
        std::cerr << "pj_timer_heap_create() error\n";
        return 1;
    }

    pj_stun_config stun_cfg;
    pj_stun_config_init(&stun_cfg, &cp.factory, PJ_AF_INET, ioqueue, timer_heap);

    // ---------------------
    // 2) TURN 소켓 생성
    // ---------------------
    pj_turn_sock *turn_sock = NULL;

    // 콜백 설정
    pj_bzero(&g_turn_cb, sizeof(g_turn_cb));
    // on_rx_data만 사용 (나머지는 필요시 추가)
    g_turn_cb.on_rx_data = &on_rx_data;

    // create turn sock
    status = pj_turn_sock_create(&stun_cfg,
                                 PJ_AF_INET,      // IPv4
                                 PJ_TURN_TP_UDP,  // UDP transport
                                 &g_turn_cb,
                                 NULL,            // user_data
                                 NULL,            // 프로락시 소켓?
                                 &turn_sock);
    if (status != PJ_SUCCESS) {
        std::cerr << "pj_turn_sock_create() error\n";
        return 1;
    }

    // ---------------------
    // 3) TURN 서버에 Allocate
    // ---------------------
    // 서버도 ephemeral username 사용 가능(유효기간+식별자). 여기선 식별자 "server"로.
    std::string user_str   = generate_turn_username("server", 300 /*유효 300초*/);
    std::string pass_str   = compute_turn_password(user_str + std::string(":") + TURN_REALM,
                                                   TURN_SECRET);

    pj_stun_auth_cred creds;
    memset(&creds, 0, sizeof(creds));
    creds.type = PJ_STUN_AUTH_CRED_STATIC;
    creds.data.static_cred.username = pj_str(const_cast<char*>(user_str.c_str()));
    creds.data.static_cred.data     = pj_str(const_cast<char*>(pass_str.c_str()));
    creds.data.static_cred.data_type = PJ_STUN_PASSWD_PLAIN;

    pj_str_t srv_ip = pj_str(const_cast<char*>(TURN_SERVER_IP));

    status = pj_turn_sock_alloc(turn_sock,
                                &srv_ip,
                                TURN_SERVER_PORT,
                                NULL,   // (optional) local bound address
                                &creds,
                                NULL);  // user_data for on_state callback
    if (status != PJ_SUCCESS) {
        std::cerr << "pj_turn_sock_alloc() error\n";
        return 1;
    }

    // 4) 할당 완료까지 대기(비동기로 진행되므로, poll을 돌려야 함)
    //    실제로는 on_state_changed 콜백 등에서 확인 가능. 여기서는 간단 폴링
    bool allocated = false;
    for(int i=0; i<100; i++){
        pj_time_val delay = {0, 10}; // 10 msec
        pj_ioqueue_poll(ioqueue, &delay);
        pj_timer_heap_poll(timer_heap, NULL);

        pj_turn_sock_info info;
        pj_bzero(&info, sizeof(info));
        pj_turn_sock_get_info(turn_sock, &info);

        if (info.state == PJ_TURN_STATE_READY) {
            allocated = true;
            break;
        }
        pj_thread_sleep(0,10);
    }

    if (!allocated) {
        std::cerr << "[SERVER] TURN allocation not finished.\n";
        return 1;
    }

    // 5) 할당된 relay 주소 확인
    {
        pj_turn_sock_info info;
        pj_bzero(&info, sizeof(info));
        pj_turn_sock_get_info(turn_sock, &info);

        char ipstr[128];
        pj_sockaddr_print(&info.relay_addr, ipstr, sizeof(ipstr), 0);
        std::cout << "[SERVER] TURN relay allocated! Relay addr = " << ipstr << "\n"
                  << "         Please give this address to the client so it can send data here.\n";
    }

    // ---------------------
    // 6) 메인 루프: 데이터 수신/ACK 전송
    // ---------------------
    std::cout << "[SERVER] Entering loop. Press Ctrl+C to stop.\n";
    while (g_running) {
        pj_time_val delay = {0, 10}; // 10 msec
        pj_ioqueue_poll(ioqueue, &delay);
        pj_timer_heap_poll(timer_heap, NULL);

        // 간단히 Sleep
        pj_thread_sleep(0,10);
    }

    // 종료
    if (turn_sock) {
        pj_turn_sock_destroy(turn_sock);
        turn_sock = NULL;
    }
    pj_timer_heap_destroy(timer_heap);
    pj_ioqueue_destroy(ioqueue);
    pj_pool_release(pool);
    pj_caching_pool_destroy(&cp);
    pj_shutdown();
    return 0;
}
