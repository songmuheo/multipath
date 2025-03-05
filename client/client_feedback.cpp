// client_turn.cpp
#include <iostream>
#include <string>
#include <atomic>
#include <pjlib.h>
#include <pjlib-util.h>
#include <pjnath.h>
#include "common_turn_utils.h"

static std::atomic<bool> g_running{true};

// 수신 콜백 (서버의 ACK 등 수신)
static void on_rx_data(pj_turn_sock *sock,
                       void *user_data,
                       unsigned int size,
                       const pj_sockaddr_t *src_addr,
                       unsigned int addr_len)
{
    if (size > 0) {
        // 넘어온 user_data는 기본적으로 send시 사용되는 버퍼 등인데,
        // 여기서는 수신한 payload를 직접 캐스팅하여 읽을 수 있음
        const char* data_ptr = (const char*)user_data;
        std::string recv_str(data_ptr, size);
        std::cout << "[CLIENT] Received from server: " << recv_str << "\n";
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

    pj_pool_t *pool = pj_pool_create(&cp.factory, "cli_pool", 4000, 4000, NULL);
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

    pj_bzero(&g_turn_cb, sizeof(g_turn_cb));
    g_turn_cb.on_rx_data = &on_rx_data;  // 수신 콜백

    status = pj_turn_sock_create(&stun_cfg,
                                 PJ_AF_INET,
                                 PJ_TURN_TP_UDP,
                                 &g_turn_cb,
                                 NULL,  // user_data
                                 NULL,  // 포락시 소켓
                                 &turn_sock);
    if (status != PJ_SUCCESS) {
        std::cerr << "pj_turn_sock_create() error\n";
        return 1;
    }

    // ---------------------
    // 3) TURN 서버에 Allocate
    // ---------------------
    // 클라이언트도 ephemeral username (식별자 "client")
    std::string user_str   = generate_turn_username("client", 300);
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
                                NULL, // local bound
                                &creds,
                                NULL);
    if (status != PJ_SUCCESS) {
        std::cerr << "pj_turn_sock_alloc() error\n";
        return 1;
    }

    // 할당 완료 대기
    bool allocated = false;
    for(int i=0; i<100; i++){
        pj_time_val delay = {0, 10};
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
        std::cerr << "[CLIENT] TURN allocation not finished.\n";
        return 1;
    }

    // 4) 할당된 클라이언트 relay 주소 확인
    {
        pj_turn_sock_info info;
        pj_turn_sock_get_info(turn_sock, &info);

        char ipstr[128];
        pj_sockaddr_print(&info.relay_addr, ipstr, sizeof(ipstr), 0);
        std::cout << "[CLIENT] My TURN relay = " << ipstr << "\n";
    }

    // -------------------------------------------
    // 5) "서버 relay 주소"를 사용하여 Permission 설정
    //    (서버 콘솔에서 복사해온 relay 주소를 여기다가 세팅)
    // -------------------------------------------
    // 예: 서버 콘솔에 "121.128.220.205:6000" 이라면 아래처럼:
    std::string server_relay_ip   = "121.128.220.205";
    int         server_relay_port = 6000;  // 실제 할당 포트

    // 문자열로부터 pj_sockaddr 생성
    pj_sockaddr_in server_addr;
    pj_bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = PJ_AF_INET;
    server_addr.sin_port   = pj_htons((pj_uint16_t)server_relay_port);
    pj_inet_pton(PJ_AF_INET, server_relay_ip.c_str(), &server_addr.sin_addr);

    // Permission
    status = pj_turn_sock_set_perm(turn_sock,
                                   1, // 1개 주소
                                   (pj_sockaddr*)&server_addr,
                                   sizeof(server_addr));
    if (status != PJ_SUCCESS) {
        std::cerr << "[CLIENT] pj_turn_sock_set_perm() error\n";
        // 에러 시에도 계속 진행 가능, 그러나 실제 전송은 실패할 수도
    }

    // ---------------------
    // 6) 데이터 전송 테스트
    // ---------------------
    // 예: "Hello from client"를 몇 번 보내기
    for (int i=0; i<10; i++) {
        std::string msg = "Hello from client " + std::to_string(i);

        status = pj_turn_sock_sendto(turn_sock,
                                     msg.data(),
                                     (pj_size_t)msg.size(),
                                     0, // flags
                                     (pj_sockaddr*)&server_addr,
                                     sizeof(server_addr));
        if (status != PJ_SUCCESS) {
            std::cerr << "[CLIENT] sendto() via TURN failed\n";
        } else {
            std::cout << "[CLIENT] Sent: " << msg << "\n";
        }

        // 잠시 대기
        pj_thread_sleep(0, 500); // 500ms
        // 이벤트 처리
        for (int k=0; k<10; k++){
            pj_time_val delay = {0, 10};
            pj_ioqueue_poll(ioqueue, &delay);
            pj_timer_heap_poll(timer_heap, NULL);
            pj_thread_sleep(0,10);
        }
    }

    // ---------------------
    // 메인 루프 (수신 대기)
    // ---------------------
    std::cout << "[CLIENT] Entering loop. Press Ctrl+C to stop.\n";
    while (g_running) {
        pj_time_val delay = {0, 10};
        pj_ioqueue_poll(ioqueue, &delay);
        pj_timer_heap_poll(timer_heap, NULL);

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
