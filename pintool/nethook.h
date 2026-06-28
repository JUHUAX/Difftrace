#ifndef NETHOOK_H
#define NETHOOK_H

#include "pin.H"
#include "address.h"
#include "logger.h"
#include "taintset.h"
#include <map>

// 网络接收函数 Hook 管理器
class NetHook {
private:
    struct RecvState {
        int streamId;
        uint64_t baseBuffer;
        uint64_t lastRecvEnd;
        uint64_t currentOffset;
        bool initialized;

        RecvState()
            : streamId(-1), baseBuffer(0), lastRecvEnd(0), currentOffset(0),
              initialized(false) {}
    };

    struct RecentSyscallInjection {
        int streamId;
        uint64_t buffer;
        size_t nbytes;
        bool valid;

        RecentSyscallInjection()
            : streamId(-1), buffer(0), nbytes(0), valid(false) {}
    };

    struct RecentInjection {
        int streamId;
        uint64_t buffer;
        size_t nbytes;
        bool valid;

        RecentInjection()
            : streamId(-1), buffer(0), nbytes(0), valid(false) {}
    };

    // 连续 recv 检测状态，按线程维护，避免不同 worker 互相污染偏移
    static std::map<THREADID, RecvState> recvStates;
    static std::map<THREADID, RecentSyscallInjection> recentSyscallInjections;
    static std::map<THREADID, RecentInjection> recentInjections;
    static PIN_LOCK indexLock;

    // 每个 fd 的污点注入记录（用于累积字节计数）
    static std::map<int, size_t> fdTaintedBytes;
    static PIN_LOCK fdLock;

    static bool shouldSuppressDuplicateRtnInjection(THREADID tid,
                                                   uint64_t bufferAddr,
                                                   size_t nbytes,
                                                   int streamId);
    static bool shouldSuppressDuplicateInjection(THREADID tid,
                                                 uint64_t bufferAddr,
                                                 size_t nbytes,
                                                 int streamId);
    static void markRecentSyscallInjection(THREADID tid, uint64_t bufferAddr,
                                           size_t nbytes, int streamId);
    static void markRecentInjection(THREADID tid, uint64_t bufferAddr,
                                    size_t nbytes, int streamId);

public:
    // 初始化
    static void init();

    // 销毁
    static void fini();

    // ============== recv() Hook ==============

    // recv 前处理
    static void beforeRecv(THREADID tid, CONTEXT* ctxt, AFUNPTR originalFunc,
                          int socket, void* buffer, size_t length);

    // recv 后处理（返回值在 RAX 中）
    static void afterRecv(THREADID tid, CONTEXT* ctxt, int socket,
                         void* buffer, size_t length);

    // ============== recvmsg() Hook ==============

    // recvmsg 前处理
    static void beforeRecvmsg(THREADID tid, CONTEXT* ctxt, int socket,
                             void* msg);

    // recvmsg 后处理
    static void afterRecvmsg(THREADID tid, CONTEXT* ctxt, int socket,
                            void* msg);

    // ============== recvfrom() Hook ==============

    // recvfrom 前处理
    static void beforeRecvfrom(THREADID tid, CONTEXT* ctxt, int socket,
                              void* buffer, size_t length);

    // recvfrom 后处理
    static void afterRecvfrom(THREADID tid, CONTEXT* ctxt, int socket,
                             void* buffer, size_t length);

    // ============== read() Hook ==============

    // read 前处理
    static void beforeRead(THREADID tid, CONTEXT* ctxt, int fd, void* buf,
                          size_t count);

    // read 后处理
    static void afterRead(THREADID tid, CONTEXT* ctxt, int fd, void* buf,
                         size_t count);

    // ============== 污点注入 ==============

    // 核心污点注入函数
    // 为 buffer[0..nbytes-1] 的每个字节分配污点索引
    static void injectTaint(THREADID tid, const void* buffer, size_t nbytes,
                           int streamId = -1, bool fromSyscall = false);

    // ============== 外部接口 ==============

    // 获取当前线程的累积偏移（用于查询）
    static uint64_t getCurrentOffset(THREADID tid);

    // 应用代码继续执行后，上一轮 recv 的重复 hook 窗口结束。
    static void clearRecentInjection(THREADID tid);
};

#endif  // NETHOOK_H
