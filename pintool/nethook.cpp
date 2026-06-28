#include "nethook.h"
#include "instrumentation.h"
#include "moduleinfo.h"
#include <cstring>

// 静态成员初始化
std::map<THREADID, NetHook::RecvState> NetHook::recvStates;
std::map<THREADID, NetHook::RecentSyscallInjection> NetHook::recentSyscallInjections;
std::map<THREADID, NetHook::RecentInjection> NetHook::recentInjections;
PIN_LOCK NetHook::indexLock;
std::map<int, size_t> NetHook::fdTaintedBytes;
PIN_LOCK NetHook::fdLock;

void NetHook::init() {
    PIN_InitLock(&indexLock);
    PIN_InitLock(&fdLock);
}

void NetHook::fini() {
    // PIN_LOCK doesn't require explicit cleanup
}

bool NetHook::shouldSuppressDuplicateRtnInjection(THREADID tid,
                                                  uint64_t bufferAddr,
                                                  size_t nbytes,
                                                  int streamId) {
    PIN_GetLock(&indexLock, PIN_ThreadId());
    std::map<THREADID, RecentSyscallInjection>::iterator it =
        recentSyscallInjections.find(tid);
    bool suppress = false;
    if (it != recentSyscallInjections.end() && it->second.valid &&
        it->second.buffer == bufferAddr && it->second.nbytes == nbytes &&
        it->second.streamId == streamId) {
        recentSyscallInjections.erase(it);
        suppress = true;
    }
    PIN_ReleaseLock(&indexLock);
    return suppress;
}

void NetHook::markRecentSyscallInjection(THREADID tid, uint64_t bufferAddr,
                                         size_t nbytes, int streamId) {
    PIN_GetLock(&indexLock, PIN_ThreadId());
    RecentSyscallInjection& recent = recentSyscallInjections[tid];
    recent.streamId = streamId;
    recent.buffer = bufferAddr;
    recent.nbytes = nbytes;
    recent.valid = true;
    PIN_ReleaseLock(&indexLock);
}

bool NetHook::shouldSuppressDuplicateInjection(THREADID tid,
                                               uint64_t bufferAddr,
                                               size_t nbytes,
                                               int streamId) {
    std::map<THREADID, RecentInjection>::iterator it = recentInjections.find(tid);
    return it != recentInjections.end() && it->second.valid &&
           it->second.buffer == bufferAddr && it->second.nbytes == nbytes;
}

void NetHook::markRecentInjection(THREADID tid, uint64_t bufferAddr,
                                  size_t nbytes, int streamId) {
    RecentInjection& recent = recentInjections[tid];
    recent.streamId = streamId;
    recent.buffer = bufferAddr;
    recent.nbytes = nbytes;
    recent.valid = true;
}

void NetHook::injectTaint(THREADID tid, const void* buffer, size_t nbytes,
                          int streamId, bool fromSyscall) {
    if (nbytes == 0 || !buffer) {
        return;
    }

    uint64_t bufferAddr = reinterpret_cast<uint64_t>(buffer);

    if (!fromSyscall &&
        shouldSuppressDuplicateRtnInjection(tid, bufferAddr, nbytes, streamId)) {
        return;
    }

    PIN_GetLock(&indexLock, PIN_ThreadId());
    if (shouldSuppressDuplicateInjection(tid, bufferAddr, nbytes, streamId)) {
        PIN_ReleaseLock(&indexLock);
        return;
    }
    PIN_ReleaseLock(&indexLock);

    // 检测是否为连续 recv（同一个逻辑数据包），按线程和 socket/fd 维护接收状态。
    // 只有紧贴上一次 recv 末尾的写入才视为续接；缓冲区回退/重叠通常意味着
    // 服务端复用了接收缓冲区处理新包，不能继承上一包的全局 offset。
    bool isNewLogicalPacket = false;
    uint64_t startOffset = 0;
    uint64_t staleBase = 0;
    uint64_t staleSize = 0;
    PIN_GetLock(&indexLock, PIN_ThreadId());
    RecvState& recvState = recvStates[tid];

    if (!recvState.initialized || recvState.streamId != streamId) {
        recvState.streamId = streamId;
        recvState.baseBuffer = bufferAddr;
        recvState.currentOffset = 0;
        recvState.lastRecvEnd = bufferAddr;
        recvState.initialized = true;
        isNewLogicalPacket = true;
    }

    if (!isNewLogicalPacket) {
        if (bufferAddr == recvState.lastRecvEnd) {
            startOffset = recvState.currentOffset;
        } else {
            staleBase = recvState.baseBuffer;
            staleSize = recvState.currentOffset;
            recvState.baseBuffer = bufferAddr;
            recvState.currentOffset = 0;
            recvState.lastRecvEnd = bufferAddr;
            startOffset = 0;
            isNewLogicalPacket = true;
        }
    }

    uint64_t newEndOffset = startOffset + nbytes;
    if (newEndOffset > recvState.currentOffset) {
        recvState.currentOffset = newEndOffset;
    }
    recvState.lastRecvEnd = bufferAddr + nbytes;
    PIN_ReleaseLock(&indexLock);

    TaintState* state = InstrumentationManager::getTaintState(tid);

    if (isNewLogicalPacket) {
        // A new logical packet should not inherit loop history accumulated
        // before the first taint injection of this packet.
        InstrumentationManager::loopDetector.clearStack(tid);

        // Analysis is packet-scoped: taint from an earlier logical packet must
        // not survive in long-lived protocol objects reused by the server.
        state->clearAll();

        // If the application reuses the same receive buffer for a new packet,
        // stale taint past the newly received length would otherwise survive.
        // This is redundant after clearAll(), but kept as a narrow safeguard if
        // the packet-boundary cleanup is changed in the future.
        if (staleBase != 0 && staleSize != 0) {
            for (uint64_t i = 0; i < staleSize; ++i) {
                state->clearMemoryTaint(staleBase + i);
            }
        }
    }

    PIN_GetLock(&indexLock, PIN_ThreadId());
    markRecentInjection(tid, bufferAddr, nbytes, streamId);
    PIN_ReleaseLock(&indexLock);

    // 通过模块映射表获取地址信息（无需加锁）
    Address addr = ModuleInfoManager::getAddress((ADDRINT)buffer);
    
    // 如果未找到模块，使用 memory 前缀
    if (!addr.isValid()) {
        addr = Address("memory", (ADDRINT)buffer);
    }

    // 记录污点注入
    Logger::getInstance().logTaintInject(tid, addr, nbytes);

    // 更新内存污点状态
    const uint8_t* base = static_cast<const uint8_t*>(buffer);
    for (size_t i = 0; i < nbytes; ++i) {
        TaintSet t;
        t.addIndex(startOffset + i);  // 使用累积偏移
        state->setMemoryTaint(reinterpret_cast<uint64_t>(base + i), t);
    }

    if (fromSyscall) {
        markRecentSyscallInjection(tid, bufferAddr, nbytes, streamId);
    }

    if (config::DEBUG_MODE) {
        fprintf(stderr,
                "[TAINT] Injected %zu bytes at %s (stream=%d, offsets %lu-%lu)\n",
                nbytes, addr.toString().c_str(), streamId, startOffset,
                startOffset + nbytes - 1);
    }
}

void NetHook::beforeRecv(THREADID tid, CONTEXT* ctxt, AFUNPTR originalFunc,
                        int socket, void* buffer, size_t length) {
    // recv 前不需要做什么特殊处理
}

void NetHook::afterRecv(THREADID tid, CONTEXT* ctxt, int socket, void* buffer,
                       size_t length) {
    // 从 RAX 获取返回值（接收字节数）
    ssize_t ret = (ssize_t)PIN_GetContextReg(ctxt, REG_RAX);

    if (ret <= 0) {
        return;  // 没有数据接收
    }

    // 为接收的数据注入污点
    injectTaint(tid, buffer, (size_t)ret, socket, false);
}

void NetHook::beforeRecvmsg(THREADID tid, CONTEXT* ctxt, int socket,
                           void* msg) {
}

void NetHook::afterRecvmsg(THREADID tid, CONTEXT* ctxt, int socket,
                          void* msg) {
    // recvmsg 返回值在 RAX 中
    ssize_t ret = (ssize_t)PIN_GetContextReg(ctxt, REG_RAX);

    if (ret <= 0 || !msg) {
        return;
    }

    // msghdr 结构
    struct msghdr* m = (struct msghdr*)msg;

    // 获取第一个 iovec 缓冲区（简化版本）
    if (m->msg_iov && m->msg_iovlen > 0) {
        void* buffer = m->msg_iov[0].iov_base;
        injectTaint(tid, buffer, (size_t)ret, socket, false);
    }
}

void NetHook::beforeRecvfrom(THREADID tid, CONTEXT* ctxt, int socket,
                            void* buffer, size_t length) {
}

void NetHook::afterRecvfrom(THREADID tid, CONTEXT* ctxt, int socket,
                           void* buffer, size_t length) {
    // recvfrom 返回值在 RAX 中
    ssize_t ret = (ssize_t)PIN_GetContextReg(ctxt, REG_RAX);

    if (ret <= 0) {
        return;
    }

    injectTaint(tid, buffer, (size_t)ret, socket, false);
}

void NetHook::beforeRead(THREADID tid, CONTEXT* ctxt, int fd, void* buf,
                        size_t count) {
}

void NetHook::afterRead(THREADID tid, CONTEXT* ctxt, int fd, void* buf,
                       size_t count) {
    // 仅对网络套接字 fd 进行污点注入
    // 这里简化处理：仅针对 fd > 0 的情况
    if (fd <= 0) {
        return;
    }

    ssize_t ret = (ssize_t)PIN_GetContextReg(ctxt, REG_RAX);

    if (ret <= 0) {
        return;
    }

    injectTaint(tid, buf, (size_t)ret, fd, false);
}

uint64_t NetHook::getCurrentOffset(THREADID tid) {
    PIN_GetLock(&indexLock, PIN_ThreadId());
    uint64_t offset = 0;
    std::map<THREADID, RecvState>::const_iterator it = recvStates.find(tid);
    if (it != recvStates.end()) {
        offset = it->second.currentOffset;
    }
    PIN_ReleaseLock(&indexLock);
    return offset;
}

void NetHook::clearRecentInjection(THREADID tid) {
    PIN_GetLock(&indexLock, PIN_ThreadId());
    recentInjections.erase(tid);
    PIN_ReleaseLock(&indexLock);
}
