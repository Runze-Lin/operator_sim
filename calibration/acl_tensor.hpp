#pragma once

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

static void check(int code, const char *expr) {
    if (code) {
        const char *detail = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(expr) + " code=" + std::to_string(code)
                                 + (detail ? std::string(" ") + detail : ""));
    }
}
#define CHECK(expr) check(expr,#expr)

struct Buffer {
    void *ptr = nullptr;
    explicit Buffer(size_t bytes) {
        if (bytes) CHECK(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ~Buffer() { if (ptr) aclrtFree(ptr); }
    Buffer(const Buffer&) = delete;
};

struct Tensor {
    Buffer buffer;
    aclTensor *desc;
    Tensor(const std::vector<int64_t>& shape, aclDataType type, const std::vector<uint8_t>& bytes)
        : buffer(bytes.size()) {
        std::vector<int64_t> stride(shape.size(), 1);
        for (size_t i=shape.size()-1; i>0; --i) stride[i-1]=stride[i]*shape[i];
        CHECK(aclrtMemcpy(buffer.ptr,bytes.size(),bytes.data(),bytes.size(),ACL_MEMCPY_HOST_TO_DEVICE));
        desc=aclCreateTensor(shape.data(),shape.size(),type,stride.data(),0,ACL_FORMAT_ND,
                             shape.data(),shape.size(),buffer.ptr);
        if (!desc) throw std::runtime_error("aclCreateTensor failed");
    }
    ~Tensor() { aclDestroyTensor(desc); }
};
