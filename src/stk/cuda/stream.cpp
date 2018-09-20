#include "stream.h"

namespace stk {
namespace cuda {
 
Event::Event(uint32_t flags)
{
    CUDA_CHECK_ERRORS(cudaEventCreateWithFlags(&_event, flags));
}
Event::~Event()
{
    CUDA_CHECK_ERRORS(cudaEventDestroy(_event));
}
bool Event::query()
{
    cudaError_t status = cudaEventQuery(_event);
    if (status != cudaSuccess && status != cudaErrorNotReady) {
        CUDA_CHECK_ERRORS(status);
    }
    return status == cudaSuccess;
}
void Event::record(const Stream& stream)
{
    CUDA_CHECK_ERRORS(cudaEventRecord(_event, stream));
}
void Event::synchronize()
{
    CUDA_CHECK_ERRORS(cudaEventSynchronize(_event));
}
Event::operator cudaEvent_t() const
{
    return _event;
}
float Event::elapsed(const Event& start, const Event& end)
{
    float ms;
    CUDA_CHECK_ERRORS(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

struct Stream::Internal
{
    Internal()
    {
        destroy = true;
        CUDA_CHECK_ERRORS(cudaStreamCreate(&stream));
    }
    Internal(cudaStream_t s)
    {
        destroy = false;
        stream = s;
    }
    ~Internal()
    {
        if (destroy) 
            CUDA_CHECK_ERRORS(cudaStreamDestroy(stream));
    }

    bool destroy; // We only want to destroy streams we actually own
    cudaStream_t stream;
};

Stream::Stream()
{
    _impl = std::make_shared<Internal>();
}
Stream::~Stream()
{
}
void Stream::add_callback(Callback cb)
{
    struct CallbackData {
        Stream stream;
        Callback cb;
    };
    
    auto user_data = new CallbackData{*this, cb};
    CUDA_CHECK_ERRORS(cudaStreamAddCallback(
        _impl->stream,
        [](cudaStream_t, cudaError_t status, void* user_data) {
            auto data = reinterpret_cast<CallbackData*>(user_data);
            data->cb(data->stream, (int)status);
            delete data;
        },
        user_data,
        0
    ));
}
bool Stream::query()
{
    cudaError_t status = cudaStreamQuery(_impl->stream);
    if (status != cudaSuccess && status != cudaErrorNotReady) {
        CUDA_CHECK_ERRORS(status);
    }
    return status == cudaSuccess;
}
void Stream::synchronize()
{
    CUDA_CHECK_ERRORS(cudaStreamSynchronize(_impl->stream));
}
void Stream::wait_event(const Event& event)
{
    CUDA_CHECK_ERRORS(cudaStreamWaitEvent(_impl->stream, event, 0));
}
Stream::operator cudaStream_t() const
{
    return _impl->stream;
}

Stream& Stream::null()
{
    static Stream s_stream(std::make_shared<Internal>(cudaStream_t{0}));
    return s_stream;
}
Stream::Stream(std::shared_ptr<Internal> impl)
{
    _impl = impl;
}

}
}
