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


Stream::Stream() : _destroy(true)
{
    CUDA_CHECK_ERRORS(cudaStreamCreate(&_stream));
}
Stream::Stream(cudaStream_t stream) : _destroy(false), _stream(stream)
{
}
Stream::~Stream()
{
    if (_destroy) 
        CUDA_CHECK_ERRORS(cudaStreamDestroy(_stream));
}
void Stream::add_callback(Callback cb, void* user_data)
{
    CUDA_CHECK_ERRORS(cudaStreamAddCallback(
        _stream,
        cb,
        user_data,
        0
    ));
}
bool Stream::query()
{
    cudaError_t status = cudaStreamQuery(_stream);
    if (status != cudaSuccess && status != cudaErrorNotReady) {
        CUDA_CHECK_ERRORS(status);
    }
    return status == cudaSuccess;
}
void Stream::synchronize()
{
    CUDA_CHECK_ERRORS(cudaStreamSynchronize(_stream));
}
void Stream::wait_event(const Event& event)
{
    CUDA_CHECK_ERRORS(cudaStreamWaitEvent(_stream, event, 0));
}
Stream::operator cudaStream_t() const
{
    return _stream;
}

Stream& Stream::null()
{
    static Stream s_stream(cudaStream_t{0});
    return s_stream;
}

}
}
