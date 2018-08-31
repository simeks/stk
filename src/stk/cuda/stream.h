#pragma once

#ifdef STK_USE_CUDA

#include "cuda.h"

namespace stk
{
    namespace cuda
    {
        class Stream;
        class Event
        {
        public:
            // flags : See cudaEventCreateWithFlags for flags
            Event(uint32_t flags = cudaEventDefault);
            ~Event();

            // Queries the event for completion status.
            // Returns true if work captured by the event has been completed.
            bool query();
            
            // Records the event into the given stream
            void record(const Stream& stream);

            // Waits for the event to complete
            void synchronize();

            operator cudaEvent_t() const;

            // Returns the elapsed time between two events in milliseconds
            static float elapsed(const Event& start, const Event& end);

        private:
            Event(const Event&);
            Event& operator=(const Event&);

            cudaEvent_t _event;
        };

        // Wrapper around cudaStream_t
        // Will automatically create and destroy a cudaStream_t object within
        //  the scope of the wrapper.
        class Stream
        {
        public:
            typedef void (*Callback)(cudaStream_t stream, cudaError_t status, void* user_data);

            Stream();
            Stream(cudaStream_t stream);
            ~Stream();

            // Adds a callback to the stream. Added callbacks will be executed
            //  only once.
            // Read CUDA docs (cudaStreamAddCallback) for an in-depth description
            //  on the behaviour and restrictions of stream callbacks.
            void add_callback(Callback cb, void* user_data);

            // Queries the stream for completion status
            // Returns true if all tasks in the stream are completed
            bool query();

            // Waits for all tasks in stream to complete
            void synchronize();

            // Make the stream wait for an event
            void wait_event(const Event& event);

            operator cudaStream_t() const;

            // Default stream
            static Stream& null();

        private:
            Stream(const Stream&);
            Stream& operator=(const Stream&);

            bool _destroy;
            cudaStream_t _stream; 
        };
        
    }
}
#endif // STK_USE_CUDA
