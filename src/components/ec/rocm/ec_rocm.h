/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_ROCM_H_
#define UCC_EC_ROCM_H_

#include "components/ec/base/ucc_ec_base.h"
#include "components/ec/ucc_ec_log.h"
#include "utils/ucc_mpool.h"
#include <hip/hip_runtime_api.h>

typedef enum ucc_ec_rocm_strm_task_mode {
    UCC_EC_ROCM_TASK_KERNEL,
    UCC_EC_ROCM_TASK_MEM_OPS,
    UCC_EC_ROCM_TASK_AUTO,
    UCC_EC_ROCM_TASK_LAST,
} ucc_ec_rocm_strm_task_mode_t;

typedef enum ucc_ec_rocm_task_stream_type {
    UCC_EC_ROCM_USER_STREAM,
    UCC_EC_ROCM_INTERNAL_STREAM,
    UCC_EC_ROCM_TASK_STREAM_LAST
} ucc_ec_rocm_task_stream_type_t;

typedef enum ucc_ec_task_status {
    UCC_EC_ROCM_TASK_COMPLETED,
    UCC_EC_ROCM_TASK_POSTED,
    UCC_EC_ROCM_TASK_STARTED,
    UCC_EC_ROCM_TASK_COMPLETED_ACK
} ucc_ec_task_status_t;

static inline ucc_status_t hip_error_to_ucc_status(hipError_t hip_err)
{
    switch(hip_err) {
    case hipSuccess:
        return UCC_OK;
    case hipErrorNotReady:
        return UCC_INPROGRESS;
    default:
        break;
    }
    return UCC_ERR_NO_MESSAGE;
}

typedef ucc_status_t (*ucc_ec_rocm_task_post_fn) (uint32_t *dev_status,
                                                  int blocking_wait,
                                                  hipStream_t stream);

typedef struct ucc_ec_rocm_config {
    ucc_ec_config_t                super;
    ucc_ec_rocm_strm_task_mode_t   strm_task_mode;
    ucc_ec_rocm_task_stream_type_t task_strm_type;
    int                            stream_blocking_wait;
} ucc_ec_rocm_config_t;

typedef struct ucc_ec_rocm {
    ucc_ec_base_t                  super;
    hipStream_t                    stream;
    ucc_mpool_t                    events;
    ucc_mpool_t                    strm_reqs;
    ucc_thread_mode_t              thread_mode;
    ucc_ec_rocm_strm_task_mode_t   strm_task_mode;
    ucc_ec_rocm_task_stream_type_t task_strm_type;
    ucc_ec_rocm_task_post_fn       post_strm_task;
    ucc_spinlock_t                 init_spinlock;
} ucc_ec_rocm_t;

typedef struct ucc_rocm_ec_event {
    hipEvent_t    event;
} ucc_ec_rocm_event_t;

typedef struct ucc_ec_rocm_stream_request {
    uint32_t            status;
    uint32_t           *dev_status;
    hipStream_t         stream;
} ucc_ec_rocm_stream_request_t;

extern ucc_ec_rocm_t ucc_ec_rocm;
#define ROCMCHECK(cmd) do {                                                    \
        hipError_t e = cmd;                                                    \
        if(e != hipSuccess) {                                                  \
            ec_error(&ucc_ec_rocm.super, "ROCm failed with ret:%d(%s)", e,     \
                     hipGetErrorString(e));                                    \
            return UCC_ERR_NO_MESSAGE;                                         \
        }                                                                      \
} while(0)

#define ROCM_FUNC(_func)                                                       \
    ({                                                                         \
        ucc_status_t _status = UCC_OK;                                         \
        do {                                                                   \
            hipError_t _result = (_func);                                      \
            if (hipSuccess != _result) {                                       \
                ec_error(&ucc_ec_rocm.super, "%s() failed: %s",                \
                       #_func, hipGetErrorString(_result));                    \
                _status = UCC_ERR_INVALID_PARAM;                               \
            }                                                                  \
        } while (0);                                                           \
        _status;                                                               \
    })

#define EC_ROCM_CONFIG                                                         \
    (ucc_derived_of(ucc_ec_rocm.super.config, ucc_ec_rocm_config_t))

#define UCC_EC_ROCM_INIT_STREAM() do {                                         \
    if (ucc_ec_rocm.stream == NULL) {                                          \
        hipError_t hip_st = hipSuccess;                                        \
        ucc_spin_lock(&ucc_ec_rocm.init_spinlock);                             \
        if (ucc_ec_rocm.stream == NULL) {                                      \
            hip_st = hipStreamCreateWithFlags(&ucc_ec_rocm.stream,             \
                                                hipStreamNonBlocking);         \
        }                                                                      \
        ucc_spin_unlock(&ucc_ec_rocm.init_spinlock);                           \
        if(hip_st != hipSuccess) {                                             \
            ec_error(&ucc_ec_rocm.super, "rocm failed with ret:%d(%s)",        \
                     hip_st, hipGetErrorString(hip_st));                       \
            return UCC_ERR_NO_MESSAGE;                                         \
        }                                                                      \
    }                                                                          \
} while(0)

#endif
