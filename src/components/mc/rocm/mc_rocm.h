/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_ROCM_H_
#define UCC_MC_ROCM_H_

#include "components/mc/base/ucc_mc_base.h"
#include "components/mc/ucc_mc_log.h"
#include "utils/ucc_mpool.h"
#include <hip/hip_runtime_api.h>

typedef enum ucc_mc_rocm_strm_task_mode {
    UCC_MC_ROCM_TASK_KERNEL,
    UCC_MC_ROCM_TASK_MEM_OPS,
    UCC_MC_ROCM_TASK_AUTO,
    UCC_MC_ROCM_TASK_LAST,
} ucc_mc_rocm_strm_task_mode_t;

typedef enum ucc_mc_rocm_task_stream_type {
    UCC_MC_ROCM_USER_STREAM,
    UCC_MC_ROCM_INTERNAL_STREAM,
    UCC_MC_ROCM_TASK_STREAM_LAST
} ucc_mc_rocm_task_stream_type_t;

typedef enum ucc_mc_task_status {
    UCC_MC_ROCM_TASK_COMPLETED,
    UCC_MC_ROCM_TASK_POSTED,
    UCC_MC_ROCM_TASK_STARTED,
    UCC_MC_ROCM_TASK_COMPLETED_ACK
} ucc_mc_task_status_t;

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

typedef ucc_status_t (*ucc_mc_rocm_task_post_fn) (uint32_t *dev_status,
                                                  int blocking_wait,
                                                  hipStream_t stream);

typedef struct ucc_mc_rocm_config {
    ucc_mc_config_t                super;
    unsigned long                  reduce_num_blocks;
    int                            reduce_num_threads;
    ucc_mc_rocm_strm_task_mode_t   strm_task_mode;
    ucc_mc_rocm_task_stream_type_t task_strm_type;
    int                            stream_blocking_wait;
} ucc_mc_rocm_config_t;

typedef struct ucc_mc_rocm {
    ucc_mc_base_t                  super;
    hipStream_t                    stream;
    ucc_mpool_t                    events;
    ucc_mpool_t                    strm_reqs;
    ucc_mc_rocm_strm_task_mode_t   strm_task_mode;
    ucc_mc_rocm_task_stream_type_t task_strm_type;
    ucc_mc_rocm_task_post_fn       post_strm_task;
} ucc_mc_rocm_t;

typedef struct ucc_rocm_mc_event {
    hipEvent_t    event;
} ucc_mc_rocm_event_t;

typedef struct ucc_mc_rocm_stream_request {
    uint32_t            status;
    uint32_t           *dev_status;
    hipStream_t         stream;
} ucc_mc_rocm_stream_request_t;

ucc_status_t ucc_mc_rocm_reduce(const void *src1, const void *src2,
                                void *dst, size_t count, ucc_datatype_t dt,
                                ucc_reduction_op_t op);

ucc_status_t ucc_mc_rocm_reduce_multi(const void *src1, const void *src2,
                                      void *dst, size_t size, size_t count,
                                      size_t stride, ucc_datatype_t dt,
                                      ucc_reduction_op_t op);

extern ucc_mc_rocm_t ucc_mc_rocm;
#define ROCMCHECK(cmd) do {                                                    \
        hipError_t e = cmd;                                                    \
        if(e != hipSuccess) {                                                  \
            mc_error(&ucc_mc_rocm.super, "ROCm failed with ret:%d(%s)", e,     \
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
                mc_error(&ucc_mc_rocm.super, "%s() failed: %s",                \
                       #_func, hipGetErrorString(_result));                    \
                _status = UCC_ERR_INVALID_PARAM;                               \
            }                                                                  \
        } while (0);                                                           \
        _status;                                                               \
    })

#define ROCMDRV_FUNC(_func)                                                    \
    ({                                                                         \
        ucc_status_t _status = UCS_OK;                                         \
        do {                                                                   \
            hsa_status_t _result = (_func);                                        \
            const char *hip_err_str;                                            \
            if (HSA_SUCCESS != _result) {                                     \
                hipGetErrorString(_result, &hip_err_str);                        \
                mc_error(&ucc_mc_rocm.super, "%s() failed: %s",                \
                        #_func, hip_err_str);                                   \
                _status = UCC_ERR_INVALID_PARAM;                               \
            }                                                                  \
        } while (0);                                                           \
        _status;                                                               \
    })

#define MC_ROCM_CONFIG                                                         \
    (ucc_derived_of(ucc_mc_rocm.super.config, ucc_mc_rocm_config_t))

#endif
