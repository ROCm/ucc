/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_rocm.h"
#include "utils/ucc_malloc.h"
#include "utils/arch/cpu.h"
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

static const char *stream_task_modes[] = {
    [UCC_MC_ROCM_TASK_KERNEL]  = "kernel",
    [UCC_MC_ROCM_TASK_MEM_OPS] = "driver",
    [UCC_MC_ROCM_TASK_AUTO]    = "auto",
    [UCC_MC_ROCM_TASK_LAST]    = NULL
};

static const char *task_stream_types[] = {
    [UCC_MC_ROCM_USER_STREAM]      = "user",
    [UCC_MC_ROCM_INTERNAL_STREAM]  = "ucc",
    [UCC_MC_ROCM_TASK_STREAM_LAST] = NULL
};

static ucc_config_field_t ucc_mc_rocm_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_mc_rocm_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_mc_config_table)},

    {"REDUCE_NUM_BLOCKS", "auto",
     "Number of thread blocks to use for reduction",
     ucc_offsetof(ucc_mc_rocm_config_t, reduce_num_blocks),
     UCC_CONFIG_TYPE_ULUNITS},

    {"STREAM_TASK_MODE", "auto",
     "Mechanism to create stream dependency\n"
     "kernel - use waiting kernel\n"
     "driver - use driver MEM_OPS\n"
     "auto   - runtime automatically chooses best one",
     ucc_offsetof(ucc_mc_rocm_config_t, strm_task_mode),
     UCC_CONFIG_TYPE_ENUM(stream_task_modes)},

    {"TASK_STREAM", "user",
     "Stream for rocm task\n"
     "user - user stream provided in execution engine context\n"
     "ucc  - ucc library internal stream",
     ucc_offsetof(ucc_mc_rocm_config_t, task_strm_type),
     UCC_CONFIG_TYPE_ENUM(task_stream_types)},

    {"STREAM_BLOCKING_WAIT", "1",
     "Stream is blocked until collective operation is done",
     ucc_offsetof(ucc_mc_rocm_config_t, stream_blocking_wait),
     UCC_CONFIG_TYPE_UINT},

    {"MPOOL_ELEM_SIZE", "1Mb", "The size of each element in mc rocm mpool",
     ucc_offsetof(ucc_mc_rocm_config_t, mpool_elem_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"MPOOL_MAX_ELEMS", "8", "The max amount of elements in mc rocm mpool",
     ucc_offsetof(ucc_mc_rocm_config_t, mpool_max_elems), UCC_CONFIG_TYPE_UINT},

    {NULL}
};

static ucc_status_t ucc_mc_rocm_stream_req_mpool_chunk_malloc(ucc_mpool_t *mp,
                                                              size_t *size_p,
                                                              void ** chunk_p)
{
    ucc_status_t status;

    status = ROCM_FUNC(hipHostMalloc((void**)chunk_p, *size_p,
                       hipHostMallocMapped));
    return status;
}

static void ucc_mc_rocm_stream_req_mpool_chunk_free(ucc_mpool_t *mp,
                                                    void *       chunk)
{
    hipHostFree(chunk);
}

static void ucc_mc_rocm_stream_req_init(ucc_mpool_t *mp, void *obj, void *chunk)
{
    ucc_mc_rocm_stream_request_t *req = (ucc_mc_rocm_stream_request_t*) obj;

    ROCM_FUNC(hipHostGetDevicePointer(
                  (void**)(&req->dev_status), (void *)&req->status, 0));
}

static ucc_mpool_ops_t ucc_mc_rocm_stream_req_mpool_ops = {
    .chunk_alloc   = ucc_mc_rocm_stream_req_mpool_chunk_malloc,
    .chunk_release = ucc_mc_rocm_stream_req_mpool_chunk_free,
    .obj_init      = ucc_mc_rocm_stream_req_init,
    .obj_cleanup   = NULL
};

static void ucc_mc_rocm_event_init(ucc_mpool_t *mp, void *obj, void *chunk)
{
    ucc_mc_rocm_event_t *base = (ucc_mc_rocm_event_t *) obj;

    if (ucc_unlikely(
          hipSuccess !=
          hipEventCreateWithFlags(&base->event, hipEventDisableTiming))) {
      mc_error(&ucc_mc_rocm.super, "hipEventCreateWithFlags Failed");
    }
}

static void ucc_mc_rocm_event_cleanup(ucc_mpool_t *mp, void *obj)
{
    ucc_mc_rocm_event_t *base = (ucc_mc_rocm_event_t *) obj;
    if (ucc_unlikely(hipSuccess != hipEventDestroy(base->event))) {
        mc_error(&ucc_mc_rocm.super, "hipEventDestroy Failed");
    }
}

static ucc_mpool_ops_t ucc_mc_rocm_event_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_mc_rocm_event_init,
    .obj_cleanup   = ucc_mc_rocm_event_cleanup,
};

static ucc_status_t ucc_mc_rocm_post_kernel_stream_task(uint32_t *status,
                                                 int blocking_wait,
                                                 hipStream_t stream)
{
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_init(const ucc_mc_params_t *mc_params)
{
    ucc_mc_rocm_config_t *cfg = MC_ROCM_CONFIG;
    struct hipDeviceProp_t prop;
    ucc_status_t status;
    int device, num_devices;
    hipError_t rocm_st;

    ucc_mc_rocm.stream = NULL;
    ucc_strncpy_safe(ucc_mc_rocm.super.config->log_component.name,
                     ucc_mc_rocm.super.super.name,
                     sizeof(ucc_mc_rocm.super.config->log_component.name));
    ucc_mc_rocm.thread_mode = mc_params->thread_mode;
    rocm_st = hipGetDeviceCount(&num_devices);
    if ((rocm_st != hipSuccess) || (num_devices == 0)) {
        mc_info(&ucc_mc_rocm.super, "rocm devices are not found");
        return UCC_ERR_NO_RESOURCE;
    }
    ROCMCHECK(hipGetDevice(&device));
    ROCMCHECK(hipGetDeviceProperties(&prop, device));
    cfg->reduce_num_threads = prop.maxThreadsPerBlock;
    if (cfg->reduce_num_blocks != UCC_ULUNITS_AUTO) {
        if (prop.maxGridSize[0] < cfg->reduce_num_blocks) {
            mc_warn(&ucc_mc_rocm.super, "number of blocks is too large, "
                    "max supported %d", prop.maxGridSize[0]);
            cfg->reduce_num_blocks = prop.maxGridSize[0];
        }
    }

    /*create event pool */
    status = ucc_mpool_init(&ucc_mc_rocm.events, 0, sizeof(ucc_mc_rocm_event_t),
                            0, UCC_CACHE_LINE_SIZE, 16, UINT_MAX,
                            &ucc_mc_rocm_event_mpool_ops, UCC_THREAD_MULTIPLE,
                            "ROCM Event Objects");
    if (status != UCC_OK) {
        mc_error(&ucc_mc_rocm.super, "Error to create event pool");
        return status;
    }

    /* create request pool */
    status = ucc_mpool_init(
        &ucc_mc_rocm.strm_reqs, 0, sizeof(ucc_mc_rocm_stream_request_t), 0,
        UCC_CACHE_LINE_SIZE, 16, UINT_MAX, &ucc_mc_rocm_stream_req_mpool_ops,
        UCC_THREAD_MULTIPLE, "ROCM Event Objects");
    if (status != UCC_OK) {
        mc_error(&ucc_mc_rocm.super, "Error to create event pool");
        return status;
    }

    if (cfg->strm_task_mode == UCC_MC_ROCM_TASK_KERNEL) {
        ucc_mc_rocm.strm_task_mode = UCC_MC_ROCM_TASK_KERNEL;
        ucc_mc_rocm.post_strm_task = ucc_mc_rocm_post_kernel_stream_task;
    } else {
        if (cfg->strm_task_mode == UCC_MC_ROCM_TASK_AUTO) {
            ucc_mc_rocm.strm_task_mode = UCC_MC_ROCM_TASK_KERNEL;
            ucc_mc_rocm.post_strm_task = ucc_mc_rocm_post_kernel_stream_task;
        } else {
            mc_error(&ucc_mc_rocm.super, "ROCM MEM OPS are not supported");
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    ucc_mc_rocm.task_strm_type = cfg->task_strm_type;

    // lock assures single mpool initiation when multiple threads concurrently execute
    // different collective operations thus concurrently entering init function.
    ucc_spinlock_init(&ucc_mc_rocm.init_spinlock, 0);

    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_get_attr(ucc_mc_attr_t *mc_attr)
{
    if (mc_attr->field_mask & UCC_MC_ATTR_FIELD_THREAD_MODE) {
        mc_attr->thread_mode = ucc_mc_rocm.thread_mode;
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_mem_alloc(ucc_mc_buffer_header_t **h_ptr,
                                          size_t                   size)
{
    hipError_t st;

    ucc_mc_buffer_header_t *h = ucc_malloc(sizeof(ucc_mc_buffer_header_t), "mc rocm");
    if (ucc_unlikely(!h)) {
      mc_error(&ucc_mc_rocm.super, "failed to allocate %zd bytes",
               sizeof(ucc_mc_buffer_header_t));
    }
    st = hipMalloc(&h->addr, size);
    if (ucc_unlikely(st != hipSuccess)) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to allocate %zd bytes, "
                 "rocm error %d(%s)",
                 size, st, hipGetErrorString(st));
        ucc_free(h);
        return UCC_ERR_NO_MEMORY;
    }
    h->from_pool = 0;
    h->mt        = UCC_MEMORY_TYPE_ROCM;
    *h_ptr       = h;
    mc_trace(&ucc_mc_rocm.super, "allocated %ld bytes with hipMalloc", size);
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_mem_pool_alloc(ucc_mc_buffer_header_t **h_ptr,
                                               size_t                   size)
{
    ucc_mc_buffer_header_t *h = NULL;
    if (size <= MC_ROCM_CONFIG->mpool_elem_size) {
        h = (ucc_mc_buffer_header_t *)ucc_mpool_get(&ucc_mc_rocm.mpool);
    }
    if (!h) {
        // Slow path
        return ucc_mc_rocm_mem_alloc(h_ptr, size);
    }
    if (ucc_unlikely(!h->addr)){
        return UCC_ERR_NO_MEMORY;
    }
    *h_ptr = h;
    mc_trace(&ucc_mc_rocm.super, "allocated %ld bytes from rocm mpool", size);
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_chunk_alloc(ucc_mpool_t *mp,
                                            size_t *size_p,
                                            void **chunk_p)
{
    *chunk_p = ucc_malloc(*size_p, "mc rocm");
    if (!*chunk_p) {
        mc_error(&ucc_mc_rocm.super, "failed to allocate %zd bytes", *size_p);
        return UCC_ERR_NO_MEMORY;
    }

    return UCC_OK;
}

static void ucc_mc_rocm_chunk_init(ucc_mpool_t *mp,
                                   void *obj, void *chunk)
{
    ucc_mc_buffer_header_t *h = (ucc_mc_buffer_header_t *)obj;
    hipError_t st = hipMalloc(&h->addr, MC_ROCM_CONFIG->mpool_elem_size);
    if (st != hipSuccess) {
        // h->addr will be 0 so ucc_mc_rocm_mem_alloc_pool function will
        // return UCC_ERR_NO_MEMORY. As such mc_error message is suffice.
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to allocate %zd bytes, "
                 "rocm error %d(%s)",
                 MC_ROCM_CONFIG->mpool_elem_size, st, hipGetErrorString(st));
    }
    h->from_pool = 1;
    h->mt        = UCC_MEMORY_TYPE_ROCM;
}

static void ucc_mc_rocm_chunk_release(ucc_mpool_t *mp, void *chunk)
{
    ucc_free(chunk);
}

static void ucc_mc_rocm_chunk_cleanup(ucc_mpool_t *mp, void *obj)
{
    ucc_mc_buffer_header_t *h = (ucc_mc_buffer_header_t *)obj;
    hipError_t st;
    st = hipFree(h->addr);
    if (st != hipSuccess) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to free mem at %p, "
                 "rocm error %d(%s)",
                 obj, st, hipGetErrorString(st));
    }
}

static ucc_mpool_ops_t ucc_mc_ops = {.chunk_alloc   = ucc_mc_rocm_chunk_alloc,
                                     .chunk_release = ucc_mc_rocm_chunk_release,
                                     .obj_init      = ucc_mc_rocm_chunk_init,
                                     .obj_cleanup = ucc_mc_rocm_chunk_cleanup};

static ucc_status_t ucc_mc_rocm_mem_free(ucc_mc_buffer_header_t *h_ptr)
{
    hipError_t st;
    st = hipFree(h_ptr->addr);
    if (ucc_unlikely(st != hipSuccess)) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to free mem at %p, "
                 "hip error %d(%s)",
                 h_ptr->addr, st, hipGetErrorString(st));
        return UCC_ERR_NO_MESSAGE;
    }
    ucc_free(h_ptr);
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_mem_pool_free(ucc_mc_buffer_header_t *h_ptr)
{
    if (!h_ptr->from_pool) {
        return ucc_mc_rocm_mem_free(h_ptr);
    }
    ucc_mpool_put(h_ptr);
    return UCC_OK;
}

static ucc_status_t
ucc_mc_rocm_mem_pool_alloc_with_init(ucc_mc_buffer_header_t **h_ptr,
                                     size_t                   size)
{
    // lock assures single mpool initiation when multiple threads concurrently execute
    // different collective operations thus concurrently entering init function.
    ucc_spin_lock(&ucc_mc_rocm.init_spinlock);

    if (MC_ROCM_CONFIG->mpool_max_elems == 0) {
        ucc_mc_rocm.super.ops.mem_alloc = ucc_mc_rocm_mem_alloc;
        ucc_mc_rocm.super.ops.mem_free  = ucc_mc_rocm_mem_free;
        ucc_spin_unlock(&ucc_mc_rocm.init_spinlock);
        return ucc_mc_rocm_mem_alloc(h_ptr, size);
    }

    if (!ucc_mc_rocm.mpool_init_flag) {
        ucc_status_t status = ucc_mpool_init(
            &ucc_mc_rocm.mpool, 0, sizeof(ucc_mc_buffer_header_t), 0,
            UCC_CACHE_LINE_SIZE, 1, MC_ROCM_CONFIG->mpool_max_elems,
            &ucc_mc_ops, ucc_mc_rocm.thread_mode, "mc rocm mpool buffers");
        if (status != UCC_OK) {
            ucc_spin_unlock(&ucc_mc_rocm.init_spinlock);
            return status;
        }
        ucc_mc_rocm.super.ops.mem_alloc = ucc_mc_rocm_mem_pool_alloc;
        ucc_mc_rocm.mpool_init_flag     = 1;
    }
    ucc_spin_unlock(&ucc_mc_rocm.init_spinlock);
    return ucc_mc_rocm_mem_pool_alloc(h_ptr, size);
}

static ucc_status_t ucc_mc_rocm_memcpy(void *dst, const void *src, size_t len,
                                       ucc_memory_type_t dst_mem,
                                       ucc_memory_type_t src_mem)
{
    hipError_t st;
    ucc_assert(dst_mem == UCC_MEMORY_TYPE_ROCM ||
               src_mem == UCC_MEMORY_TYPE_ROCM);

    UCC_MC_ROCM_INIT_STREAM();
    st = hipMemcpyAsync(dst, src, len, hipMemcpyDefault, ucc_mc_rocm.stream);
    if (ucc_unlikely(st != hipSuccess)) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to launch hipMemcpyAsync,  dst %p, src %p, len %zd "
                 "hip error %d(%s)",
                 dst, src, len, st, hipGetErrorString(st));
        return UCC_ERR_NO_MESSAGE;
    }
    st = hipStreamSynchronize(ucc_mc_rocm.stream);
    if (ucc_unlikely(st != hipSuccess)) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to synchronize mc_rocm.stream "
                 "hip error %d(%s)",
                 st, hipGetErrorString(st));
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_mem_query(const void *ptr,
                                          ucc_mem_attr_t *mem_attr)
{
    struct hipPointerAttribute_t attr;
    hipError_t                   st;
    ucc_memory_type_t            mem_type;
    void                        *base_address;
    size_t                       alloc_length;

    if (!(mem_attr->field_mask & (UCC_MEM_ATTR_FIELD_MEM_TYPE     |
                                  UCC_MEM_ATTR_FIELD_BASE_ADDRESS |
                                  UCC_MEM_ATTR_FIELD_ALLOC_LENGTH))) {
        return UCC_OK;
    }

    if (mem_attr->field_mask & UCC_MEM_ATTR_FIELD_MEM_TYPE) {
        st = hipPointerGetAttributes(&attr, ptr);
        if (st != hipSuccess) {
            hipGetLastError();
            return UCC_ERR_NOT_SUPPORTED;
        }
        switch (attr.memoryType) {
        case hipMemoryTypeHost:
            mem_type = (attr.isManaged ? UCC_MEMORY_TYPE_ROCM_MANAGED : UCC_MEMORY_TYPE_HOST);
            break;
        case hipMemoryTypeDevice:
            mem_type = UCC_MEMORY_TYPE_ROCM;
            break;
        default:
            return UCC_ERR_NOT_SUPPORTED;
        }
        mem_attr->mem_type = mem_type;
    }

    if (mem_attr->field_mask & (UCC_MEM_ATTR_FIELD_ALLOC_LENGTH |
                                UCC_MEM_ATTR_FIELD_BASE_ADDRESS)) {
      st = hipMemGetAddressRange((hipDeviceptr_t*)&base_address,
                                      &alloc_length, (hipDeviceptr_t)ptr);
      if (st != hipSuccess) {
        mc_error(&ucc_mc_rocm.super,
                 "hipMemGetAddressRange(%p) error: %d(%s)",
                 ptr, st, hipGetErrorString(st));
        return UCC_ERR_NOT_SUPPORTED;
      }

      mem_attr->base_address = base_address;
      mem_attr->alloc_length = alloc_length;
    }

    return UCC_OK;
}

ucc_status_t ucc_ee_rocm_task_post(void *ee_stream, void **ee_req)
{
    ucc_mc_rocm_stream_request_t *req;
    ucc_mc_rocm_event_t *rocm_event;
    ucc_status_t status;
    ucc_mc_rocm_config_t *cfg = MC_ROCM_CONFIG;

    UCC_MC_ROCM_INIT_STREAM();
    req = ucc_mpool_get(&ucc_mc_rocm.strm_reqs);
    ucc_assert(req);
    req->status = UCC_MC_ROCM_TASK_POSTED;
    req->stream = (hipStream_t)ee_stream;

    if (ucc_mc_rocm.task_strm_type == UCC_MC_ROCM_USER_STREAM) {
        status = ucc_mc_rocm.post_strm_task(req->dev_status,
                                            cfg->stream_blocking_wait,
                                            req->stream);
        if (status != UCC_OK) {
            goto free_req;
        }
    } else {
        rocm_event = ucc_mpool_get(&ucc_mc_rocm.events);
        ucc_assert(rocm_event);
        ROCMCHECK(hipEventRecord(rocm_event->event, req->stream));
        ROCMCHECK(hipStreamWaitEvent(ucc_mc_rocm.stream, rocm_event->event, 0));
        status = ucc_mc_rocm.post_strm_task(req->dev_status,
                                            cfg->stream_blocking_wait,
                                            ucc_mc_rocm.stream);
        if (ucc_unlikely(status != UCC_OK)) {
            goto free_event;
        }
        ROCMCHECK(hipEventRecord(rocm_event->event, ucc_mc_rocm.stream));
        ROCMCHECK(hipStreamWaitEvent(req->stream, rocm_event->event, 0));
        ucc_mpool_put(rocm_event);
    }

    *ee_req = (void *) req;

    mc_info(&ucc_mc_rocm.super, "ROCM stream task posted on \"%s\" stream. req:%p",
            task_stream_types[ucc_mc_rocm.task_strm_type], req);

    return UCC_OK;

free_event:
    ucc_mpool_put(rocm_event);
free_req:
    ucc_mpool_put(req);
    return status;
}

ucc_status_t ucc_ee_rocm_task_query(void *ee_req)
{
    ucc_mc_rocm_stream_request_t *req = ee_req;

    /* ee task might be only in POSTED, STARTED or COMPLETED_ACK state
       COMPLETED state is used by ucc_ee_rocm_task_end function to request
       stream unblock*/
    ucc_assert(req->status != UCC_MC_ROCM_TASK_COMPLETED);
    if (req->status == UCC_MC_ROCM_TASK_POSTED) {
        return UCC_INPROGRESS;
    }
    mc_info(&ucc_mc_rocm.super, "ROCM stream task started. req:%p", req);
    return UCC_OK;
}

ucc_status_t ucc_ee_rocm_task_end(void *ee_req)
{
    ucc_mc_rocm_stream_request_t *req = ee_req;
    volatile ucc_mc_task_status_t *st = &req->status;

    /* can be safely ended only if it's in STARTED or COMPLETED_ACK state */
    ucc_assert((*st != UCC_MC_ROCM_TASK_POSTED) &&
               (*st != UCC_MC_ROCM_TASK_COMPLETED));
    if (*st == UCC_MC_ROCM_TASK_STARTED) {
        *st = UCC_MC_ROCM_TASK_COMPLETED;
        while(*st != UCC_MC_ROCM_TASK_COMPLETED_ACK) { }
    }
    ucc_mpool_put(req);
    mc_info(&ucc_mc_rocm.super, "ROCM stream task done. req:%p", req);
    return UCC_OK;
}

ucc_status_t ucc_ee_rocm_create_event(void **event)
{
    ucc_mc_rocm_event_t *rocm_event;

    rocm_event = ucc_mpool_get(&ucc_mc_rocm.events);
    ucc_assert(rocm_event);
    *event = rocm_event;
    return UCC_OK;
}

ucc_status_t ucc_ee_rocm_destroy_event(void *event)
{
    ucc_mc_rocm_event_t *rocm_event = event;

    ucc_mpool_put(rocm_event);
    return UCC_OK;
}

ucc_status_t ucc_ee_rocm_event_post(void *ee_context, void *event)
{
    hipStream_t stream = (hipStream_t )ee_context;
    ucc_mc_rocm_event_t *rocm_event = event;

    ROCMCHECK(hipEventRecord(rocm_event->event, stream));
    return UCC_OK;
}

ucc_status_t ucc_ee_rocm_event_test(void *event)
{
    hipError_t hip_err;
    ucc_mc_rocm_event_t *rocm_event = event;

    hip_err = hipEventQuery(rocm_event->event);
    if (ucc_unlikely((hip_err != hipSuccess) &&
                     (hip_err != hipErrorNotReady))) {
        ROCMCHECK(hip_err);
    }
    return hip_error_to_ucc_status(hip_err);
}

static ucc_status_t ucc_mc_rocm_finalize()
{
    if (ucc_mc_rocm.stream != NULL) {
        ROCMCHECK(hipStreamDestroy(ucc_mc_rocm.stream));
        ucc_mc_rocm.stream = NULL;
    }
    if (ucc_mc_rocm.mpool_init_flag) {
        ucc_mpool_cleanup(&ucc_mc_rocm.mpool, 1);
        ucc_mc_rocm.mpool_init_flag     = 0;
        ucc_mc_rocm.super.ops.mem_alloc = ucc_mc_rocm_mem_pool_alloc_with_init;
    }
    ucc_spinlock_destroy(&ucc_mc_rocm.init_spinlock);
    return UCC_OK;
}

ucc_mc_rocm_t ucc_mc_rocm = {
    .super.super.name             = "rocm mc",
    .super.ref_cnt                = 0,
    .super.ee_type                = UCC_EE_CUDA_STREAM,
    .super.type                   = UCC_MEMORY_TYPE_ROCM,
    .super.init                   = ucc_mc_rocm_init,
    .super.get_attr               = ucc_mc_rocm_get_attr,
    .super.finalize               = ucc_mc_rocm_finalize,
    .super.ops.mem_query          = ucc_mc_rocm_mem_query,
    .super.ops.mem_alloc          = ucc_mc_rocm_mem_pool_alloc_with_init,
    .super.ops.mem_free           = ucc_mc_rocm_mem_pool_free,
    .super.ops.memcpy             = ucc_mc_rocm_memcpy,
    .super.config_table =
        {
            .name   = "ROCM memory component",
            .prefix = "MC_ROCM_",
            .table  = ucc_mc_rocm_config_table,
            .size   = sizeof(ucc_mc_rocm_config_t),
        },
    .super.ee_ops.ee_task_post     = ucc_ee_rocm_task_post,
    .super.ee_ops.ee_task_query    = ucc_ee_rocm_task_query,
    .super.ee_ops.ee_task_end      = ucc_ee_rocm_task_end,
    .super.ee_ops.ee_create_event  = ucc_ee_rocm_create_event,
    .super.ee_ops.ee_destroy_event = ucc_ee_rocm_destroy_event,
    .super.ee_ops.ee_event_post    = ucc_ee_rocm_event_post,
    .super.ee_ops.ee_event_test    = ucc_ee_rocm_event_test,
    .mpool_init_flag               = 0,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_rocm.super.config_table,
                                &ucc_config_global_list);
