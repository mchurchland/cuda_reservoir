#include <cstdio>

#include <cuda/version>        // CCCL version
#include <thrust/version.h>    // Thrust (part of CCCL)
#include <cub/version.cuh>     // CUB (part of CCCL)

int main() {
#ifdef CCCL_VERSION
    printf("CCCL_VERSION: %d\n", CCCL_VERSION);
#else
    printf("CCCL_VERSION not defined\n");
#endif

#ifdef CCCL_MAJOR_VERSION
    printf("CCCL: %d.%d.%d\n",
        CCCL_MAJOR_VERSION,
        CCCL_MINOR_VERSION,
        CCCL_PATCH_VERSION);
#endif

#ifdef THRUST_VERSION
    printf("THRUST_VERSION: %d\n", THRUST_VERSION);
#endif

#ifdef CUB_VERSION
    printf("CUB_VERSION: %d\n", CUB_VERSION);
#endif

    return 0;
}
