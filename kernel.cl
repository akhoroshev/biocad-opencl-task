#include "config.h"

__kernel void coulomb(const __global float *charges,
                      const __global float4 *coords,
                      const __global float *scale, __global float *output) {
  int global_x = get_global_id(0);
  int global_y = get_global_id(1);

  int local_x = get_local_id(0);
  int local_y = get_local_id(1);

  int global_size_x = get_global_size(0);
  int global_size_y = get_global_size(1);

  int local_size_x = get_local_size(0);
  int local_size_y = get_local_size(1);

  int group_x = get_group_id(0);
  int group_y = get_group_id(1);

  int num_group_x = get_num_groups(0);
  int num_group_y = get_num_groups(1);

  __local float storage[GROUP_SIZE * GROUP_SIZE];

  float value = 0;
  if (global_x > global_y) {
    value = scale[global_y * global_size_x + global_x] * C *
            charges[global_x] * charges[global_y] /
            distance(coords[global_x], coords[global_y]);
  }
  storage[local_size_x * local_y + local_x] = value;

  barrier(CLK_LOCAL_MEM_FENCE);

  if (local_x == 0 && local_y == 0) {
    float4 sum = (float4)(0,0,0,0);
    int sz = local_size_y * local_size_x;
    int i = 0;
    for (int div4 = sz / 4; i < div4; i += 4) {
      float4 *p = (float4 *)(storage + i);
      sum += *p;
    }
    for (; i < sz; i++) {
      sum.x += storage[i];
    }
    output[group_x * num_group_y + group_y] = dot(sum, (float4)(1.0f, 1.0f, 1.0f, 1.0f));;
  }
}

__global int *incrementQueuePointer(__global int *queuePointer,
                                    __global int *queueStart, int queueOffset,
                                    int mod) {
  ptrdiff_t diff = queuePointer - queueStart - queueOffset + 1;
  diff %= mod;
  return queueStart + diff + queueOffset;
}

__kernel void bfs(const __global int *edges, const __global int *edgesCount,
                  __global int *queue, __global float *path) {
  const int vertex = get_global_id(0);
  const int vertexTotal = get_global_size(0);

  const int queueOffset = vertex * vertexTotal;

  __global int *queueBegin = queueOffset + queue;
  __global int *queueEnd = queueOffset + queue;

  float scale[3] = {0, 0, 0.5};

  // self path equal 0
  path[vertex * vertexTotal + vertex] = 0.0; // 0 bound
  *queueEnd = vertex;
  queueEnd = incrementQueuePointer(queueEnd, queue, queueOffset, vertexTotal);

#pragma unroll
  for (int iter = 0; iter < 3; iter++) {
    int queueSize = abs(queueEnd - queueBegin);

    for (int i = 0; i < queueSize; i++) {
      int popVertex = *queueBegin;
      queueBegin =
          incrementQueuePointer(queueBegin, queue, queueOffset, vertexTotal);

      for (int j = 0, count = edgesCount[popVertex]; j < count; j++) {
        int toVertex = edges[popVertex * vertexTotal + j];
        if (path[vertex * vertexTotal + toVertex] < 1.0)
          continue;

        path[vertex * vertexTotal + toVertex] = scale[iter];

        *queueEnd = toVertex;
        queueEnd =
            incrementQueuePointer(queueEnd, queue, queueOffset, vertexTotal);
      }
    }
  }
}
