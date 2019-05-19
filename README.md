### How to run

- install OpenCL 
- clone repo
- `make && ./biocad_opencl_task`

### Results

On my laptop without dedicated graphics card:
```

-------------------------------------------------
Device: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz 

Atoms loaded: 6904
Time for preparing buffers: 332.42 ms
Time for executing: 41.0409 ms
Energy: -64470.1

-------------------------------------------------
Device: Intel(R) HD Graphics Kabylake ULT GT2 

Atoms loaded: 6904
Time for preparing buffers: 253.995 ms
Time for executing: 155.638 ms
Energy: -64470.1

-------------------------------------------------
Default algorithm without using OpenCL

Atoms loaded: 6904
Time for preparing buffers: 177.578 ms
Time for executing: 592.611 ms
Energy: -64470.1

```

`Time for executing` is the time spent on calculations

