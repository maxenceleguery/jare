# 3D Render Engine

3D render engine project in C++.
Require a Nvidia graphic card to use CUDA acceleration.

## Dependencies

```bash
$ sudo apt install libpng-dev
$ sudo apt install nvidia-cuda-toolkit
```

## Compilation

```bash
$ make all
```

## Execution

```bash
$ build/main
```

## Some results

![Simple render of cube](/cube.png)

![Raytraced render of cube](/cube4.png)

CUDA render almost 40 times faster than classic CPU render

![Raytraced chess knight](/knight2.png)

Took 2924.65s to render (64 samples per ray and 4 threads per ray)

Maxence Leguéry
