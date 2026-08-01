Convert two real models to the Native4D four-axis dialect, and verify the
conversion map for one of them. Gated on PT2_DATA; run with `make pt2.runtest`
after `make pt2.download-cram`. See .ai/native4d_plan.md stage 8.

The models are chosen for what they exercise, per .ai/native4d_design.md §11
stage 6: ResNet-18 for the plain convolution and residual shape, MobileNet-v2
for depthwise convolution and the scalar-bounded activation.

`--fold` is not optional here. Conversion is only meaningful on a CANONICAL
graph: the importer emits every conv weight right-aligned from ATen's
[out, in, kh, kw], which lands on D/H/W/C, so an unpipelined graph has non-unit
D on every weight and is outside the dialect entirely. Constant folding is what
materialises the relaid weights, and batch-norm folding is what removes the
BatchNorms that would otherwise each need their own precomputation.

ResNet-18, whole structure, with each edge's verification verdict beside it.
Twenty-one Conv2D is twenty convolutions plus the final Linear, legalized to a
1x1 — the Native weight is already [Out,1,1,1,1,In], so that is a params-only
rewrite. The single MeanKeepDims is the global average pool.

The verdicts are what make this more than a shape dump. `proved (structural)`
holds for every input; `proved (structural, for these constants)` is the weaker
tier that substitutes the folded weights; `too large` is the budget refusing an
activation tensor before doing any work, which is a decline rather than a doubt.
What must not appear anywhere is `refuted`.

  $ ../bin/native_graph.exe to4d --pt2 "$PT2_DATA/resnet18/resnet18.pt2" --fold --verify-symbolic standard
  canonical native: nodes=49
  native4d: nodes=49 inputs=43
    Add                8
    Conv2D             21
    Max_pool2d         1
    MeanKeepDims       1
    Permute4           1
    Relu               17
  map verification: 92 clusters: 28 proved (for these constants) [sampled 8], 23 proved (structural) [sampled 8], 1 tested (coefficients agree) [sampled 8], 40 unproved (too large)
  graph4
  inputs: [t61 {proved (structural, for these constants) [sampled 8]} [C=1000],
  t122 {unproved: too large (150528 coords)} [H=3 W=224 C=224],
  t295 {unproved: too large (512000 coords)} [N=1000 T=1 D=1 H=1 W=1 C=512],
  t297 {proved (structural, for these constants) [sampled 8]} [N=64 T=1 D=1 H=7 W=7 C=3],
  t298 {proved (structural, for these constants) [sampled 8]} [C=64],
  t299 {proved (structural, for these constants) [sampled 8]} [N=64 T=1 D=1 H=3 W=3 C=64],
  t300 {proved (structural, for these constants) [sampled 8]} [C=64],
  t301 {proved (structural, for these constants) [sampled 8]} [N=64 T=1 D=1 H=3 W=3 C=64],
  t302 {proved (structural, for these constants) [sampled 8]} [C=64],
  t303 {proved (structural, for these constants) [sampled 8]} [N=64 T=1 D=1 H=3 W=3 C=64],
  t304 {proved (structural, for these constants) [sampled 8]} [C=64],
  t305 {proved (structural, for these constants) [sampled 8]} [N=64 T=1 D=1 H=3 W=3 C=64],
  t306 {proved (structural, for these constants) [sampled 8]} [C=64],
  t307 {unproved: too large (73728 coords)} [N=128 T=1 D=1 H=3 W=3 C=64],
  t308 {proved (structural, for these constants) [sampled 8]} [C=128],
  t309 {proved (structural, for these constants) [sampled 8]} [N=128 T=1 D=1 H=1 W=1 C=64],
  t310 {proved (structural, for these constants) [sampled 8]} [C=128],
  t311 {unproved: too large (147456 coords)} [N=128 T=1 D=1 H=3 W=3 C=128],
  t312 {proved (structural, for these constants) [sampled 8]} [C=128],
  t313 {unproved: too large (147456 coords)} [N=128 T=1 D=1 H=3 W=3 C=128],
  t314 {proved (structural, for these constants) [sampled 8]} [C=128],
  t315 {unproved: too large (147456 coords)} [N=128 T=1 D=1 H=3 W=3 C=128],
  t316 {proved (structural, for these constants) [sampled 8]} [C=128],
  t317 {unproved: too large (294912 coords)} [N=256 T=1 D=1 H=3 W=3 C=128],
  t318 {proved (structural, for these constants) [sampled 8]} [C=256],
  t319 {proved (structural, for these constants) [sampled 8]} [N=256 T=1 D=1 H=1 W=1 C=128],
  t320 {proved (structural, for these constants) [sampled 8]} [C=256],
  t321 {unproved: too large (589824 coords)} [N=256 T=1 D=1 H=3 W=3 C=256],
  t322 {proved (structural, for these constants) [sampled 8]} [C=256],
  t323 {unproved: too large (589824 coords)} [N=256 T=1 D=1 H=3 W=3 C=256],
  t324 {proved (structural, for these constants) [sampled 8]} [C=256],
  t325 {unproved: too large (589824 coords)} [N=256 T=1 D=1 H=3 W=3 C=256],
  t326 {proved (structural, for these constants) [sampled 8]} [C=256],
  t327 {unproved: too large (1179648 coords)} [N=512 T=1 D=1 H=3 W=3 C=256],
  t328 {proved (structural, for these constants) [sampled 8]} [C=512],
  t329 {unproved: too large (131072 coords)} [N=512 T=1 D=1 H=1 W=1 C=256],
  t330 {proved (structural, for these constants) [sampled 8]} [C=512],
  t331 {unproved: too large (2359296 coords)} [N=512 T=1 D=1 H=3 W=3 C=512],
  t332 {proved (structural, for these constants) [sampled 8]} [C=512],
  t333 {unproved: too large (2359296 coords)} [N=512 T=1 D=1 H=3 W=3 C=512],
  t334 {proved (structural, for these constants) [sampled 8]} [C=512],
  t335 {unproved: too large (2359296 coords)} [N=512 T=1 D=1 H=3 W=3 C=512],
  t336 {proved (structural, for these constants) [sampled 8]} [C=512]]
  nodes:
    n0: [t123 {unproved: too large (150528 coords)}] =
      permute4
        x=t122 {unproved: too large (150528 coords)}
        perm=[H<-W, W<-C, C<-H]
    n174: [t337 {unproved: too large (802816 coords)}] =
      conv2d
        x=t123 {unproved: too large (150528 coords)}
        weight=t297 {proved (structural, for these constants) [sampled 8]}
        bias=t298 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=7; stride=2; pad_before=3; pad_after=3; dilation=1};
               w={kernel=7; stride=2; pad_before=3; pad_after=3; dilation=1};
               in_channels=3}
    n175: [t338 {unproved: too large (802816 coords)}] =
      relu x=t337 {unproved: too large (802816 coords)}
    n176: [t132 {unproved: too large (200704 coords)}] =
      max_pool2d
        x=t338 {unproved: too large (802816 coords)}
        params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
    n177: [t339 {unproved: too large (200704 coords)}] =
      conv2d
        x=t132 {unproved: too large (200704 coords)}
        weight=t299 {proved (structural, for these constants) [sampled 8]}
        bias=t300 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=64}
    n178: [t340 {unproved: too large (200704 coords)}] =
      relu x=t339 {unproved: too large (200704 coords)}
    n179: [t341 {unproved: too large (200704 coords)}] =
      conv2d
        x=t340 {unproved: too large (200704 coords)}
        weight=t301 {proved (structural, for these constants) [sampled 8]}
        bias=t302 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=64}
    n180: [t342 {unproved: too large (200704 coords)}] =
      add
        a=t341 {unproved: too large (200704 coords)}
        b=t132 {unproved: too large (200704 coords)}
    n181: [t343 {unproved: too large (200704 coords)}] =
      relu x=t342 {unproved: too large (200704 coords)}
    n182: [t344 {unproved: too large (200704 coords)}] =
      conv2d
        x=t343 {unproved: too large (200704 coords)}
        weight=t303 {proved (structural, for these constants) [sampled 8]}
        bias=t304 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=64}
    n183: [t345 {unproved: too large (200704 coords)}] =
      relu x=t344 {unproved: too large (200704 coords)}
    n184: [t346 {unproved: too large (200704 coords)}] =
      conv2d
        x=t345 {unproved: too large (200704 coords)}
        weight=t305 {proved (structural, for these constants) [sampled 8]}
        bias=t306 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=64}
    n185: [t347 {unproved: too large (200704 coords)}] =
      add
        a=t346 {unproved: too large (200704 coords)}
        b=t343 {unproved: too large (200704 coords)}
    n186: [t348 {unproved: too large (200704 coords)}] =
      relu x=t347 {unproved: too large (200704 coords)}
    n187: [t349 {unproved: too large (100352 coords)}] =
      conv2d
        x=t348 {unproved: too large (200704 coords)}
        weight=t307 {unproved: too large (73728 coords)}
        bias=t308 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=64}
    n188: [t350 {unproved: too large (100352 coords)}] =
      conv2d
        x=t348 {unproved: too large (200704 coords)}
        weight=t309 {proved (structural, for these constants) [sampled 8]}
        bias=t310 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               in_channels=64}
    n189: [t351 {unproved: too large (100352 coords)}] =
      relu x=t349 {unproved: too large (100352 coords)}
    n190: [t352 {unproved: too large (100352 coords)}] =
      conv2d
        x=t351 {unproved: too large (100352 coords)}
        weight=t311 {unproved: too large (147456 coords)}
        bias=t312 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=128}
    n191: [t353 {unproved: too large (100352 coords)}] =
      add
        a=t352 {unproved: too large (100352 coords)}
        b=t350 {unproved: too large (100352 coords)}
    n192: [t354 {unproved: too large (100352 coords)}] =
      relu x=t353 {unproved: too large (100352 coords)}
    n193: [t355 {unproved: too large (100352 coords)}] =
      conv2d
        x=t354 {unproved: too large (100352 coords)}
        weight=t313 {unproved: too large (147456 coords)}
        bias=t314 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=128}
    n194: [t356 {unproved: too large (100352 coords)}] =
      relu x=t355 {unproved: too large (100352 coords)}
    n195: [t357 {unproved: too large (100352 coords)}] =
      conv2d
        x=t356 {unproved: too large (100352 coords)}
        weight=t315 {unproved: too large (147456 coords)}
        bias=t316 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=128}
    n196: [t358 {unproved: too large (100352 coords)}] =
      add
        a=t357 {unproved: too large (100352 coords)}
        b=t354 {unproved: too large (100352 coords)}
    n197: [t359 {unproved: too large (100352 coords)}] =
      relu x=t358 {unproved: too large (100352 coords)}
    n198: [t360 {proved (structural) [sampled 8]}] =
      conv2d
        x=t359 {unproved: too large (100352 coords)}
        weight=t317 {unproved: too large (294912 coords)}
        bias=t318 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=128}
    n199: [t361 {proved (structural) [sampled 8]}] =
      conv2d
        x=t359 {unproved: too large (100352 coords)}
        weight=t319 {proved (structural, for these constants) [sampled 8]}
        bias=t320 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               in_channels=128}
    n200: [t362 {proved (structural) [sampled 8]}] =
      relu x=t360 {proved (structural) [sampled 8]}
    n201: [t363 {proved (structural) [sampled 8]}] =
      conv2d
        x=t362 {proved (structural) [sampled 8]}
        weight=t321 {unproved: too large (589824 coords)}
        bias=t322 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=256}
    n202: [t364 {proved (structural) [sampled 8]}] =
      add
        a=t363 {proved (structural) [sampled 8]}
        b=t361 {proved (structural) [sampled 8]}
    n203: [t365 {proved (structural) [sampled 8]}] =
      relu x=t364 {proved (structural) [sampled 8]}
    n204: [t366 {proved (structural) [sampled 8]}] =
      conv2d
        x=t365 {proved (structural) [sampled 8]}
        weight=t323 {unproved: too large (589824 coords)}
        bias=t324 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=256}
    n205: [t367 {proved (structural) [sampled 8]}] =
      relu x=t366 {proved (structural) [sampled 8]}
    n206: [t368 {proved (structural) [sampled 8]}] =
      conv2d
        x=t367 {proved (structural) [sampled 8]}
        weight=t325 {unproved: too large (589824 coords)}
        bias=t326 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=256}
    n207: [t369 {proved (structural) [sampled 8]}] =
      add
        a=t368 {proved (structural) [sampled 8]}
        b=t365 {proved (structural) [sampled 8]}
    n208: [t370 {proved (structural) [sampled 8]}] =
      relu x=t369 {proved (structural) [sampled 8]}
    n209: [t371 {proved (structural) [sampled 8]}] =
      conv2d
        x=t370 {proved (structural) [sampled 8]}
        weight=t327 {unproved: too large (1179648 coords)}
        bias=t328 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=256}
    n210: [t372 {proved (structural) [sampled 8]}] =
      conv2d
        x=t370 {proved (structural) [sampled 8]}
        weight=t329 {unproved: too large (131072 coords)}
        bias=t330 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               in_channels=256}
    n211: [t373 {proved (structural) [sampled 8]}] =
      relu x=t371 {proved (structural) [sampled 8]}
    n212: [t374 {proved (structural) [sampled 8]}] =
      conv2d
        x=t373 {proved (structural) [sampled 8]}
        weight=t331 {unproved: too large (2359296 coords)}
        bias=t332 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=512}
    n213: [t375 {proved (structural) [sampled 8]}] =
      add
        a=t374 {proved (structural) [sampled 8]}
        b=t372 {proved (structural) [sampled 8]}
    n214: [t376 {proved (structural) [sampled 8]}] =
      relu x=t375 {proved (structural) [sampled 8]}
    n215: [t377 {proved (structural) [sampled 8]}] =
      conv2d
        x=t376 {proved (structural) [sampled 8]}
        weight=t333 {unproved: too large (2359296 coords)}
        bias=t334 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=512}
    n216: [t378 {proved (structural) [sampled 8]}] =
      relu x=t377 {proved (structural) [sampled 8]}
    n217: [t379 {proved (structural) [sampled 8]}] =
      conv2d
        x=t378 {proved (structural) [sampled 8]}
        weight=t335 {unproved: too large (2359296 coords)}
        bias=t336 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=512}
    n218: [t380 {proved (structural) [sampled 8]}] =
      add
        a=t379 {proved (structural) [sampled 8]}
        b=t376 {proved (structural) [sampled 8]}
    n219: [t381 {proved (structural) [sampled 8]}] =
      relu x=t380 {proved (structural) [sampled 8]}
    n220: [t382 {proved (structural) [sampled 8]}] =
      mean_keepdims
        x=t381 {proved (structural) [sampled 8]}
        params={dims=[W, H]}
    n173: [t296 {tested: agrees (1e-05) [sampled 8]}] =
      conv2d
        x=t382 {proved (structural) [sampled 8]}
        weight=t295 {unproved: too large (512000 coords)}
        bias=t61 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=512}
  outputs: [t296 {tested: agrees (1e-05) [sampled 8]} [C=1000]]

MobileNet-v2. Seventeen DepthwiseConv2D is the point of including it: grouping
is a CONSTRUCTOR in this dialect rather than a parameter, so a depthwise
convolution is a different op from a dense one and the classification has to be
right for the graph to convert at all. Hardtanh is ReLU6.

  $ ../bin/native_graph.exe to4d --pt2 "$PT2_DATA/mobilenet_v2/mobilenet_v2.pt2" --fold --verify-symbolic standard
  canonical native: nodes=101
  native4d: nodes=100 inputs=107
    Add                10
    Conv2D             36
    DepthwiseConv2D    17
    Hardtanh           35
    MeanKeepDims       1
    Permute4           1
  map verification: 207 clusters: 97 proved (for these constants) [sampled 8], 41 proved (structural) [sampled 8], 1 tested (coefficients agree) [sampled 8], 1 unproved (over max_nodes) [sampled 8], 67 unproved (too large)
  graph4
  inputs: [t157 {proved (structural, for these constants) [sampled 8]} [C=1000],
  t314 {unproved: too large (150528 coords)} [H=3 W=224 C=224],
  t728 {unproved: too large (1280000 coords)} [N=1000 T=1 D=1 H=1 W=1 C=1280],
  t730 {proved (structural, for these constants) [sampled 8]} [N=32 T=1 D=1 H=3 W=3 C=3],
  t731 {proved (structural, for these constants) [sampled 8]} [C=32],
  t732 {proved (structural, for these constants) [sampled 8]} [N=32 T=1 D=1 H=3 W=3 C=1],
  t733 {proved (structural, for these constants) [sampled 8]} [C=32],
  t734 {proved (structural, for these constants) [sampled 8]} [N=16 T=1 D=1 H=1 W=1 C=32],
  t735 {proved (structural, for these constants) [sampled 8]} [C=16],
  t736 {proved (structural, for these constants) [sampled 8]} [N=96 T=1 D=1 H=1 W=1 C=16],
  t737 {proved (structural, for these constants) [sampled 8]} [C=96],
  t738 {proved (structural, for these constants) [sampled 8]} [N=96 T=1 D=1 H=3 W=3 C=1],
  t739 {proved (structural, for these constants) [sampled 8]} [C=96],
  t740 {proved (structural, for these constants) [sampled 8]} [N=24 T=1 D=1 H=1 W=1 C=96],
  t741 {proved (structural, for these constants) [sampled 8]} [C=24],
  t742 {proved (structural, for these constants) [sampled 8]} [N=144 T=1 D=1 H=1 W=1 C=24],
  t743 {proved (structural, for these constants) [sampled 8]} [C=144],
  t744 {proved (structural, for these constants) [sampled 8]} [N=144 T=1 D=1 H=3 W=3 C=1],
  t745 {proved (structural, for these constants) [sampled 8]} [C=144],
  t746 {proved (structural, for these constants) [sampled 8]} [N=24 T=1 D=1 H=1 W=1 C=144],
  t747 {proved (structural, for these constants) [sampled 8]} [C=24],
  t748 {proved (structural, for these constants) [sampled 8]} [N=144 T=1 D=1 H=1 W=1 C=24],
  t749 {proved (structural, for these constants) [sampled 8]} [C=144],
  t750 {proved (structural, for these constants) [sampled 8]} [N=144 T=1 D=1 H=3 W=3 C=1],
  t751 {proved (structural, for these constants) [sampled 8]} [C=144],
  t752 {proved (structural, for these constants) [sampled 8]} [N=32 T=1 D=1 H=1 W=1 C=144],
  t753 {proved (structural, for these constants) [sampled 8]} [C=32],
  t754 {proved (structural, for these constants) [sampled 8]} [N=192 T=1 D=1 H=1 W=1 C=32],
  t755 {proved (structural, for these constants) [sampled 8]} [C=192],
  t756 {proved (structural, for these constants) [sampled 8]} [N=192 T=1 D=1 H=3 W=3 C=1],
  t757 {proved (structural, for these constants) [sampled 8]} [C=192],
  t758 {proved (structural, for these constants) [sampled 8]} [N=32 T=1 D=1 H=1 W=1 C=192],
  t759 {proved (structural, for these constants) [sampled 8]} [C=32],
  t760 {proved (structural, for these constants) [sampled 8]} [N=192 T=1 D=1 H=1 W=1 C=32],
  t761 {proved (structural, for these constants) [sampled 8]} [C=192],
  t762 {proved (structural, for these constants) [sampled 8]} [N=192 T=1 D=1 H=3 W=3 C=1],
  t763 {proved (structural, for these constants) [sampled 8]} [C=192],
  t764 {proved (structural, for these constants) [sampled 8]} [N=32 T=1 D=1 H=1 W=1 C=192],
  t765 {proved (structural, for these constants) [sampled 8]} [C=32],
  t766 {proved (structural, for these constants) [sampled 8]} [N=192 T=1 D=1 H=1 W=1 C=32],
  t767 {proved (structural, for these constants) [sampled 8]} [C=192],
  t768 {proved (structural, for these constants) [sampled 8]} [N=192 T=1 D=1 H=3 W=3 C=1],
  t769 {proved (structural, for these constants) [sampled 8]} [C=192],
  t770 {proved (structural, for these constants) [sampled 8]} [N=64 T=1 D=1 H=1 W=1 C=192],
  t771 {proved (structural, for these constants) [sampled 8]} [C=64],
  t772 {proved (structural, for these constants) [sampled 8]} [N=384 T=1 D=1 H=1 W=1 C=64],
  t773 {proved (structural, for these constants) [sampled 8]} [C=384],
  t774 {proved (structural, for these constants) [sampled 8]} [N=384 T=1 D=1 H=3 W=3 C=1],
  t775 {proved (structural, for these constants) [sampled 8]} [C=384],
  t776 {proved (structural, for these constants) [sampled 8]} [N=64 T=1 D=1 H=1 W=1 C=384],
  t777 {proved (structural, for these constants) [sampled 8]} [C=64],
  t778 {proved (structural, for these constants) [sampled 8]} [N=384 T=1 D=1 H=1 W=1 C=64],
  t779 {proved (structural, for these constants) [sampled 8]} [C=384],
  t780 {proved (structural, for these constants) [sampled 8]} [N=384 T=1 D=1 H=3 W=3 C=1],
  t781 {proved (structural, for these constants) [sampled 8]} [C=384],
  t782 {proved (structural, for these constants) [sampled 8]} [N=64 T=1 D=1 H=1 W=1 C=384],
  t783 {proved (structural, for these constants) [sampled 8]} [C=64],
  t784 {proved (structural, for these constants) [sampled 8]} [N=384 T=1 D=1 H=1 W=1 C=64],
  t785 {proved (structural, for these constants) [sampled 8]} [C=384],
  t786 {proved (structural, for these constants) [sampled 8]} [N=384 T=1 D=1 H=3 W=3 C=1],
  t787 {proved (structural, for these constants) [sampled 8]} [C=384],
  t788 {proved (structural, for these constants) [sampled 8]} [N=64 T=1 D=1 H=1 W=1 C=384],
  t789 {proved (structural, for these constants) [sampled 8]} [C=64],
  t790 {proved (structural, for these constants) [sampled 8]} [N=384 T=1 D=1 H=1 W=1 C=64],
  t791 {proved (structural, for these constants) [sampled 8]} [C=384],
  t792 {proved (structural, for these constants) [sampled 8]} [N=384 T=1 D=1 H=3 W=3 C=1],
  t793 {proved (structural, for these constants) [sampled 8]} [C=384],
  t794 {proved (structural, for these constants) [sampled 8]} [N=96 T=1 D=1 H=1 W=1 C=384],
  t795 {proved (structural, for these constants) [sampled 8]} [C=96],
  t796 {proved (structural, for these constants) [sampled 8]} [N=576 T=1 D=1 H=1 W=1 C=96],
  t797 {proved (structural, for these constants) [sampled 8]} [C=576],
  t798 {proved (structural, for these constants) [sampled 8]} [N=576 T=1 D=1 H=3 W=3 C=1],
  t799 {proved (structural, for these constants) [sampled 8]} [C=576],
  t800 {proved (structural, for these constants) [sampled 8]} [N=96 T=1 D=1 H=1 W=1 C=576],
  t801 {proved (structural, for these constants) [sampled 8]} [C=96],
  t802 {proved (structural, for these constants) [sampled 8]} [N=576 T=1 D=1 H=1 W=1 C=96],
  t803 {proved (structural, for these constants) [sampled 8]} [C=576],
  t804 {proved (structural, for these constants) [sampled 8]} [N=576 T=1 D=1 H=3 W=3 C=1],
  t805 {proved (structural, for these constants) [sampled 8]} [C=576],
  t806 {proved (structural, for these constants) [sampled 8]} [N=96 T=1 D=1 H=1 W=1 C=576],
  t807 {proved (structural, for these constants) [sampled 8]} [C=96],
  t808 {proved (structural, for these constants) [sampled 8]} [N=576 T=1 D=1 H=1 W=1 C=96],
  t809 {proved (structural, for these constants) [sampled 8]} [C=576],
  t810 {proved (structural, for these constants) [sampled 8]} [N=576 T=1 D=1 H=3 W=3 C=1],
  t811 {proved (structural, for these constants) [sampled 8]} [C=576],
  t812 {unproved: too large (92160 coords)} [N=160 T=1 D=1 H=1 W=1 C=576],
  t813 {proved (structural, for these constants) [sampled 8]} [C=160],
  t814 {unproved: too large (153600 coords)} [N=960 T=1 D=1 H=1 W=1 C=160],
  t815 {proved (structural, for these constants) [sampled 8]} [C=960],
  t816 {proved (structural, for these constants) [sampled 8]} [N=960 T=1 D=1 H=3 W=3 C=1],
  t817 {proved (structural, for these constants) [sampled 8]} [C=960],
  t818 {unproved: too large (153600 coords)} [N=160 T=1 D=1 H=1 W=1 C=960],
  t819 {proved (structural, for these constants) [sampled 8]} [C=160],
  t820 {unproved: too large (153600 coords)} [N=960 T=1 D=1 H=1 W=1 C=160],
  t821 {proved (structural, for these constants) [sampled 8]} [C=960],
  t822 {proved (structural, for these constants) [sampled 8]} [N=960 T=1 D=1 H=3 W=3 C=1],
  t823 {proved (structural, for these constants) [sampled 8]} [C=960],
  t824 {unproved: too large (153600 coords)} [N=160 T=1 D=1 H=1 W=1 C=960],
  t825 {proved (structural, for these constants) [sampled 8]} [C=160],
  t826 {unproved: too large (153600 coords)} [N=960 T=1 D=1 H=1 W=1 C=160],
  t827 {proved (structural, for these constants) [sampled 8]} [C=960],
  t828 {proved (structural, for these constants) [sampled 8]} [N=960 T=1 D=1 H=3 W=3 C=1],
  t829 {proved (structural, for these constants) [sampled 8]} [C=960],
  t830 {unproved: too large (307200 coords)} [N=320 T=1 D=1 H=1 W=1 C=960],
  t831 {proved (structural, for these constants) [sampled 8]} [C=320],
  t832 {unproved: too large (409600 coords)} [N=1280 T=1 D=1 H=1 W=1 C=320],
  t833 {proved (structural, for these constants) [sampled 8]} [C=1280]]
  nodes:
    n0: [t315 {unproved: too large (150528 coords)}] =
      permute4
        x=t314 {unproved: too large (150528 coords)}
        perm=[H<-W, W<-C, C<-H]
    n415: [t834 {unproved: too large (401408 coords)}] =
      conv2d
        x=t315 {unproved: too large (150528 coords)}
        weight=t730 {proved (structural, for these constants) [sampled 8]}
        bias=t731 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=3}
    n416: [t835 {unproved: too large (401408 coords)}] =
      hardtanh
        x=t834 {unproved: too large (401408 coords)}
        params={min_val=0; max_val=6}
    n417: [t836 {unproved: too large (401408 coords)}] =
      depthwise_conv2d
        x=t835 {unproved: too large (401408 coords)}
        weight=t732 {proved (structural, for these constants) [sampled 8]}
        bias=t733 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=32}
    n418: [t837 {unproved: too large (401408 coords)}] =
      hardtanh
        x=t836 {unproved: too large (401408 coords)}
        params={min_val=0; max_val=6}
    n419: [t838 {unproved: too large (200704 coords)}] =
      conv2d
        x=t837 {unproved: too large (401408 coords)}
        weight=t734 {proved (structural, for these constants) [sampled 8]}
        bias=t735 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32}
    n420: [t839 {unproved: too large (1204224 coords)}] =
      conv2d
        x=t838 {unproved: too large (200704 coords)}
        weight=t736 {proved (structural, for these constants) [sampled 8]}
        bias=t737 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16}
    n421: [t840 {unproved: too large (1204224 coords)}] =
      hardtanh
        x=t839 {unproved: too large (1204224 coords)}
        params={min_val=0; max_val=6}
    n422: [t841 {unproved: too large (301056 coords)}] =
      depthwise_conv2d
        x=t840 {unproved: too large (1204224 coords)}
        weight=t738 {proved (structural, for these constants) [sampled 8]}
        bias=t739 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=96}
    n423: [t842 {unproved: too large (301056 coords)}] =
      hardtanh
        x=t841 {unproved: too large (301056 coords)}
        params={min_val=0; max_val=6}
    n424: [t843 {unproved: too large (75264 coords)}] =
      conv2d
        x=t842 {unproved: too large (301056 coords)}
        weight=t740 {proved (structural, for these constants) [sampled 8]}
        bias=t741 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96}
    n425: [t844 {unproved: too large (451584 coords)}] =
      conv2d
        x=t843 {unproved: too large (75264 coords)}
        weight=t742 {proved (structural, for these constants) [sampled 8]}
        bias=t743 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=24}
    n426: [t845 {unproved: too large (451584 coords)}] =
      hardtanh
        x=t844 {unproved: too large (451584 coords)}
        params={min_val=0; max_val=6}
    n427: [t846 {unproved: too large (451584 coords)}] =
      depthwise_conv2d
        x=t845 {unproved: too large (451584 coords)}
        weight=t744 {proved (structural, for these constants) [sampled 8]}
        bias=t745 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=144}
    n428: [t847 {unproved: too large (451584 coords)}] =
      hardtanh
        x=t846 {unproved: too large (451584 coords)}
        params={min_val=0; max_val=6}
    n429: [t848 {unproved: too large (75264 coords)}] =
      conv2d
        x=t847 {unproved: too large (451584 coords)}
        weight=t746 {proved (structural, for these constants) [sampled 8]}
        bias=t747 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=144}
    n430: [t849 {unproved: too large (75264 coords)}] =
      add
        a=t843 {unproved: too large (75264 coords)}
        b=t848 {unproved: too large (75264 coords)}
    n431: [t850 {unproved: too large (451584 coords)}] =
      conv2d
        x=t849 {unproved: too large (75264 coords)}
        weight=t748 {proved (structural, for these constants) [sampled 8]}
        bias=t749 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=24}
    n432: [t851 {unproved: too large (451584 coords)}] =
      hardtanh
        x=t850 {unproved: too large (451584 coords)}
        params={min_val=0; max_val=6}
    n433: [t852 {unproved: too large (112896 coords)}] =
      depthwise_conv2d
        x=t851 {unproved: too large (451584 coords)}
        weight=t750 {proved (structural, for these constants) [sampled 8]}
        bias=t751 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=144}
    n434: [t853 {unproved: too large (112896 coords)}] =
      hardtanh
        x=t852 {unproved: too large (112896 coords)}
        params={min_val=0; max_val=6}
    n435: [t854 {proved (structural) [sampled 8]}] =
      conv2d
        x=t853 {unproved: too large (112896 coords)}
        weight=t752 {proved (structural, for these constants) [sampled 8]}
        bias=t753 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=144}
    n436: [t855 {unproved: too large (150528 coords)}] =
      conv2d
        x=t854 {proved (structural) [sampled 8]}
        weight=t754 {proved (structural, for these constants) [sampled 8]}
        bias=t755 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32}
    n437: [t856 {unproved: too large (150528 coords)}] =
      hardtanh
        x=t855 {unproved: too large (150528 coords)}
        params={min_val=0; max_val=6}
    n438: [t857 {unproved: too large (150528 coords)}] =
      depthwise_conv2d
        x=t856 {unproved: too large (150528 coords)}
        weight=t756 {proved (structural, for these constants) [sampled 8]}
        bias=t757 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=192}
    n439: [t858 {unproved: too large (150528 coords)}] =
      hardtanh
        x=t857 {unproved: too large (150528 coords)}
        params={min_val=0; max_val=6}
    n440: [t859 {proved (structural) [sampled 8]}] =
      conv2d
        x=t858 {unproved: too large (150528 coords)}
        weight=t758 {proved (structural, for these constants) [sampled 8]}
        bias=t759 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=192}
    n441: [t860 {proved (structural) [sampled 8]}] =
      add
        a=t854 {proved (structural) [sampled 8]}
        b=t859 {proved (structural) [sampled 8]}
    n442: [t861 {unproved: too large (150528 coords)}] =
      conv2d
        x=t860 {proved (structural) [sampled 8]}
        weight=t760 {proved (structural, for these constants) [sampled 8]}
        bias=t761 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32}
    n443: [t862 {unproved: too large (150528 coords)}] =
      hardtanh
        x=t861 {unproved: too large (150528 coords)}
        params={min_val=0; max_val=6}
    n444: [t863 {unproved: too large (150528 coords)}] =
      depthwise_conv2d
        x=t862 {unproved: too large (150528 coords)}
        weight=t762 {proved (structural, for these constants) [sampled 8]}
        bias=t763 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=192}
    n445: [t864 {unproved: too large (150528 coords)}] =
      hardtanh
        x=t863 {unproved: too large (150528 coords)}
        params={min_val=0; max_val=6}
    n446: [t865 {proved (structural) [sampled 8]}] =
      conv2d
        x=t864 {unproved: too large (150528 coords)}
        weight=t764 {proved (structural, for these constants) [sampled 8]}
        bias=t765 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=192}
    n447: [t866 {proved (structural) [sampled 8]}] =
      add
        a=t860 {proved (structural) [sampled 8]}
        b=t865 {proved (structural) [sampled 8]}
    n448: [t867 {unproved: too large (150528 coords)}] =
      conv2d
        x=t866 {proved (structural) [sampled 8]}
        weight=t766 {proved (structural, for these constants) [sampled 8]}
        bias=t767 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32}
    n449: [t868 {unproved: too large (150528 coords)}] =
      hardtanh
        x=t867 {unproved: too large (150528 coords)}
        params={min_val=0; max_val=6}
    n450: [t869 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t868 {unproved: too large (150528 coords)}
        weight=t768 {proved (structural, for these constants) [sampled 8]}
        bias=t769 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=192}
    n451: [t870 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t869 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n452: [t871 {proved (structural) [sampled 8]}] =
      conv2d
        x=t870 {proved (structural) [sampled 8]}
        weight=t770 {proved (structural, for these constants) [sampled 8]}
        bias=t771 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=192}
    n453: [t872 {unproved: too large (75264 coords)}] =
      conv2d
        x=t871 {proved (structural) [sampled 8]}
        weight=t772 {proved (structural, for these constants) [sampled 8]}
        bias=t773 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=64}
    n454: [t873 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t872 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n455: [t874 {unproved: too large (75264 coords)}] =
      depthwise_conv2d
        x=t873 {unproved: too large (75264 coords)}
        weight=t774 {proved (structural, for these constants) [sampled 8]}
        bias=t775 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=384}
    n456: [t875 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t874 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n457: [t876 {proved (structural) [sampled 8]}] =
      conv2d
        x=t875 {unproved: too large (75264 coords)}
        weight=t776 {proved (structural, for these constants) [sampled 8]}
        bias=t777 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=384}
    n458: [t877 {proved (structural) [sampled 8]}] =
      add
        a=t871 {proved (structural) [sampled 8]}
        b=t876 {proved (structural) [sampled 8]}
    n459: [t878 {unproved: too large (75264 coords)}] =
      conv2d
        x=t877 {proved (structural) [sampled 8]}
        weight=t778 {proved (structural, for these constants) [sampled 8]}
        bias=t779 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=64}
    n460: [t879 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t878 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n461: [t880 {unproved: too large (75264 coords)}] =
      depthwise_conv2d
        x=t879 {unproved: too large (75264 coords)}
        weight=t780 {proved (structural, for these constants) [sampled 8]}
        bias=t781 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=384}
    n462: [t881 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t880 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n463: [t882 {proved (structural) [sampled 8]}] =
      conv2d
        x=t881 {unproved: too large (75264 coords)}
        weight=t782 {proved (structural, for these constants) [sampled 8]}
        bias=t783 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=384}
    n464: [t883 {proved (structural) [sampled 8]}] =
      add
        a=t877 {proved (structural) [sampled 8]}
        b=t882 {proved (structural) [sampled 8]}
    n465: [t884 {unproved: too large (75264 coords)}] =
      conv2d
        x=t883 {proved (structural) [sampled 8]}
        weight=t784 {proved (structural, for these constants) [sampled 8]}
        bias=t785 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=64}
    n466: [t885 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t884 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n467: [t886 {unproved: too large (75264 coords)}] =
      depthwise_conv2d
        x=t885 {unproved: too large (75264 coords)}
        weight=t786 {proved (structural, for these constants) [sampled 8]}
        bias=t787 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=384}
    n468: [t887 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t886 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n469: [t888 {proved (structural) [sampled 8]}] =
      conv2d
        x=t887 {unproved: too large (75264 coords)}
        weight=t788 {proved (structural, for these constants) [sampled 8]}
        bias=t789 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=384}
    n470: [t889 {proved (structural) [sampled 8]}] =
      add
        a=t883 {proved (structural) [sampled 8]}
        b=t888 {proved (structural) [sampled 8]}
    n471: [t890 {unproved: too large (75264 coords)}] =
      conv2d
        x=t889 {proved (structural) [sampled 8]}
        weight=t790 {proved (structural, for these constants) [sampled 8]}
        bias=t791 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=64}
    n472: [t891 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t890 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n473: [t892 {unproved: too large (75264 coords)}] =
      depthwise_conv2d
        x=t891 {unproved: too large (75264 coords)}
        weight=t792 {proved (structural, for these constants) [sampled 8]}
        bias=t793 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=384}
    n474: [t893 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t892 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n475: [t894 {proved (structural) [sampled 8]}] =
      conv2d
        x=t893 {unproved: too large (75264 coords)}
        weight=t794 {proved (structural, for these constants) [sampled 8]}
        bias=t795 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=384}
    n476: [t895 {unproved: too large (112896 coords)}] =
      conv2d
        x=t894 {proved (structural) [sampled 8]}
        weight=t796 {proved (structural, for these constants) [sampled 8]}
        bias=t797 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96}
    n477: [t896 {unproved: too large (112896 coords)}] =
      hardtanh
        x=t895 {unproved: too large (112896 coords)}
        params={min_val=0; max_val=6}
    n478: [t897 {unproved: too large (112896 coords)}] =
      depthwise_conv2d
        x=t896 {unproved: too large (112896 coords)}
        weight=t798 {proved (structural, for these constants) [sampled 8]}
        bias=t799 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=576}
    n479: [t898 {unproved: too large (112896 coords)}] =
      hardtanh
        x=t897 {unproved: too large (112896 coords)}
        params={min_val=0; max_val=6}
    n480: [t899 {proved (structural) [sampled 8]}] =
      conv2d
        x=t898 {unproved: too large (112896 coords)}
        weight=t800 {proved (structural, for these constants) [sampled 8]}
        bias=t801 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=576}
    n481: [t900 {proved (structural) [sampled 8]}] =
      add
        a=t894 {proved (structural) [sampled 8]}
        b=t899 {proved (structural) [sampled 8]}
    n482: [t901 {unproved: too large (112896 coords)}] =
      conv2d
        x=t900 {proved (structural) [sampled 8]}
        weight=t802 {proved (structural, for these constants) [sampled 8]}
        bias=t803 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96}
    n483: [t902 {unproved: too large (112896 coords)}] =
      hardtanh
        x=t901 {unproved: too large (112896 coords)}
        params={min_val=0; max_val=6}
    n484: [t903 {unproved: too large (112896 coords)}] =
      depthwise_conv2d
        x=t902 {unproved: too large (112896 coords)}
        weight=t804 {proved (structural, for these constants) [sampled 8]}
        bias=t805 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=576}
    n485: [t904 {unproved: too large (112896 coords)}] =
      hardtanh
        x=t903 {unproved: too large (112896 coords)}
        params={min_val=0; max_val=6}
    n486: [t905 {proved (structural) [sampled 8]}] =
      conv2d
        x=t904 {unproved: too large (112896 coords)}
        weight=t806 {proved (structural, for these constants) [sampled 8]}
        bias=t807 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=576}
    n487: [t906 {proved (structural) [sampled 8]}] =
      add
        a=t900 {proved (structural) [sampled 8]}
        b=t905 {proved (structural) [sampled 8]}
    n488: [t907 {unproved: too large (112896 coords)}] =
      conv2d
        x=t906 {proved (structural) [sampled 8]}
        weight=t808 {proved (structural, for these constants) [sampled 8]}
        bias=t809 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96}
    n489: [t908 {unproved: too large (112896 coords)}] =
      hardtanh
        x=t907 {unproved: too large (112896 coords)}
        params={min_val=0; max_val=6}
    n490: [t909 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t908 {unproved: too large (112896 coords)}
        weight=t810 {proved (structural, for these constants) [sampled 8]}
        bias=t811 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=576}
    n491: [t910 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t909 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n492: [t911 {proved (structural) [sampled 8]}] =
      conv2d
        x=t910 {proved (structural) [sampled 8]}
        weight=t812 {unproved: too large (92160 coords)}
        bias=t813 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=576}
    n493: [t912 {proved (structural) [sampled 8]}] =
      conv2d
        x=t911 {proved (structural) [sampled 8]}
        weight=t814 {unproved: too large (153600 coords)}
        bias=t815 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=160}
    n494: [t913 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t912 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n495: [t914 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t913 {proved (structural) [sampled 8]}
        weight=t816 {proved (structural, for these constants) [sampled 8]}
        bias=t817 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=960}
    n496: [t915 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t914 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n497: [t916 {proved (structural) [sampled 8]}] =
      conv2d
        x=t915 {proved (structural) [sampled 8]}
        weight=t818 {unproved: too large (153600 coords)}
        bias=t819 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=960}
    n498: [t917 {proved (structural) [sampled 8]}] =
      add
        a=t911 {proved (structural) [sampled 8]}
        b=t916 {proved (structural) [sampled 8]}
    n499: [t918 {proved (structural) [sampled 8]}] =
      conv2d
        x=t917 {proved (structural) [sampled 8]}
        weight=t820 {unproved: too large (153600 coords)}
        bias=t821 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=160}
    n500: [t919 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t918 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n501: [t920 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t919 {proved (structural) [sampled 8]}
        weight=t822 {proved (structural, for these constants) [sampled 8]}
        bias=t823 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=960}
    n502: [t921 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t920 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n503: [t922 {proved (structural) [sampled 8]}] =
      conv2d
        x=t921 {proved (structural) [sampled 8]}
        weight=t824 {unproved: too large (153600 coords)}
        bias=t825 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=960}
    n504: [t923 {proved (structural) [sampled 8]}] =
      add
        a=t917 {proved (structural) [sampled 8]}
        b=t922 {proved (structural) [sampled 8]}
    n505: [t924 {proved (structural) [sampled 8]}] =
      conv2d
        x=t923 {proved (structural) [sampled 8]}
        weight=t826 {unproved: too large (153600 coords)}
        bias=t827 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=160}
    n506: [t925 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t924 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n507: [t926 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t925 {proved (structural) [sampled 8]}
        weight=t828 {proved (structural, for these constants) [sampled 8]}
        bias=t829 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=960}
    n508: [t927 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t926 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n509: [t928 {proved (structural) [sampled 8]}] =
      conv2d
        x=t927 {proved (structural) [sampled 8]}
        weight=t830 {unproved: too large (307200 coords)}
        bias=t831 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=960}
    n510: [t929 {proved (structural) [sampled 8]}] =
      conv2d
        x=t928 {proved (structural) [sampled 8]}
        weight=t832 {unproved: too large (409600 coords)}
        bias=t833 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=320}
    n511: [t930 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t929 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n512: [t931 {unproved: over max_nodes (52767) [sampled 8]}] =
      mean_keepdims
        x=t930 {proved (structural) [sampled 8]}
        params={dims=[W, H]}
    n414: [t729 {tested: agrees (1e-05) [sampled 8]}] =
      conv2d
        x=t931 {unproved: over max_nodes (52767) [sampled 8]}
        weight=t728 {unproved: too large (1280000 coords)}
        bias=t157 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=1280}
  outputs: [t729 {tested: agrees (1e-05) [sampled 8]} [C=1000]]

A refuted cluster anywhere would be the failure worth catching, so it is
asserted directly rather than left for a reader to notice its absence in the
dumps above.

This is not the same claim as the numeric check `transform --verify` makes. That
one runs a model twice on one real input and compares outputs; this one checks
every corresponding edge without payloads for the graph inputs, so it speaks
about every input within its budget. Neither replaces the other.

  $ for m in resnet18 mobilenet_v2; do
  >   n=$(../bin/native_graph.exe to4d --pt2 "$PT2_DATA/$m/$m.pt2" --fold \
  >       --verify-symbolic standard | grep -c refuted || true)
  >   echo "$m: $n refuted"
  > done
  resnet18: 0 refuted
  mobilenet_v2: 0 refuted
