(* The JavaScript [Loop_js] emits for pointwise, layout, pooling and
   reduction ops, one golden per op at its walk's initial config, so the
   generated text is readable in the tree. These pin the text only: the JS's
   agreement with the reference is checked by the loop_js pt2 runs under node.
   Print one from the toplevel with [Loop_js_walk.print]. *)

let%expect_test "adaptive_avg_pool2d" =
  Loop_js_walk.print "adaptive_avg_pool2d";
  [%expect
    {|
    // adaptive_avg_pool2d {shape=[1,4,8,8] output_size=[4,4]}
    "use strict";
    function loop_kernel(b0, b1) {
      let x0 = 0;
      let x1 = 0;
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 4; i3++) {
              for (let i4 = 0; i4 < 4; i4++) {
                for (let i5 = 0; i5 < 4; i5++) {
                  x0 = 0;
                  for (let i6 = Math.floor(8 * i3 / 4); i6 < Math.ceil(8 * (i3 + 1) / 4); i6++) {
                    x1 = 0;
                    for (let i7 = Math.floor(8 * i4 / 4); i7 < Math.ceil(8 * (i4 + 1) / 4); i7++) {
                      x1 = x1 + b0[4 * (8 * (8 * (i0 + i1 + i2) + i6) + i7) + i5];
                    }
                    x0 = x0 + x1;
                  }
                  b1[4 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] = Math.fround(x0 / ((Math.ceil(8 * (i3 + 1) / 4) + -1 * Math.floor(8 * i3 / 4) + 0) * (Math.ceil(8 * (i4 + 1) / 4) + -1 * Math.floor(8 * i4 / 4) + 0)));
                }
              }
            }
          }
        }
      }
      return null;
    } |}]

let%expect_test "add" =
  Loop_js_walk.print "add";
  [%expect
    {|
    // add [n=1 c=3 h=4 w=4]
    "use strict";
    function loop_kernel(b0, b1, b2) {
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 4; i3++) {
              for (let i4 = 0; i4 < 4; i4++) {
                for (let i5 = 0; i5 < 3; i5++) {
                  b2[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] = Math.fround(b0[3 * (4 * i3 + i4) + i5] + b1[3 * (4 * i3 + i4) + i5]);
                }
              }
            }
          }
        }
      }
      return null;
    } |}]

let%expect_test "avg_pool2d" =
  Loop_js_walk.print "avg_pool2d";
  [%expect
    {|
    // avg_pool2d {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=false count_include_pad=true
    "use strict";
    function loop_kernel(b0, b1) {
      let x0 = 0;
      let x1 = 0;
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 4; i3++) {
              for (let i4 = 0; i4 < 4; i4++) {
                for (let i5 = 0; i5 < 4; i5++) {
                  x0 = 0;
                  for (let i6 = Math.max(0, -1 * (2 * i3)); i6 < Math.min(2, 8 + -1 + -1 * (2 * i3) + 1); i6++) {
                    x1 = 0;
                    for (let i7 = Math.max(0, -1 * (2 * i4)); i7 < Math.min(2, 8 + -1 + -1 * (2 * i4) + 1); i7++) {
                      x1 = x1 + b0[4 * (8 * (8 * (i0 + i1 + i2) + (2 * i3 + i6)) + (2 * i4 + i7)) + i5];
                    }
                    x0 = x0 + x1;
                  }
                  b1[4 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] = Math.fround(x0 / 4);
                }
              }
            }
          }
        }
      }
      return null;
    } |}]

let%expect_test "gelu" =
  Loop_js_walk.print "gelu";
  [%expect
    {|
    // gelu [n=1 c=3 h=4 w=4] none
    "use strict";
    function erf(x) {
      const p = 0.32759110000000002;
      const a1 = 0.25482959199999999;
      const a2 = -0.28449673599999997;
      const a3 = 1.4214137410000001;
      const a4 = -1.453152027;
      const a5 = 1.0614054289999999;
      const sign = x < 0 ? -1 : 1;
      const ax = Math.abs(x);
      const t = 1 / (1 + p * ax);
      const poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
      return sign * (1 - poly * Math.exp(-ax * ax));
    }
    function loop_kernel(b0, b1) {
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 4; i3++) {
              for (let i4 = 0; i4 < 4; i4++) {
                for (let i5 = 0; i5 < 3; i5++) {
                  b1[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] = Math.fround(0.5 * b0[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] * (1 + erf(b0[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] / Math.sqrt(2))));
                }
              }
            }
          }
        }
      }
      return null;
    } |}]

let%expect_test "max_pool2d" =
  Loop_js_walk.print "max_pool2d";
  [%expect
    {|
    // max_pool2d {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=false
    "use strict";
    function pool_better(best, value) {
      return value > best || value !== value;
    }
    function loop_kernel(b0, b1) {
      let x0 = 0;
      let x2 = 0;
      let x1 = 0;
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 4; i3++) {
              for (let i4 = 0; i4 < 4; i4++) {
                for (let i5 = 0; i5 < 4; i5++) {
                  x0 = -Infinity;
                  x1 = 0;
                  for (let i6 = Math.max(0, 2 * i3); i6 < Math.min(8, 2 * i3 + 2); i6++) {
                    for (let i7 = Math.max(0, 2 * i4); i7 < Math.min(8, 2 * i4 + 2); i7++) {
                      x2 = b0[4 * (8 * (8 * (i0 + i1 + i2) + i6) + i7) + i5];
                      if (pool_better(x0, x2)) {
                        x0 = x2;
                        x1 = 8 * i6 + i7;
                      }
                    }
                  }
                  b1[4 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] = Math.fround(x0);
                }
              }
            }
          }
        }
      }
      return null;
    } |}]

let%expect_test "mean" =
  Loop_js_walk.print "mean";
  [%expect
    {|
    // mean {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    "use strict";
    function loop_kernel(b0, b1) {
      let x0 = 0;
      let x1 = 0;
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 2; i2++) {
            for (let i3 = 0; i3 < 1; i3++) {
              for (let i4 = 0; i4 < 1; i4++) {
                for (let i5 = 0; i5 < 4; i5++) {
                  x0 = 0;
                  for (let i6 = 0; i6 < 4; i6++) {
                    x1 = 0;
                    for (let i7 = 0; i7 < 4; i7++) {
                      x1 = x1 + b0[4 * (4 * (4 * (i2 + i3 + i4) + i6) + i7) + i5];
                    }
                    x0 = x0 + x1;
                  }
                  b1[4 * (2 * (i0 + i1) + i2 + i3 + i4) + i5] = Math.fround(x0 / 16);
                }
              }
            }
          }
        }
      }
      return null;
    } |}]

let%expect_test "mul" =
  Loop_js_walk.print "mul";
  [%expect
    {|
    // mul [n=1 c=3 h=4 w=4]
    "use strict";
    function loop_kernel(b0, b1, b2) {
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 4; i3++) {
              for (let i4 = 0; i4 < 4; i4++) {
                for (let i5 = 0; i5 < 3; i5++) {
                  b2[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] = Math.fround(b0[3 * (4 * i3 + i4) + i5] * b1[3 * (4 * i3 + i4) + i5]);
                }
              }
            }
          }
        }
      }
      return null;
    } |}]

let%expect_test "permute" =
  Loop_js_walk.print "permute";
  [%expect
    {|
    // permute {shape=[n=1 c=4 h=4 w=4] perm=[H<-W, W<-H]}
    "use strict";
    function loop_kernel(b0, b1) {
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 4; i3++) {
              for (let i4 = 0; i4 < 4; i4++) {
                for (let i5 = 0; i5 < 4; i5++) {
                  b1[4 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] = Math.fround(b0[4 * (4 * (4 * (i0 + i1 + i2) + i4) + i3) + i5]);
                }
              }
            }
          }
        }
      }
      return null;
    } |}]

let%expect_test "relu" =
  Loop_js_walk.print "relu";
  [%expect
    {|
    // relu [n=1 c=3 h=4 w=4]
    "use strict";
    function loop_kernel(b0, b1) {
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 4; i3++) {
              for (let i4 = 0; i4 < 4; i4++) {
                for (let i5 = 0; i5 < 3; i5++) {
                  b1[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] = Math.fround(b0[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] < 0 ? 0 : b0[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5]);
                }
              }
            }
          }
        }
      }
      return null;
    } |}]

let%expect_test "reshape" =
  Loop_js_walk.print "reshape";
  [%expect
    {|
    // reshape {shape=[n=1 c=4 h=4 w=4] -> flat}
    "use strict";
    function coord_failure(buffer, extents, coord) {
      for (let axis = 0; axis < 6; axis++) {
        if (coord[axis] < 0 || coord[axis] >= extents[axis]) {
          return { kind: "coord_out_of_range", buffer: buffer, axis: axis, index: coord[axis], coord: coord };
        }
      }
      return { kind: "defect" };
    }
    function loop_kernel(b0, b1) {
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 1; i3++) {
              for (let i4 = 0; i4 < 1; i4++) {
                for (let i5 = 0; i5 < 64; i5++) {
                  if (Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4) + -4 * Math.floor(Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4) / 4) < 0 || Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4) + -4 * Math.floor(Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4) / 4) >= 4 || (64 * (i0 + i1 + i2 + i3 + i4) + i5 + -4 * Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4) < 0 || 64 * (i0 + i1 + i2 + i3 + i4) + i5 + -4 * Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4) >= 4)) {
                    return coord_failure(0, [1, 1, 1, 4, 4, 4], [Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64) + -1 * Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64), Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64) + -1 * Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64), Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64) + -1 * Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64), Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 16) + -4 * Math.floor(Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 16) / 4), Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4) + -4 * Math.floor(Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4) / 4), 64 * (i0 + i1 + i2 + i3 + i4) + i5 + -4 * Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4)]);
                  }
                  b1[64 * (i0 + i1 + i2 + i3 + i4) + i5] = Math.fround(b0[4 * (4 * (4 * (Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64) + -1 * Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64) + (Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64) + -1 * Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64)) + (Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64) + -1 * Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 64))) + (Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 16) + -4 * Math.floor(Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 16) / 4))) + (Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4) + -4 * Math.floor(Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4) / 4))) + (64 * (i0 + i1 + i2 + i3 + i4) + i5 + -4 * Math.floor((64 * (i0 + i1 + i2 + i3 + i4) + i5) / 4))]);
                }
              }
            }
          }
        }
      }
      return null;
    } |}]

let%expect_test "sigmoid" =
  Loop_js_walk.print "sigmoid";
  [%expect
    {|
    // sigmoid [n=1 c=3 h=4 w=4]
    "use strict";
    function loop_kernel(b0, b1) {
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 4; i3++) {
              for (let i4 = 0; i4 < 4; i4++) {
                for (let i5 = 0; i5 < 3; i5++) {
                  b1[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] = Math.fround(1 / (1 + Math.exp(0 - b0[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5])));
                }
              }
            }
          }
        }
      }
      return null;
    } |}]

let%expect_test "silu" =
  Loop_js_walk.print "silu";
  [%expect
    {|
    // silu [n=1 c=3 h=4 w=4]
    "use strict";
    function loop_kernel(b0, b1) {
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 4; i3++) {
              for (let i4 = 0; i4 < 4; i4++) {
                for (let i5 = 0; i5 < 3; i5++) {
                  b1[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] = Math.fround(b0[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5] / (1 + Math.exp(0 - b0[3 * (4 * (4 * (i0 + i1 + i2) + i3) + i4) + i5])));
                }
              }
            }
          }
        }
      }
      return null;
    } |}]
