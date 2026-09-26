(* The JavaScript [Loop_js] emits for matmul, convolution, attention and
   normalization ops, one golden per op at its walk's initial config, so the
   generated text is readable in the tree. These pin the text only: the JS's
   agreement with the reference is checked by the loop_js pt2 runs under node.
   Print one from the toplevel with [Loop_js_walk.print]. *)

let%expect_test "batch_norm" =
  Loop_js_walk.print "batch_norm";
  [%expect
    {|
    // batch_norm {shape=[n=2 c=4 h=4 w=4] eps=1e-05}
    "use strict";
    function loop_kernel(b0, b1, b2, b3, b4, b5) {
      for (let i0 = 0; i0 < 32; i0++) {
        for (let i1 = 0; i1 < 4; i1++) {
          b5[4 * i0 + i1] = (b0[4 * i0 + i1] - b3[i1]) * (1 / Math.sqrt(b4[i1] + 1.0000000000000001e-05)) * b1[i1] + b2[i1];
        }
      }
      return null;
    } |}]

let%expect_test "bmm" =
  Loop_js_walk.print "bmm";
  [%expect
    {|
    // bmm {batch=1 n=2 m=3 p=4}
    "use strict";
    function loop_kernel(b0, b1, b2) {
      let x0 = 0;
      for (let i0 = 0; i0 < 2; i0++) {
        for (let i1 = 0; i1 < 4; i1++) {
          x0 = 0;
          for (let i2 = 0; i2 < 3; i2++) {
            x0 = x0 + b0[3 * i0 + i2] * b1[4 * i2 + i1];
          }
          b2[4 * i0 + i1] = x0;
        }
      }
      return null;
    } |}]

let%expect_test "conv2d" =
  Loop_js_walk.print "conv2d";
  [%expect
    {|
    // conv2d {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    "use strict";
    function loop_kernel(b0, b1, b2, b3) {
      let x0 = 0;
      let x1 = 0;
      let x2 = 0;
      for (let i0 = 0; i0 < 8; i0++) {
        for (let i1 = 0; i1 < 8; i1++) {
          for (let i2 = 0; i2 < 8; i2++) {
            x0 = 0;
            for (let i3 = 0; i3 < 4; i3++) {
              x1 = 0;
              const n4 = Math.min(3, 9 - i0);
              for (let i4 = Math.max(0, -(i0 - 1)); i4 < n4; i4++) {
                x2 = 0;
                const n5 = Math.min(3, 9 - i1);
                for (let i5 = Math.max(0, -(i1 - 1)); i5 < n5; i5++) {
                  x2 = x2 + b0[4 * (8 * (i0 - 1 + i4) + (i1 - 1 + i5)) + i3] * b1[4 * (3 * (3 * i2 + i4) + i5) + i3];
                }
                x1 = x1 + x2;
              }
              x0 = x0 + x1;
            }
            b3[8 * (8 * i0 + i1) + i2] = x0 + b2[i2];
          }
        }
      }
      return null;
    } |}]

let%expect_test "layer_norm" =
  Loop_js_walk.print "layer_norm";
  [%expect
    {|
    // layer_norm {shape=[n=1 c=5 h=4 w=3] k=1 eps=1e-05 weight=true bias=true}
    "use strict";
    function loop_kernel(b0, b1, b2, b3) {
      let x0 = 0;
      let x1 = 0;
      let x2 = 0;
      const a0 = new Float64Array(4);
      for (let i0 = 0; i0 < 12; i0++) {
        x0 = 0;
        for (let i1 = 0; i1 < 5; i1++) {
          x0 = x0 + b0[5 * i0 + i1];
        }
        a0[0] = x0;
        a0[1] = a0[0] / 5;
        x1 = 0;
        for (let i2 = 0; i2 < 5; i2++) {
          x2 = b0[5 * i0 + i2];
          x1 = x1 + (x2 - a0[1]) * (x2 - a0[1]);
        }
        a0[2] = x1;
        a0[3] = 1 / Math.sqrt(a0[2] / 5 + 1.0000000000000001e-05);
        for (let i3 = 0; i3 < 5; i3++) {
          b3[5 * i0 + i3] = (b0[5 * i0 + i3] - a0[1]) * a0[3] * b1[i3] + b2[i3];
        }
      }
      return null;
    } |}]

let%expect_test "linear" =
  Loop_js_walk.print "linear";
  [%expect
    {|
    // linear {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    "use strict";
    function loop_kernel(b0, b1, b2, b3) {
      let x0 = 0;
      for (let i0 = 0; i0 < 4; i0++) {
        for (let i1 = 0; i1 < 6; i1++) {
          x0 = 0;
          for (let i2 = 0; i2 < 8; i2++) {
            x0 = x0 + b0[8 * i0 + i2] * b1[8 * i1 + i2];
          }
          b3[6 * i0 + i1] = x0 + b2[i1];
        }
      }
      return null;
    } |}]

let%expect_test "sdpa" =
  Loop_js_walk.print "sdpa";
  [%expect
    {|
    // sdpa {batch=1 heads=2 wq=3 wk=4 e=5 mask=present scale=default}
    "use strict";
    function loop_kernel(b0, b1, b2, b3, b4) {
      let x0 = 0;
      let x1 = 0;
      let x2 = 0;
      let x3 = 0;
      let x4 = 0;
      const a0 = new Float64Array(11);
      a0[0] = Math.sqrt(1 / Math.sqrt(5));
      for (let i0 = 0; i0 < 2; i0++) {
        for (let i1 = 0; i1 < 3; i1++) {
          for (let i2 = 0; i2 < 4; i2++) {
            x0 = 0;
            for (let i3 = 0; i3 < 5; i3++) {
              x0 = x0 + b0[5 * (3 * i0 + i1) + i3] * a0[0] * (b1[5 * (4 * i0 + i2) + i3] * a0[0]);
            }
            a0[1 + i2] = x0 + b3[4 * (3 * i0 + i1) + i2];
          }
          x1 = -Infinity;
          for (let i4 = 0; i4 < 4; i4++) {
            x1 = Math.max(x1, a0[1 + i4]);
          }
          a0[5] = x1;
          x2 = 0;
          for (let i5 = 0; i5 < 4; i5++) {
            x2 = x2 + Math.exp(a0[1 + i5] - a0[5]);
          }
          a0[6] = x2;
          for (let i6 = 0; i6 < 4; i6++) {
            a0[7 + i6] = Math.exp(a0[1 + i6] - a0[5]) / a0[6];
          }
          for (let i7 = 0; i7 < 5; i7++) {
            if (-Infinity < a0[5]) {
              x3 = 0;
              for (let i8 = 0; i8 < 4; i8++) {
                x3 = x3 + a0[7 + i8] * b2[5 * (4 * i0 + i8) + i7];
              }
              x4 = x3;
            } else {
              x4 = 0;
            }
            b4[5 * (3 * i0 + i1) + i7] = x4;
          }
        }
      }
      return null;
    } |}]

let%expect_test "softmax" =
  Loop_js_walk.print "softmax";
  [%expect
    {|
    // softmax {shape=[n=2 c=4 h=4 w=4] axis=C}
    "use strict";
    function loop_kernel(b0, b1) {
      let x0 = 0;
      let x1 = 0;
      const a0 = new Float64Array(2);
      for (let i0 = 0; i0 < 32; i0++) {
        x0 = -Infinity;
        for (let i1 = 0; i1 < 4; i1++) {
          x0 = Math.max(x0, b0[4 * i0 + i1]);
        }
        a0[0] = x0;
        x1 = 0;
        for (let i2 = 0; i2 < 4; i2++) {
          x1 = x1 + Math.exp(b0[4 * i0 + i2] - a0[0]);
        }
        a0[1] = x1;
        for (let i3 = 0; i3 < 4; i3++) {
          b1[4 * i0 + i3] = Math.exp(b0[4 * i0 + i3] - a0[0]) / a0[1];
        }
      }
      return null;
    } |}]
