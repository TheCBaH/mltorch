(* Random-walk equivalence for native ops (2/2): the second half of
   native_walk_coverage_test.ml's own split -- see that file's header for
   why. This half covers walks [36, length) of [all_walks], with the seed
   offset by 36 so every step here draws EXACTLY the same PCG sequence it
   would have in the merged file (seed = absolute index in [all_walks],
   not position within this sublist). Pure OCaml (no libtorch) -- the
   oracle is Direct vs Symbolic. *)

let capture f = print_string (Core.Pretty.capture_to_string f)

module Pcg = Walk_core.Pcg

let%expect_test "native walk coverage (2/2)" =
  capture (fun ppf ->
      List.iteri
        (fun i m ->
          assert (
            Native_op_walk.run m ~ppf
              ~pcg:(Pcg.seed ~seed:(Int64.of_int (i + 36)) ~seq:1L)
              ~steps:5))
        (List.filteri (fun i _ -> i >= 36) Native_op_walk.all_walks));
  [%expect
    {|
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] mul_scalar: direct==symbolic
    step 1 [scalar]: [n=1 c=3 h=4 w=4] scalar=3
    [native] mul_scalar: direct==symbolic
    step 2 [scalar]: [n=1 c=3 h=4 w=4] scalar=-2
    [native] mul_scalar: direct==symbolic
    step 3 [input]: [n=1 c=3 h=3 w=4] scalar=-2
    [native] mul_scalar: direct==symbolic
    step 4 [input]: [n=1 c=3 h=3 w=15] scalar=-2
    [native] mul_scalar: direct==symbolic
    step 5 [input]: [n=1 c=3 h=3 w=6] scalar=-2
    [native] mul_scalar: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] mul_scalar_i64: direct==symbolic
    step 1 [scalar]: [n=1 c=3 h=4 w=4] scalar=3
    [native] mul_scalar_i64: direct==symbolic
    step 2 [scalar]: [n=1 c=3 h=4 w=4] scalar=0.5
    [native] mul_scalar_i64: direct==symbolic
    step 3 [input]: [n=1 c=3 h=4 w=15] scalar=0.5
    [native] mul_scalar_i64: direct==symbolic
    step 4 [scalar]: [n=1 c=3 h=4 w=15] scalar=6
    [native] mul_scalar_i64: direct==symbolic
    step 5 [input]: [n=1 c=3 h=4 w=7] scalar=6
    [native] mul_scalar_i64: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] ne_scalar: direct==symbolic
    step 1 [scalar]: [n=1 c=3 h=4 w=4] scalar=-2
    [native] ne_scalar: direct==symbolic
    step 2 [input]: [n=1 c=3 h=3 w=4] scalar=-2
    [native] ne_scalar: direct==symbolic
    step 3 [input]: [n=1 c=3 h=3 w=4] scalar=-2
    [native] ne_scalar: direct==symbolic
    step 4 [scalar]: [n=1 c=3 h=3 w=4] scalar=6
    [native] ne_scalar: direct==symbolic
    step 5 [input]: [n=1 c=3 h=5 w=4] scalar=6
    [native] ne_scalar: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] ne_tensor: direct==symbolic
    step 1 [input]: [n=1 c=3 h=6 w=4]
    [native] ne_tensor: direct==symbolic
    step 2 [input]: [n=1 c=3 h=6 w=4]
    [native] ne_tensor: direct==symbolic
    step 3 [input]: [n=1 c=3 h=6 w=15]
    [native] ne_tensor: direct==symbolic
    step 4 [input]: [n=1 c=14 h=6 w=15]
    [native] ne_tensor: direct==symbolic
    step 5 [input]: [n=1 c=14 h=6 w=9]
    [native] ne_tensor: direct==symbolic
    step 0: {shape=[n=1 c=3 h=6 w=6] pattern=pad_hw}
    [native] pad: direct==symbolic
    step 1 [pattern]: {shape=[n=1 c=3 h=6 w=6] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 2 [pattern]: {shape=[n=1 c=3 h=6 w=6] pattern=reflect_crop_h}
    [native] pad: direct==symbolic
    step 3 [pattern]: {shape=[n=1 c=3 h=6 w=6] pattern=pad_hw}
    [native] pad: direct==symbolic
    step 4 [pattern]: {shape=[n=1 c=3 h=6 w=6] pattern=reflect_asym_w}
    [native] pad: direct==symbolic
    step 5 [input]: {shape=[n=1 c=3 h=6 w=12] pattern=reflect_asym_w}
    [native] pad: direct==symbolic
    step 0: {shape=[n=1 c=4 h=4 w=4] perm=[H<-W, W<-H]}
    [native] permute: direct==symbolic
    step 1 [perm]: {shape=[n=1 c=4 h=4 w=4] perm=[N<-C, C<-N]}
    [native] permute: direct==symbolic
    step 2 [input]: {shape=[n=1 c=4 h=1 w=4] perm=[N<-C, C<-N]}
    [native] permute: direct==symbolic
    step 3 [input]: {shape=[n=1 c=8 h=1 w=4] perm=[N<-C, C<-N]}
    [native] permute: direct==symbolic
    step 4 [perm]: {shape=[n=1 c=8 h=1 w=4] perm=[H<-C, C<-H]}
    [native] permute: direct==symbolic
    step 5 [perm]: {shape=[n=1 c=8 h=1 w=4] perm=[H<-W, W<-H]}
    [native] permute: direct==symbolic
    step 0: {shape=[n=1 c=4 h=4 w=4] perm=[H<-W, W<-H]}
    [native] permute_i64: direct==symbolic
    step 1 [input]: {shape=[n=1 c=4 h=6 w=4] perm=[H<-W, W<-H]}
    [native] permute_i64: direct==symbolic
    step 2 [perm]: {shape=[n=1 c=4 h=6 w=4] perm=[N<-H, H<-N]}
    [native] permute_i64: direct==symbolic
    step 3 [perm]: {shape=[n=1 c=4 h=6 w=4] perm=[H<-W, W<-C, C<-H]}
    [native] permute_i64: direct==symbolic
    step 4 [input]: {shape=[n=1 c=4 h=6 w=3] perm=[H<-W, W<-C, C<-H]}
    [native] permute_i64: direct==symbolic
    step 5 [input]: {shape=[n=1 c=29 h=6 w=3] perm=[H<-W, W<-C, C<-H]}
    [native] permute_i64: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] pow: direct==symbolic
    step 1 [scalar]: [n=1 c=3 h=4 w=4] scalar=3
    [native] pow: direct==symbolic
    step 2 [scalar]: [n=1 c=3 h=4 w=4] scalar=6
    [native] pow: direct==symbolic
    step 3 [input]: [n=1 c=22 h=4 w=4] scalar=6
    [native] pow: direct==symbolic
    step 4 [input]: [n=1 c=22 h=4 w=4] scalar=6
    [native] pow: direct==symbolic
    step 5 [input]: [n=1 c=27 h=4 w=4] scalar=6
    [native] pow: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] relu: direct==symbolic
    step 1 [input]: [n=1 c=3 h=5 w=4]
    [native] relu: direct==symbolic
    step 2 [input]: [n=1 c=3 h=8 w=4]
    [native] relu: direct==symbolic
    step 3 [input]: [n=1 c=32 h=8 w=4]
    [native] relu: direct==symbolic
    step 4 [input]: [n=1 c=19 h=8 w=4]
    [native] relu: direct==symbolic
    step 5 [input]: [n=1 c=19 h=8 w=2]
    [native] relu: direct==symbolic
    step 0: {shape=[n=1 c=4 h=4 w=4] -> flat}
    [native] reshape: direct==symbolic
    step 1 [input]: {shape=[n=1 c=17 h=4 w=4] -> flat}
    [native] reshape: direct==symbolic
    step 2 [input]: {shape=[n=1 c=17 h=4 w=1] -> flat}
    [native] reshape: direct==symbolic
    step 3 [input]: {shape=[n=1 c=17 h=3 w=1] -> flat}
    [native] reshape: direct==symbolic
    step 4 [input]: {shape=[n=1 c=17 h=14 w=1] -> flat}
    [native] reshape: direct==symbolic
    step 5 [input]: {shape=[n=1 c=17 h=14 w=1] -> flat}
    [native] reshape: direct==symbolic
    step 0: {shape=[n=1 c=4 h=4 w=4] -> flat}
    [native] reshape_i64: direct==symbolic
    step 1 [input]: {shape=[n=1 c=4 h=6 w=4] -> flat}
    [native] reshape_i64: direct==symbolic
    step 2 [input]: {shape=[n=2 c=4 h=6 w=4] -> flat}
    [native] reshape_i64: direct==symbolic
    step 3 [input]: {shape=[n=2 c=10 h=6 w=4] -> flat}
    [native] reshape_i64: direct==symbolic
    step 4 [input]: {shape=[n=2 c=10 h=10 w=4] -> flat}
    [native] reshape_i64: direct==symbolic
    step 5 [input]: {shape=[n=2 c=21 h=10 w=4] -> flat}
    [native] reshape_i64: direct==symbolic
    step 0: {shape=[n=1 c=5 h=4 w=3] k=1 eps=1e-05 weight=true}
    [native] rms_norm: direct==symbolic
    step 1 [k]: {shape=[n=1 c=5 h=4 w=3] k=3 eps=1e-05 weight=true}
    [native] rms_norm: direct==symbolic
    step 2 [weight]: {shape=[n=1 c=5 h=4 w=3] k=3 eps=1e-05 weight=false}
    [native] rms_norm: direct==symbolic
    step 3 [input]: {shape=[n=1 c=5 h=4 w=3] k=3 eps=1e-05 weight=false}
    [native] rms_norm: direct==symbolic
    step 4 [eps]: {shape=[n=1 c=5 h=4 w=3] k=3 eps=0 weight=false}
    [native] rms_norm: direct==symbolic
    step 5 [weight]: {shape=[n=1 c=5 h=4 w=3] k=3 eps=0 weight=true}
    [native] rms_norm: direct==symbolic
    step 0: {batch=1 heads=2 wq=3 wk=4 e=5 mask=present scale=default}
    [native] sdpa: direct==symbolic
    step 1 [wk]: {batch=1 heads=2 wq=3 wk=1 e=5 mask=present scale=default}
    [native] sdpa: direct==symbolic
    step 2 [mask]: {batch=1 heads=2 wq=3 wk=1 e=5 mask=present scale=default}
    [native] sdpa: direct==symbolic
    step 3 [heads]: {batch=1 heads=3 wq=3 wk=1 e=5 mask=present scale=default}
    [native] sdpa: direct==symbolic
    step 4 [mask]: {batch=1 heads=3 wq=3 wk=1 e=5 mask=present scale=default}
    [native] sdpa: direct==symbolic
    step 5 [wq]: {batch=1 heads=3 wq=5 wk=1 e=5 mask=present scale=default}
    [native] sdpa: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] sigmoid: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=9]
    [native] sigmoid: direct==symbolic
    step 2 [input]: [n=1 c=3 h=1 w=9]
    [native] sigmoid: direct==symbolic
    step 3 [input]: [n=1 c=29 h=1 w=9]
    [native] sigmoid: direct==symbolic
    step 4 [input]: [n=1 c=29 h=2 w=9]
    [native] sigmoid: direct==symbolic
    step 5 [input]: [n=1 c=29 h=2 w=9]
    [native] sigmoid: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] silu: direct==symbolic
    step 1 [input]: [n=1 c=3 h=14 w=4]
    [native] silu: direct==symbolic
    step 2 [input]: [n=1 c=3 h=14 w=3]
    [native] silu: direct==symbolic
    step 3 [input]: [n=1 c=3 h=14 w=11]
    [native] silu: direct==symbolic
    step 4 [input]: [n=1 c=3 h=14 w=2]
    [native] silu: direct==symbolic
    step 5 [input]: [n=1 c=3 h=14 w=2]
    [native] silu: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=H start=0 stop=2 step=1}
    [native] slice: direct==symbolic
    step 1 [axis]: {shape=[n=2 c=4 h=4 w=4] axis=H start=0 stop=2 step=1}
    [native] slice: direct==symbolic
    step 2 [axis]: {shape=[n=2 c=4 h=4 w=4] axis=C start=0 stop=2 step=1}
    [native] slice: direct==symbolic
    step 3 [pattern]: {shape=[n=2 c=4 h=4 w=4] axis=C start=0 stop=4 step=2}
    [native] slice: direct==symbolic
    step 4 [axis]: {shape=[n=2 c=4 h=4 w=4] axis=H start=0 stop=4 step=2}
    [native] slice: direct==symbolic
    step 5 [axis]: {shape=[n=2 c=4 h=4 w=4] axis=W start=0 stop=4 step=2}
    [native] slice: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=C}
    [native] softmax: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=15 w=4] axis=C}
    [native] softmax: direct==symbolic
    step 2 [input]: {shape=[n=2 c=4 h=15 w=4] axis=C}
    [native] softmax: direct==symbolic
    step 3 [axis]: {shape=[n=2 c=4 h=15 w=4] axis=D}
    [native] softmax: direct==symbolic
    step 4 [axis]: {shape=[n=2 c=4 h=15 w=4] axis=T}
    [native] softmax: direct==symbolic
    step 5 [input]: {shape=[n=2 c=26 h=15 w=4] axis=T}
    [native] softmax: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] sqrt: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=4]
    [native] sqrt: direct==symbolic
    step 2 [input]: [n=1 c=3 h=4 w=14]
    [native] sqrt: direct==symbolic
    step 3 [input]: [n=1 c=2 h=4 w=14]
    [native] sqrt: direct==symbolic
    step 4 [input]: [n=2 c=2 h=4 w=14]
    [native] sqrt: direct==symbolic
    step 5 [input]: [n=1 c=2 h=4 w=14]
    [native] sqrt: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] sub: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=16]
    [native] sub: direct==symbolic
    step 2 [input]: [n=2 c=3 h=4 w=16]
    [native] sub: direct==symbolic
    step 3 [input]: [n=2 c=3 h=4 w=11]
    [native] sub: direct==symbolic
    step 4 [input]: [n=2 c=1 h=4 w=11]
    [native] sub: direct==symbolic
    step 5 [input]: [n=2 c=1 h=4 w=3]
    [native] sub: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] sub_i64: direct==symbolic
    step 1 [input]: [n=1 c=2 h=4 w=4]
    [native] sub_i64: direct==symbolic
    step 2 [input]: [n=1 c=2 h=14 w=4]
    [native] sub_i64: direct==symbolic
    step 3 [input]: [n=1 c=2 h=14 w=4]
    [native] sub_i64: direct==symbolic
    step 4 [input]: [n=1 c=7 h=14 w=4]
    [native] sub_i64: direct==symbolic
    step 5 [input]: [n=1 c=7 h=14 w=14]
    [native] sub_i64: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    [native] sum: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    [native] sum: direct==symbolic
    step 2 [keepdim]: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=true}
    [native] sum: direct==symbolic
    step 3 [keepdim]: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    [native] sum: direct==symbolic
    step 4 [keepdim]: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=true}
    [native] sum: direct==symbolic
    step 5 [dims]: {shape=[n=2 c=4 h=4 w=4] dims=[N,H,W] keepdim=true}
    [native] sum: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] to_copy_bool: direct==symbolic
    step 1 [input]: [n=1 c=3 h=9 w=4]
    [native] to_copy_bool: direct==symbolic
    step 2 [input]: [n=1 c=3 h=11 w=4]
    [native] to_copy_bool: direct==symbolic
    step 3 [input]: [n=1 c=3 h=10 w=4]
    [native] to_copy_bool: direct==symbolic
    step 4 [input]: [n=1 c=3 h=6 w=4]
    [native] to_copy_bool: direct==symbolic
    step 5 [input]: [n=1 c=3 h=6 w=14]
    [native] to_copy_bool: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] to_copy_float_i64: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=4]
    [native] to_copy_float_i64: direct==symbolic
    step 2 [input]: [n=1 c=3 h=6 w=4]
    [native] to_copy_float_i64: direct==symbolic
    step 3 [input]: [n=2 c=3 h=6 w=4]
    [native] to_copy_float_i64: direct==symbolic
    step 4 [input]: [n=2 c=3 h=6 w=4]
    [native] to_copy_float_i64: direct==symbolic
    step 5 [input]: [n=2 c=29 h=6 w=4]
    [native] to_copy_float_i64: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=H}
    [native] unbind: direct==symbolic
    step 1 [axis]: {shape=[n=2 c=4 h=4 w=4] axis=W}
    [native] unbind: direct==symbolic
    step 2 [axis]: {shape=[n=2 c=4 h=4 w=4] axis=N}
    [native] unbind: direct==symbolic
    step 3 [input]: {shape=[n=2 c=2 h=4 w=4] axis=N}
    [native] unbind: direct==symbolic
    step 4 [input]: {shape=[n=2 c=2 h=4 w=5] axis=N}
    [native] unbind: direct==symbolic
    step 5 [input]: {shape=[n=2 c=2 h=4 w=13] axis=N}
    [native] unbind: direct==symbolic
    step 0: {shape=[1,4,8,8] output_size=[4,4] align_corners=true}
    [native] upsample_bicubic2d: direct==symbolic
    step 1 [n]: {shape=[1,4,8,8] output_size=[4,4] align_corners=true}
    [native] upsample_bicubic2d: direct==symbolic
    step 2 [n]: {shape=[2,4,8,8] output_size=[4,4] align_corners=true}
    [native] upsample_bicubic2d: direct==symbolic
    step 3 [input_w]: {shape=[2,4,8,10] output_size=[4,4] align_corners=true}
    [native] upsample_bicubic2d: direct==symbolic
    step 4 [input_h]: {shape=[2,4,3,10] output_size=[4,4] align_corners=true}
    [native] upsample_bicubic2d: direct==symbolic
    step 5 [n]: {shape=[1,4,3,10] output_size=[4,4] align_corners=true}
    [native] upsample_bicubic2d: direct==symbolic
    step 0: {shape=[1,4,8,8] output_size=[4,4] align_corners=true}
    [native] upsample_bilinear2d: direct==symbolic
    step 1 [n]: {shape=[2,4,8,8] output_size=[4,4] align_corners=true}
    [native] upsample_bilinear2d: direct==symbolic
    step 2 [align_corners]: {shape=[2,4,8,8] output_size=[4,4] align_corners=true}
    [native] upsample_bilinear2d: direct==symbolic
    step 3 [out_w]: {shape=[2,4,8,8] output_size=[4,5] align_corners=true}
    [native] upsample_bilinear2d: direct==symbolic
    step 4 [c]: {shape=[2,1,8,8] output_size=[4,5] align_corners=true}
    [native] upsample_bilinear2d: direct==symbolic
    step 5 [input_w]: {shape=[2,1,8,4] output_size=[4,5] align_corners=true}
    [native] upsample_bilinear2d: direct==symbolic
    step 0: {shape=[1,4,8,8] output_size=[4,4]}
    [native] upsample_nearest2d: direct==symbolic
    step 1 [input_h]: {shape=[1,4,6,8] output_size=[4,4]}
    [native] upsample_nearest2d: direct==symbolic
    step 2 [out_w]: {shape=[1,4,6,8] output_size=[4,5]}
    [native] upsample_nearest2d: direct==symbolic
    step 3 [c]: {shape=[1,2,6,8] output_size=[4,5]}
    [native] upsample_nearest2d: direct==symbolic
    step 4 [c]: {shape=[1,6,6,8] output_size=[4,5]}
    [native] upsample_nearest2d: direct==symbolic
    step 5 [c]: {shape=[1,4,6,8] output_size=[4,5]}
    [native] upsample_nearest2d: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    [native] vector_norm: direct==symbolic
    step 1 [keepdim]: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=true}
    [native] vector_norm: direct==symbolic
    step 2 [input]: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=true}
    [native] vector_norm: direct==symbolic
    step 3 [input]: {shape=[n=2 c=4 h=4 w=12] dims=[H,W] keepdim=true}
    [native] vector_norm: direct==symbolic
    step 4 [dims]: {shape=[n=2 c=4 h=4 w=12] dims=[H,W] keepdim=true}
    [native] vector_norm: direct==symbolic
    step 5 [input]: {shape=[n=2 c=4 h=4 w=2] dims=[H,W] keepdim=true}
    [native] vector_norm: direct==symbolic
    step 0: {shape=[n=2 c=4 h=3 w=2] eps=1e-05 weight=true bias=true}
    [native] batch_norm_no_stats: direct==symbolic
    step 1 [eps]: {shape=[n=2 c=4 h=3 w=2] eps=1e-05 weight=true bias=true}
    [native] batch_norm_no_stats: direct==symbolic
    step 2 [eps]: {shape=[n=2 c=4 h=3 w=2] eps=1e-05 weight=true bias=true}
    [native] batch_norm_no_stats: direct==symbolic
    step 3 [weight]: {shape=[n=2 c=4 h=3 w=2] eps=1e-05 weight=true bias=true}
    [native] batch_norm_no_stats: direct==symbolic
    step 4 [input]: {shape=[n=2 c=4 h=3 w=6] eps=1e-05 weight=true bias=true}
    [native] batch_norm_no_stats: direct==symbolic
    step 5 [eps]: {shape=[n=2 c=4 h=3 w=6] eps=1e-08 weight=true bias=true}
    [native] batch_norm_no_stats: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] eye: direct==symbolic
    step 1 [shape]: [n=1 c=23 h=4 w=4]
    [native] eye: direct==symbolic
    step 2 [shape]: [n=1 c=14 h=4 w=4]
    [native] eye: direct==symbolic
    step 3 [shape]: [n=1 c=14 h=8 w=4]
    [native] eye: direct==symbolic
    step 4 [shape]: [n=1 c=14 h=7 w=4]
    [native] eye: direct==symbolic
    step 5 [shape]: [n=1 c=14 h=7 w=10]
    [native] eye: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] zeros: direct==symbolic
    step 1 [shape]: [n=2 c=3 h=4 w=4]
    [native] zeros: direct==symbolic
    step 2 [shape]: [n=2 c=3 h=7 w=4]
    [native] zeros: direct==symbolic
    step 3 [shape]: [n=2 c=3 h=12 w=4]
    [native] zeros: direct==symbolic
    step 4 [shape]: [n=2 c=11 h=12 w=4]
    [native] zeros: direct==symbolic
    step 5 [shape]: [n=1 c=11 h=12 w=4]
    [native] zeros: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] to_copy_long: direct==symbolic
    step 1 [input]: [n=1 c=3 h=12 w=4]
    [native] to_copy_long: direct==symbolic
    step 2 [input]: [n=1 c=12 h=12 w=4]
    [native] to_copy_long: direct==symbolic
    step 3 [input]: [n=1 c=12 h=12 w=8]
    [native] to_copy_long: direct==symbolic
    step 4 [input]: [n=1 c=12 h=12 w=7]
    [native] to_copy_long: direct==symbolic
    step 5 [input]: [n=1 c=17 h=12 w=7]
    [native] to_copy_long: direct==symbolic
    step 0: count=4
    [native] arange: direct==symbolic
    step 1 [count]: count=7
    [native] arange: direct==symbolic
    step 2 [count]: count=25
    [native] arange: direct==symbolic
    step 3 [count]: count=32
    [native] arange: direct==symbolic
    step 4 [count]: count=10
    [native] arange: direct==symbolic
    step 5 [count]: count=12
    [native] arange: direct==symbolic
    step 0: count=4
    [native] arange_i64: direct==symbolic
    step 1 [count]: count=6
    [native] arange_i64: direct==symbolic
    step 2 [count]: count=12
    [native] arange_i64: direct==symbolic
    step 3 [count]: count=12
    [native] arange_i64: direct==symbolic
    step 4 [count]: count=25
    [native] arange_i64: direct==symbolic
    step 5 [count]: count=19
    [native] arange_i64: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=H}
    [native] unbind_i64: direct==symbolic
    step 1 [axis]: {shape=[n=2 c=4 h=4 w=4] axis=W}
    [native] unbind_i64: direct==symbolic
    step 2 [axis]: {shape=[n=2 c=4 h=4 w=4] axis=H}
    [native] unbind_i64: direct==symbolic
    step 3 [axis]: {shape=[n=2 c=4 h=4 w=4] axis=N}
    [native] unbind_i64: direct==symbolic
    step 4 [input]: {shape=[n=2 c=18 h=4 w=4] axis=N}
    [native] unbind_i64: direct==symbolic
    step 5 [input]: {shape=[n=2 c=18 h=5 w=4] axis=N}
    [native] unbind_i64: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=H sizes=[2, 2]}
    [native] split_with_sizes: direct==symbolic
    step 1 [pattern]: {shape=[n=2 c=4 h=4 w=4] axis=H sizes=[1, 3]}
    [native] split_with_sizes: direct==symbolic
    step 2 [pattern]: {shape=[n=2 c=4 h=4 w=4] axis=H sizes=[1, 1, 1, 1]}
    [native] split_with_sizes: direct==symbolic
    step 3 [input]: {shape=[n=2 c=4 h=12 w=4] axis=H sizes=[1, 1, 1, 1, 1, 1, 1,
                                                            1, 1, 1, 1, 1]}
    [native] split_with_sizes: direct==symbolic
    step 4 [pattern]: {shape=[n=2 c=4 h=12 w=4] axis=H sizes=[4, 4, 4]}
    [native] split_with_sizes: direct==symbolic
    step 5 [axis]: {shape=[n=2 c=4 h=12 w=4] axis=H sizes=[4, 4, 4]}
    [native] split_with_sizes: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=H sizes=[2, 2]}
    [native] split_with_sizes_i64: direct==symbolic
    step 1 [axis]: {shape=[n=2 c=4 h=4 w=4] axis=C sizes=[2, 2]}
    [native] split_with_sizes_i64: direct==symbolic
    step 2 [input]: {shape=[n=2 c=4 h=4 w=7] axis=C sizes=[2, 2]}
    [native] split_with_sizes_i64: direct==symbolic
    step 3 [axis]: {shape=[n=2 c=4 h=4 w=7] axis=N sizes=[1, 1]}
    [native] split_with_sizes_i64: direct==symbolic
    step 4 [pattern]: {shape=[n=2 c=4 h=4 w=7] axis=N sizes=[1, 1]}
    [native] split_with_sizes_i64: direct==symbolic
    step 5 [pattern]: {shape=[n=2 c=4 h=4 w=7] axis=N sizes=[1, 1]}
    [native] split_with_sizes_i64: direct==symbolic |}]
