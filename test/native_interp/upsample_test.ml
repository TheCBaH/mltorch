(* `upsample_bilinear2d.vec` through the serialized path: explicit
   [output_size], a resolved [scale_factors], and the "exactly one of the
   two" contract's two rejections -- the same shape [Op_bridge]'s dispatch
   tests cover for the ATen-linked path (test/native_bridge_test.ml), here
   through the metadata-only importer instead. *)

open Programs

let ints xs = jstr {|{"as_ints":%s}|} xs
let floats xs = jstr {|{"as_floats":%s}|} xs
let none = {|{"as_none":true}|}

let upsample ~output_size ~align_corners ~scale_factors =
  jstr
    {|{"target":"torch.ops.aten.upsample_bilinear2d.vec","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"output_size","arg":%s,"kind":1},{"name":"align_corners","arg":{"as_bool":%b},"kind":1},{"name":"scale_factors","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") output_size align_corners scale_factors (as_tensor "y")

let prog ?(x_sizes = [ 1; 1; 2; 2 ]) node =
  program ~x_sizes ~nodes:[ node ] ~graph_outputs:[ as_tensor "y" ] ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Graph_ir.pp Format.std_formatter l.Pt2_native_graph.graph

let%expect_test "upsample_bilinear2d.vec lowers with an explicit output_size" =
  dump "explicit:"
    (prog
       (upsample ~output_size:(ints "[3,3]") ~align_corners:true
          ~scale_factors:none));
  [%expect
    {|
    explicit:
    graph
    inputs: [t0 f32 [W=2 C=2] ->[n0]]
    nodes:
      group g1 torch.ops.aten.upsample_bilinear2d.vec:
        n0: [t1 f32 [H=2 W=2 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=3 W=3 C=1] ->[n2]] =
          upsample_bilinear2d
            x=t1 <-n0
            params={output_size={h=3; w=3};
            align_corners=true}
        n2: [t3 f32 [W=3 C=3]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=3 C=3] <-n2] |}]

(* [scale_factors] resolved against the DECLARED input size (2x2), the same
   `floor(input_size * scale_factor)` [Op_bridge] performs -- 2 * 1.5 = 3 on
   each axis, so this builds the identical params the explicit-size fixture
   above does. *)
let%expect_test "upsample_bilinear2d.vec resolves scale_factors to a size" =
  dump "scale_factors:"
    (prog
       (upsample ~output_size:none ~align_corners:false
          ~scale_factors:(floats "[1.5,1.5]")));
  [%expect
    {|
    scale_factors:
    graph
    inputs: [t0 f32 [W=2 C=2] ->[n0]]
    nodes:
      group g1 torch.ops.aten.upsample_bilinear2d.vec:
        n0: [t1 f32 [H=2 W=2 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=3 W=3 C=1] ->[n2]] =
          upsample_bilinear2d
            x=t1 <-n0
            params={output_size={h=3; w=3};
            align_corners=false}
        n2: [t3 f32 [W=3 C=3]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=3 C=3] <-n2] |}]

let%expect_test "upsample_bilinear2d.vec rejects neither or both size args" =
  dump "neither:"
    (prog (upsample ~output_size:none ~align_corners:true ~scale_factors:none));
  dump "both:"
    (prog
       (upsample ~output_size:(ints "[3,3]") ~align_corners:true
          ~scale_factors:(floats "[1.5,1.5]")));
  [%expect
    {|
    neither:
      malformed PT2 graph: torch.ops.aten.upsample_bilinear2d.vec: exactly one of output_size or scale_factors must be given
    both:
      malformed PT2 graph: torch.ops.aten.upsample_bilinear2d.vec: output_size and scale_factors are mutually exclusive |}]

(* `upsample_nearest2d.vec`: no [align_corners] argument at all, unlike
   [upsample_bilinear2d.vec] above -- see [Resize.Nearest_axis]'s module
   doc. Otherwise the identical output_size/scale_factors contract, sharing
   [Native_interp_decode.resolve_upsample_size] with the bilinear arm. *)
let nearest ~output_size ~scale_factors =
  jstr
    {|{"target":"torch.ops.aten.upsample_nearest2d.vec","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"output_size","arg":%s,"kind":1},{"name":"scale_factors","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") output_size scale_factors (as_tensor "y")

let%expect_test "upsample_nearest2d.vec lowers with an explicit output_size" =
  dump "explicit:"
    (prog (nearest ~output_size:(ints "[3,3]") ~scale_factors:none));
  [%expect
    {|
    explicit:
    graph
    inputs: [t0 f32 [W=2 C=2] ->[n0]]
    nodes:
      group g1 torch.ops.aten.upsample_nearest2d.vec:
        n0: [t1 f32 [H=2 W=2 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=3 W=3 C=1] ->[n2]] =
          upsample_nearest2d x=t1 <-n0 params={output_size={h=3; w=3}}
        n2: [t3 f32 [W=3 C=3]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=3 C=3] <-n2] |}]

(* [scale_factors] resolved against the DECLARED input size (2x2), the same
   `floor(input_size * scale_factor)` [Op_bridge] performs -- 2 * 1.5 = 3 on
   each axis, so this builds the identical params the explicit-size fixture
   above does. *)
let%expect_test "upsample_nearest2d.vec resolves scale_factors to a size" =
  dump "scale_factors:"
    (prog (nearest ~output_size:none ~scale_factors:(floats "[1.5,1.5]")));
  [%expect
    {|
    scale_factors:
    graph
    inputs: [t0 f32 [W=2 C=2] ->[n0]]
    nodes:
      group g1 torch.ops.aten.upsample_nearest2d.vec:
        n0: [t1 f32 [H=2 W=2 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=3 W=3 C=1] ->[n2]] =
          upsample_nearest2d x=t1 <-n0 params={output_size={h=3; w=3}}
        n2: [t3 f32 [W=3 C=3]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=3 C=3] <-n2] |}]

let%expect_test "upsample_nearest2d.vec rejects neither or both size args" =
  dump "neither:" (prog (nearest ~output_size:none ~scale_factors:none));
  dump "both:"
    (prog
       (nearest ~output_size:(ints "[3,3]") ~scale_factors:(floats "[1.5,1.5]")));
  [%expect
    {|
    neither:
      malformed PT2 graph: torch.ops.aten.upsample_nearest2d.vec: exactly one of output_size or scale_factors must be given
    both:
      malformed PT2 graph: torch.ops.aten.upsample_nearest2d.vec: output_size and scale_factors are mutually exclusive |}]
