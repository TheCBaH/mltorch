(* The exact `max_pool2d.default` overload through the serialized path.

   Like the other Group-2 rows, nothing else reaches this arm: every model this
   repository can download is exported post-decomposition and carries
   `max_pool2d_with_indices.default`. That other arm exists and is NOT what this
   one routes through -- it is a different overload with two outputs, and
   discarding the indices to serve this one would put the functional target in
   [materialized_output_names], where it does not belong. *)

open Programs

(* The int-list arguments are spelled as WHOLE serialized values rather than as
   bracket text, because "the argument is an explicit none" is one of the
   spellings under test and [{"as_ints": ...}] cannot express it. *)
let ints xs = jstr {|{"as_ints":%s}|} xs
let none = {|{"as_none":true}|}

let pool ?kernel ?stride ?padding ?dilation ?ceil_mode () =
  let arg name spelling =
    match spelling with
    | None -> ""
    | Some v -> jstr {|,{"name":"%s","arg":%s,"kind":1}|} name v
  in
  jstr
    {|{"target":"torch.ops.aten.max_pool2d.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"kernel_size","arg":%s,"kind":1}%s%s%s%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x")
    (Option.value kernel ~default:(ints "[2,2]"))
    (arg "stride" stride) (arg "padding" padding) (arg "dilation" dilation)
    (arg "ceil_mode" (Option.map (jstr {|{"as_bool":%b}|}) ceil_mode))
    (as_tensor "y")

let prog ?(x_sizes = [ 1; 3; 8; 8 ]) node =
  program ~x_sizes ~nodes:[ node ] ~graph_outputs:[ as_tensor "y" ] ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Graph_ir.pp Format.std_formatter l.Pt2_native_graph.graph

let%expect_test "max_pool2d.default lowers with a defaulted stride" =
  dump "stride omitted:" (prog (pool ()));
  [%expect
    {|
    stride omitted:
    graph
    inputs: [t0 f32 [H=3 W=8 C=8] ->[n0]]
    nodes:
      group g1 torch.ops.aten.max_pool2d.default:
        n0: [t1 f32 [H=8 W=8 C=3] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=4 W=4 C=3] ->[n2]] =
          max_pool2d
            x=t1 <-n0
            params={kernel={h=2; w=2};
                   stride={h=2; w=2};
                   pad={h=0; w=0};
                   ceil_mode=false}
        n2: [t3 f32 [H=3 W=4 C=4]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [H=3 W=4 C=4] <-n2] |}]

(* Three spellings of "stride defaults to the kernel": absent, an explicit empty
   list, and the value written out. PyTorch treats the first two the same, and
   an arm that normalized only one of them would accept a real export and refuse
   its equivalent. *)
(* A fourth spelling, and the one nothing covered: an explicit none. [ints_arg]
   has always mapped it to the default, and removing that arm left the whole
   suite green -- so the arm was load-bearing and unwitnessed. Every [int[]?]
   argument the importer reads goes through it. *)
let%expect_test "an explicit none int list falls back to the default" =
  show "stride none:" (prog (pool ~stride:none ()));
  show "padding none:" (prog (pool ~padding:none ()));
  [%expect
    {|
    stride none:               lowered, nodes=3
    padding none:              lowered, nodes=3 |}]

let%expect_test "an omitted, empty and explicit stride agree" =
  show "absent:" (prog (pool ()));
  show "empty list:" (prog (pool ~stride:(ints "[]") ()));
  show "explicit:" (prog (pool ~stride:(ints "[2,2]") ()));
  [%expect
    {|
    absent:                    lowered, nodes=3
    empty list:                lowered, nodes=3
    explicit:                  lowered, nodes=3 |}]

(* Square configurations are invariant under a transposed H/W pair, so the
   rectangular case is the one that can falsify it. *)
let%expect_test "rectangular kernel, stride and padding keep their axes" =
  dump "rectangular:"
    (prog ~x_sizes:[ 1; 3; 9; 7 ]
       (pool ~kernel:(ints "[3,2]") ~stride:(ints "[2,1]")
          ~padding:(ints "[1,0]") ()));
  [%expect
    {|
    rectangular:
    graph
    inputs: [t0 f32 [H=3 W=9 C=7] ->[n0]]
    nodes:
      group g1 torch.ops.aten.max_pool2d.default:
        n0: [t1 f32 [H=9 W=7 C=3] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=5 W=6 C=3] ->[n2]] =
          max_pool2d
            x=t1 <-n0
            params={kernel={h=3; w=2};
                   stride={h=2; w=1};
                   pad={h=1; w=0};
                   ceil_mode=false}
        n2: [t3 f32 [H=3 W=5 C=6]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [H=3 W=5 C=6] <-n2] |}]

let%expect_test "single-element lists are accepted as symmetric" =
  show "one-element:"
    (prog
       (pool ~kernel:(ints "[2]") ~stride:(ints "[2]") ~padding:(ints "[0]") ()));
  [%expect {| one-element:               lowered, nodes=3 |}]

(* ---- rejections --------------------------------------------------------- *)

(* A window WIDER than the padded input. The formula's numerator goes negative,
   and the division it feeds means FLOOR -- but [/] and [Int64.div] both truncate
   toward zero, so (1 - 2) / 2 + 1 came out as 1 and one output position was
   accepted from a partial window. ATen refuses the same configuration
   ("Calculated output size: (1x0x0). Output size is too small").

   The asymmetric case matters separately: with stride 2 the H axis is the one
   that goes negative and W does not, so a rule applied to only one axis still
   looks right. *)
let%expect_test "a window wider than the padded input is refused" =
  show "1x1 in, 2x2 kernel:"
    (prog ~x_sizes:[ 1; 1; 1; 1 ] (pool ~kernel:(ints "[2,2]") ()));
  show "3x1 in, 2x2 kernel:"
    (prog ~x_sizes:[ 1; 1; 3; 1 ] (pool ~kernel:(ints "[2,2]") ()));
  show "1x1 in, padded to fit:"
    (prog ~x_sizes:[ 1; 1; 1; 1 ]
       (pool ~kernel:(ints "[2,2]") ~padding:(ints "[1,1]") ()));
  [%expect
    {|
    1x1 in, 2x2 kernel:        output extent must be >= 1, got 0 (in=1 kernel=2 stride=2 pad_before=0 pad_after=0 dilation=1)
    3x1 in, 2x2 kernel:        output extent must be >= 1, got 0 (in=1 kernel=2 stride=2 pad_before=0 pad_after=0 dilation=1)
    1x1 in, padded to fit:     lowered, nodes=3 |}]

(* [dilation] has no field on [Pool.MaxPool2d.params], so silently dropping a
   non-default value would compute a different op under the right name --
   refused rather than approximated. [ceil_mode] DOES have a field (see
   [Pool.MaxPool2d.params.ceil_mode]), so both its spellings just lower; the
   shape effect is exercised separately below. *)
let%expect_test "dilation is refused, not dropped; ceil_mode is carried" =
  show "dilation 1,1:" (prog (pool ~dilation:(ints "[1,1]") ()));
  show "dilation 2,2:" (prog (pool ~dilation:(ints "[2,2]") ()));
  show "dilation 1,2:" (prog (pool ~dilation:(ints "[1,2]") ()));
  show "ceil_mode false:" (prog (pool ~ceil_mode:false ()));
  show "ceil_mode true:" (prog (pool ~ceil_mode:true ()));
  (* Arity, not just value. ATen refuses a three-element dilation outright
     ("dilation must be either a single int, or a tuple of two ints"), and a
     value-only test accepted it -- and [] -- because no element differed from
     one. Normalizing through [hw2] first keeps the typed arity diagnostic that
     kernel_size, stride and padding already get. *)
  show "dilation [1]:" (prog (pool ~dilation:(ints "[1]") ()));
  show "dilation [1,1,1]:" (prog (pool ~dilation:(ints "[1,1,1]") ()));
  show "dilation []:" (prog (pool ~dilation:(ints "[]") ()));
  [%expect
    {|
    dilation 1,1:              lowered, nodes=3
    dilation 2,2:              malformed PT2 graph: torch.ops.aten.max_pool2d.default: dilation=[2,2] is not supported (only 1)
    dilation 1,2:              malformed PT2 graph: torch.ops.aten.max_pool2d.default: dilation=[1,2] is not supported (only 1)
    ceil_mode false:           lowered, nodes=3
    ceil_mode true:            lowered, nodes=3
    dilation [1]:              lowered, nodes=3
    dilation [1,1,1]:          malformed PT2 graph: dilation must have one or two values, got 3
    dilation []:               malformed PT2 graph: dilation must have one or two values, got 0 |}]

(* An odd input extent (5) with a 2x2 kernel/stride and no padding is the case
   floor and ceil division disagree on: floor gives 2 ((5-2)/2 + 1), ceil gives
   3 (ATen's `pooling_output_shape_pad_lr`, matching real ATen -- see
   `test/pt2_op_native_walk_cram.t`'s `max_pool2d_with_indices.default` walk,
   which fuzzes `ceil_mode=true` against libtorch directly). *)
let%expect_test "ceil_mode changes the output extent" =
  dump "floor (ceil_mode=false):"
    (prog ~x_sizes:[ 1; 1; 5; 5 ] (pool ~ceil_mode:false ()));
  dump "ceil (ceil_mode=true):"
    (prog ~x_sizes:[ 1; 1; 5; 5 ] (pool ~ceil_mode:true ()));
  [%expect
    {|
    floor (ceil_mode=false):
    graph
    inputs: [t0 f32 [W=5 C=5] ->[n0]]
    nodes:
      group g1 torch.ops.aten.max_pool2d.default:
        n0: [t1 f32 [H=5 W=5 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=2 C=1] ->[n2]] =
          max_pool2d
            x=t1 <-n0
            params={kernel={h=2; w=2};
                   stride={h=2; w=2};
                   pad={h=0; w=0};
                   ceil_mode=false}
        n2: [t3 f32 [W=2 C=2]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=2 C=2] <-n2]ceil (ceil_mode=true):
    graph
    inputs: [t0 f32 [W=5 C=5] ->[n0]]
    nodes:
      group g1 torch.ops.aten.max_pool2d.default:
        n0: [t1 f32 [H=5 W=5 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=3 W=3 C=1] ->[n2]] =
          max_pool2d
            x=t1 <-n0
            params={kernel={h=2; w=2};
                   stride={h=2; w=2};
                   pad={h=0; w=0};
                   ceil_mode=true}
        n2: [t3 f32 [W=3 C=3]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=3 C=3] <-n2] |}]

let%expect_test "config faults are typed, not raised" =
  show "kernel 0:" (prog (pool ~kernel:(ints "[0,0]") ()));
  show "stride 0:" (prog (pool ~stride:(ints "[0,0]") ()));
  show "padding -1:" (prog (pool ~padding:(ints "[-1,-1]") ()));
  show "kernel arity:" (prog (pool ~kernel:(ints "[2,2,2]") ()));
  [%expect
    {|
    kernel 0:                  malformed PT2 graph: torch.ops.aten.max_pool2d.default: kernel_size must be positive, got 0
    stride 0:                  malformed PT2 graph: torch.ops.aten.max_pool2d.default: stride must be positive, got 0
    padding -1:                malformed PT2 graph: torch.ops.aten.max_pool2d.default: padding must not be negative, got -1
    kernel arity:              malformed PT2 graph: kernel_size must have one or two values, got 3 |}]

let adaptive ~output_size =
  jstr
    {|{"target":"torch.ops.aten.adaptive_avg_pool2d.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"output_size","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") output_size (as_tensor "y")

let%expect_test "adaptive_avg_pool2d accepts resolved SymInt[] and relayouts" =
  let symints = {|{"as_sym_ints":[{"as_int":3},{"as_int":2}]}|} in
  dump "adaptive:"
    (prog ~x_sizes:[ 1; 2; 5; 4 ] (adaptive ~output_size:symints));
  [%expect
    {|
    adaptive:
    graph
    inputs: [t0 f32 [H=2 W=5 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.adaptive_avg_pool2d.default:
        n0: [t1 f32 [H=5 W=4 C=2] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=3 W=2 C=2] ->[n2]] =
          adaptive_avg_pool2d x=t1 <-n0 params={output_size={h=3; w=2}}
        n2: [t3 f32 [H=2 W=3 C=2]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [H=2 W=3 C=2] <-n2] |}]

let%expect_test "adaptive_avg_pool2d rejects unsupported sizes and rank" =
  show "one element:" (prog (adaptive ~output_size:(ints "[1]")));
  show "zero:" (prog (adaptive ~output_size:(ints "[0,1]")));
  show "rank two:"
    (prog ~x_sizes:[ 2; 4 ] (adaptive ~output_size:(ints "[1,1]")));
  [%expect
    {|
    one element:               malformed PT2 graph: output_size must have exactly two values, got 1
    zero:                      malformed PT2 graph: torch.ops.aten.adaptive_avg_pool2d.default: output_size must be positive, got 0
    rank two:                  malformed PT2 graph: x must be rank-3 (CHW) or rank-4 (NCHW), got rank-2 |}]
