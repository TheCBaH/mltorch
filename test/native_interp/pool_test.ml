(* The exact `max_pool2d.default` overload through the serialized path.

   Like the other Group-2 rows, nothing else reaches this arm: every model this
   repository can download is exported post-decomposition and carries
   `max_pool2d_with_indices.default`. That other arm exists and is NOT what this
   one routes through -- it is a different overload with two outputs, and
   discarding the indices to serve this one would put the functional target in
   [materialized_output_names], where it does not belong. *)

open Programs

let pool ?kernel ?stride ?padding ?dilation ?ceil_mode () =
  let arg name spelling =
    match spelling with
    | None -> ""
    | Some v -> jstr {|,{"name":"%s","arg":%s,"kind":1}|} name v
  in
  jstr
    {|{"target":"torch.ops.aten.max_pool2d.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"kernel_size","arg":{"as_ints":%s},"kind":1}%s%s%s%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x")
    (Option.value kernel ~default:"[2,2]")
    (arg "stride" (Option.map (jstr {|{"as_ints":%s}|}) stride))
    (arg "padding" (Option.map (jstr {|{"as_ints":%s}|}) padding))
    (arg "dilation" (Option.map (jstr {|{"as_ints":%s}|}) dilation))
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
            params={kernel={h=2; w=2}; stride={h=2; w=2}; pad={h=0; w=0}}
        n2: [t3 f32 [H=3 W=4 C=4]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [H=3 W=4 C=4] <-n2] |}]

(* Three spellings of "stride defaults to the kernel": absent, an explicit empty
   list, and the value written out. PyTorch treats the first two the same, and
   an arm that normalized only one of them would accept a real export and refuse
   its equivalent. *)
let%expect_test "an omitted, empty and explicit stride agree" =
  show "absent:" (prog (pool ()));
  show "empty list:" (prog (pool ~stride:"[]" ()));
  show "explicit:" (prog (pool ~stride:"[2,2]" ()));
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
       (pool ~kernel:"[3,2]" ~stride:"[2,1]" ~padding:"[1,0]" ()));
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
            params={kernel={h=3; w=2}; stride={h=2; w=1}; pad={h=1; w=0}}
        n2: [t3 f32 [H=3 W=5 C=6]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [H=3 W=5 C=6] <-n2] |}]

let%expect_test "single-element lists are accepted as symmetric" =
  show "one-element:"
    (prog (pool ~kernel:"[2]" ~stride:"[2]" ~padding:"[0]" ()));
  [%expect {| one-element:               lowered, nodes=3 |}]

(* ---- rejections --------------------------------------------------------- *)

(* [Pool.MaxPool2d.params] has no field for either, so silently dropping them
   would compute a different op under the right name. Both spellings of the
   default are accepted; anything else is refused. *)
let%expect_test "dilation and ceil_mode are refused, not dropped" =
  show "dilation 1,1:" (prog (pool ~dilation:"[1,1]" ()));
  show "dilation 2,2:" (prog (pool ~dilation:"[2,2]" ()));
  show "dilation 1,2:" (prog (pool ~dilation:"[1,2]" ()));
  show "ceil_mode false:" (prog (pool ~ceil_mode:false ()));
  show "ceil_mode true:" (prog (pool ~ceil_mode:true ()));
  [%expect
    {|
    dilation 1,1:              lowered, nodes=3
    dilation 2,2:              malformed PT2 graph: torch.ops.aten.max_pool2d.default: dilation=[2,2] is not supported (only 1)
    dilation 1,2:              malformed PT2 graph: torch.ops.aten.max_pool2d.default: dilation=[1,2] is not supported (only 1)
    ceil_mode false:           lowered, nodes=3
    ceil_mode true:            malformed PT2 graph: torch.ops.aten.max_pool2d.default: ceil_mode=true is not supported |}]

let%expect_test "config faults are typed, not raised" =
  show "kernel 0:" (prog (pool ~kernel:"[0,0]" ()));
  show "stride 0:" (prog (pool ~stride:"[0,0]" ()));
  show "padding -1:" (prog (pool ~padding:"[-1,-1]" ()));
  show "kernel arity:" (prog (pool ~kernel:"[2,2,2]" ()));
  [%expect
    {|
    kernel 0:                  malformed PT2 graph: torch.ops.aten.max_pool2d.default: kernel_size must be positive, got 0
    stride 0:                  malformed PT2 graph: torch.ops.aten.max_pool2d.default: stride must be positive, got 0
    padding -1:                malformed PT2 graph: torch.ops.aten.max_pool2d.default: padding must not be negative, got -1
    kernel arity:              malformed PT2 graph: kernel_size must have one or two values, got 3 |}]
