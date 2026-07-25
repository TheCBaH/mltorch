(* Shared graphs for the transformation tests. There was no fixture library
   before this — [s]/[s1c]/[conv_axis] are copy-pasted across five test files —
   so new tests build on these and the existing files are left alone.

   Every fixture is a plain [Graph_ir.graph] built with [Graph_builder], named
   after the shape it exercises rather than the pass that will consume it. *)

(* ---- shape helpers ------------------------------------------------------- *)

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n
let nhwc ~h ~w ~c = s 1 1 1 h w c

let conv_axis ~kernel ~stride ~pad : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad;
    pad_after = Op_config.Nonneg.of_int pad;
    dilation = Op_config.Pos.of_int 1;
  }

let conv_params ~in_channels : Conv.Conv2d.params =
  {
    h = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    w = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    in_channels = Dim.extent in_channels;
    groups = Op_config.Pos.of_int 1;
  }

let bn_params : Norm.BatchNorm.params = { channel = Axis.C; eps = 1e-5 }

(* Conv weights live as [Cout,1,1,Kh,Kw,Cin]: Cout on N, Cin on C. *)
let weight_shape ~out_channels ~in_channels = s out_channels 1 1 2 2 in_channels

(* ---- permutations -------------------------------------------------------- *)

let identity_perm = Axis.[ (N, N); (T, T); (D, D); (H, H); (W, W); (C, C) ]
let swap_hw = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, H); (C, C) ]
let swap_wc = Axis.[ (N, N); (T, T); (D, D); (H, H); (W, C); (C, W) ]

(* A 3-cycle and its inverse, so a two-permute chain composes to the identity. *)
let rotate_hwc = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]
let unrotate_hwc = Axis.[ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

(* ---- fixtures ------------------------------------------------------------ *)

let build name m =
  match Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m with
  | Ok g -> g
  | Error e ->
      invalid_arg
        (Format.asprintf "fixture %s: %a" name Graph_builder.pp_error
           e.Core.Error.kind)

(* conv -> batch_norm -> relu, with every parameter a constant. The shape
   batch-norm folding matches. *)
let chain () =
  build "chain"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:4 ~w:4 ~c:2) () in
      let* w =
        constant ~shape:(weight_shape ~out_channels:3 ~in_channels:2) ()
      in
      let* bias = constant ~shape:(s1c 3) () in
      let* gamma = constant ~shape:(s1c 3) () in
      let* beta = constant ~shape:(s1c 3) () in
      let* mean = constant ~shape:(s1c 3) () in
      let* var = constant ~shape:(s1c 3) () in
      let* y = conv2d (conv_params ~in_channels:2) ~x ~weight:w ~bias () in
      let* n =
        batch_norm bn_params ~x:y ~weight:gamma ~bias:beta ~running_mean:mean
          ~running_var:var ()
      in
      relu n)

(* One edge feeding two consumers that rejoin: the fan-out and union a pattern
   has to cope with, and the shape where an interior edge is NOT single-use. *)
let diamond () =
  build "diamond"
    Graph_builder.(
      let* x = input ~shape:(s1c 4) () in
      let* left = relu x in
      let* right = mul x x in
      add left right)

(* A skip connection: the add's two operands come from different depths. *)
let residual () =
  build "residual"
    Graph_builder.(
      let* x = input ~shape:(s1c 4) () in
      let* a = relu x in
      let* b = relu a in
      add x b)

(* A permute that does nothing — the trivial-trim case. *)
let permute_noop () =
  build "permute_noop"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* p = permute identity_perm x in
      relu p)

(* Two permutes composing to the identity: trimming has to look at the whole
   chain, since neither node is a no-op alone. *)
let permute_sequence () =
  build "permute_sequence"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute rotate_hwc x in
      let* b = permute unrotate_hwc a in
      relu b)

(* Two permutes that compose to something else, which chaining should fuse into
   one node rather than remove. *)
let permute_pair () =
  build "permute_pair"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute swap_hw x in
      let* b = permute swap_wc a in
      relu b)

(* A reshape that only relabels axes — the non-unit extents keep their order —
   so it is expressible as a permute. *)
let reshape_relabel () =
  build "reshape_relabel"
    Graph_builder.(
      let* x = input ~shape:(s 1 1 1 1 1 6) () in
      let* r = reshape { Reshape.Reshape.shape = s 1 1 1 6 1 1 } x in
      relu r)

(* THE motivating case: a constant weight behind a permute, re-permuted on every
   inference until constant folding hoists it to load time. *)
let const_permute () =
  build "const_permute"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:3 ~w:3 ~c:2) () in
      let* w = constant ~shape:(s 3 1 1 2 2 2) () in
      let* wp = permute identity_perm w in
      conv2d (conv_params ~in_channels:2) ~x ~weight:wp ())

(* A multi-node all-constant sub-DAG, for folding under a fixed point. *)
let const_arith () =
  build "const_arith"
    Graph_builder.(
      let* x = input ~shape:(s1c 3) () in
      let* a = constant ~shape:(s1c 3) () in
      let* b = constant ~shape:(s1c 3) () in
      let* c = constant ~shape:(s1c 3) () in
      let* ab = mul a b in
      let* abc = mul ab c in
      add x abc)

(* A multi-output op with its second result routed to a sink — the arity a
   matcher's anchoring rule and constant folding both have to handle. *)
let multi_output () =
  build "multi_output"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:4 ~w:4 ~c:2) () in
      let params : Pool.MaxPool2dWithIndices.params =
        {
          kernel = { h = Dim.extent 2; w = Dim.extent 2 };
          stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
          pad = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
        }
      in
      let* values, indices = max_pool2d_with_indices params x in
      let* () = discard indices in
      relu values)

(* Two sibling groups with a rewrite opportunity spanning both: the placement
   case, where the replacement belongs in their common ancestor. *)
let grouped () =
  build "grouped"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = group ~label:"first" (permute rotate_hwc x) in
      let* b = group ~label:"second" (permute unrotate_hwc a) in
      relu b)

let all =
  [
    ("chain", chain);
    ("const_arith", const_arith);
    ("const_permute", const_permute);
    ("diamond", diamond);
    ("grouped", grouped);
    ("multi_output", multi_output);
    ("permute_noop", permute_noop);
    ("permute_pair", permute_pair);
    ("permute_sequence", permute_sequence);
    ("reshape_relabel", reshape_relabel);
    ("residual", residual);
  ]
