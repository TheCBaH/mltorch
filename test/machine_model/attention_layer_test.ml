(* A granted attention layer: an [Attention_layer] node selected as one
   [Attention_layer_block], its invocation the layer's sequence of existing
   operations, expanded before the plan and run through every Machine IR route
   against [Attention_layer_block.eval], bit for bit, and within tolerance of
   the node's binary64 reference. Its own file for the inline-test time
   ceiling. *)

open Graph_ir
open Model_test
module M = Machine_model.Mir_model
module Layer_select = Machine_model.Layer_select
module Layer_site = Machine_model.Layer_site
module Sdpa_select = Machine_model.Sdpa_select
module LF = Native_test.Attention_layer_fixtures
module L = Attention_layer.Attention_layer
module AL = Attention_layer_block
module A = Attention_block

let fp32_contracted =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_contracted;
      target = Ssa_ir.Ssa_target.scalar;
    }

let caps = { Contraction.Select.Capabilities.matrix = false; vector = false }
let grant precision = Some { Sdpa_select.Grant.block_keys = 4; precision }

let layer_node (g : graph) =
  (List.find
     (fun (n : node) ->
       match n.Node.op with Attention_layer _ -> true | _ -> false)
     (nodes g))
    .Node.id

let flat (Tensor.Tensor t as p) =
  let a = Array.make (Vec6.numel t.Tensor.shape :> int) 0. in
  Vec6.iter t.Tensor.shape (fun c ->
      a.((Vec6.offset t.Tensor.shape c :> int)) <- Tensor.read p c);
  a

(* The granted reference over the graph's own inputs, by the site's roles. *)
let layer_oracle (site : Layer_site.t) =
  Custom
    (fun (g : graph) _constants inputs ->
      let r = site.Layer_site.roles in
      let reader = function
        | None -> fun _ -> invalid_arg "the reference read an absent operand"
        | Some id ->
            let a = flat (List.assoc id inputs) in
            fun o -> a.(Int64.to_int o)
      in
      let out_sig =
        Tensor_id.Map.find r.Layer_site.Roles.output g.Graph.tensors
      in
      let shape = out_sig.Tensor_sig.shape in
      let out = Array.make (Vec6.numel shape :> int) 0. in
      AL.eval site.Layer_site.block
        {
          AL.Readers.x = reader (Some r.Layer_site.Roles.x);
          norm_weight = reader r.Layer_site.Roles.norm_weight;
          norm_bias = reader r.Layer_site.Roles.norm_bias;
          wq = reader (Some r.Layer_site.Roles.wq);
          bq = reader r.Layer_site.Roles.bq;
          wk = reader (Some r.Layer_site.Roles.wk);
          bk = reader r.Layer_site.Roles.bk;
          wv = reader (Some r.Layer_site.Roles.wv);
          bv = reader r.Layer_site.Roles.bv;
          wo = reader (Some r.Layer_site.Roles.wo);
          bo = reader r.Layer_site.Roles.bo;
          gamma = reader r.Layer_site.Roles.gamma;
          cos = reader r.Layer_site.Roles.cos;
          sin = reader r.Layer_site.Roles.sin;
          mask = reader r.Layer_site.Roles.mask;
        }
        ~write:(fun o v -> out.(Int64.to_int o) <- v);
      Tensor_id.Map.singleton r.Layer_site.Roles.output
        (Tensor.materialize shape (fun c -> out.((Vec6.offset shape c :> int)))))

(* Within [1e-4] absolute and relative of the binary64 reference: the granted
   layer rounds every step to binary32, streams its keys and takes one binary32
   scale where the Native Sdpa splits a square root. *)
let close a b =
  let x = flat a and y = flat b in
  Array.length x = Array.length y
  && Array.for_all2
       (fun u v -> Float.abs (u -. v) <= 1e-4 +. (1e-4 *. Float.abs v))
       x y

let configs =
  let base = LF.small in
  [
    ("bare", base);
    ( "pre layer, all flags, masked",
      LF.with_norm L.Norm.Kind.Layer L.Placement.Pre
        { (LF.all_flags base) with LF.Config.mask = true } );
    ( "post rms, rope",
      LF.with_norm L.Norm.Kind.Rms L.Placement.Post
        { (LF.all_flags base) with LF.Config.rope = LF.Rope_form.Shared } );
  ]

let routes =
  [
    ("generic", None);
    ("aarch64", Some (M.Route.Aarch64 M.Stage.Scanned));
    ("x86_64", Some (M.Route.X86_64 M.Stage.Scanned));
  ]

let site_of ~precision g =
  match Layer_select.choose ~grant:(grant precision) caps g (layer_node g) with
  | Ok site -> site
  | Error r -> Fmt.failwith "not selected: %a" Layer_select.Reason.pp r

let run ?route ~precision ~pipeline ~oracle ~same name c =
  let g = LF.node c in
  let site = site_of ~precision g in
  let layer id = if Node_id.equal id (layer_node g) then Some site else None in
  let values = LF.inputs g in
  check ?route ~layer ~pipeline ~oracle:(oracle site) ~same ~calls:1
    ~input:(fun ~salt:_ _ (sg : Tensor_sig.t) ->
      List.assoc sg.Tensor_sig.id values)
    name
    (fun () -> g)

let%expect_test "a granted layer through every route is its descriptor" =
  List.iter
    (fun (precision, pipeline) ->
      List.iter
        (fun (name, c) ->
          List.iter
            (fun (rname, route) ->
              run ?route ~precision ~pipeline ~oracle:layer_oracle
                ~same:same_portable
                (Fmt.str "%s, %s, %s" (A.Precision.name precision) name rname)
                c)
            routes)
        configs)
    [
      (A.Precision.F32_ordered, fp32_ordered);
      (A.Precision.F32_fused, fp32_contracted);
    ];
  [%expect
    {|
    f32_ordered, bare, generic (1 invocations): bitwise
    f32_ordered, bare, aarch64 (1 invocations): bitwise
    f32_ordered, bare, x86_64 (1 invocations): bitwise
    f32_ordered, pre layer, all flags, masked, generic (1 invocations): bitwise
    f32_ordered, pre layer, all flags, masked, aarch64 (1 invocations): bitwise
    f32_ordered, pre layer, all flags, masked, x86_64 (1 invocations): bitwise
    f32_ordered, post rms, rope, generic (1 invocations): bitwise
    f32_ordered, post rms, rope, aarch64 (1 invocations): bitwise
    f32_ordered, post rms, rope, x86_64 (1 invocations): bitwise
    f32_fused, bare, generic (1 invocations): bitwise
    f32_fused, bare, aarch64 (1 invocations): bitwise
    f32_fused, bare, x86_64 (1 invocations): bitwise
    f32_fused, pre layer, all flags, masked, generic (1 invocations): bitwise
    f32_fused, pre layer, all flags, masked, aarch64 (1 invocations): bitwise
    f32_fused, pre layer, all flags, masked, x86_64 (1 invocations): bitwise
    f32_fused, post rms, rope, generic (1 invocations): bitwise
    f32_fused, post rms, rope, aarch64 (1 invocations): bitwise
    f32_fused, post rms, rope, x86_64 (1 invocations): bitwise |}]

(* The largest deviation of the descriptor from the binary64 reference, which
   the route above equals bit for bit: what the tolerance admits. *)
let deviation ~precision c =
  let g = LF.node c in
  let site = site_of ~precision g in
  let inputs = LF.inputs g in
  let want =
    Err.or_raise ~pp_error:Eval_direct.pp_error (Eval_direct.run g ~inputs)
  in
  let out = List.hd g.Graph.outputs in
  let got =
    match layer_oracle site with
    | Custom f -> Tensor_id.Map.find out (f g [] inputs)
    | _ -> assert false
  in
  let a = flat got and b = flat (Tensor_id.Map.find out want) in
  let m = ref 0. in
  Array.iteri (fun i u -> m := Float.max !m (Float.abs (u -. b.(i)))) a;
  !m

let%expect_test "a granted layer is within tolerance of the node's reference" =
  List.iter
    (fun (precision, pipeline) ->
      List.iter
        (fun (name, c) ->
          let name = Fmt.str "%s, %s" (A.Precision.name precision) name in
          Fmt.pr "%s: largest deviation %.1e@." name (deviation ~precision c);
          run ~precision ~pipeline
            ~oracle:(fun _ -> Reference_f64)
            ~same:close name c)
        configs)
    [
      (A.Precision.F32_ordered, fp32_ordered);
      (A.Precision.F32_fused, fp32_contracted);
    ];
  [%expect
    {|
    f32_ordered, bare: largest deviation 1.2e-07
    f32_ordered, bare (1 invocations): bitwise
    f32_ordered, pre layer, all flags, masked: largest deviation 9.5e-07
    f32_ordered, pre layer, all flags, masked (1 invocations): bitwise
    f32_ordered, post rms, rope: largest deviation 4.8e-07
    f32_ordered, post rms, rope (1 invocations): bitwise
    f32_fused, bare: largest deviation 1.2e-07
    f32_fused, bare (1 invocations): bitwise
    f32_fused, pre layer, all flags, masked: largest deviation 9.5e-07
    f32_fused, pre layer, all flags, masked (1 invocations): bitwise
    f32_fused, post rms, rope: largest deviation 4.8e-07
    f32_fused, post rms, rope (1 invocations): bitwise |}]

(* What keeps a layer off the granted path: a bfloat16 grant on a core without
   the matrix engine, no grant, a pipeline that computes in binary64, and a
   plan whose contraction is not the descriptor's. *)
let%expect_test "what keeps a layer off the granted path" =
  let g = LF.node LF.small in
  (match
     Layer_select.choose
       ~grant:(grant A.Precision.Bf16_paired)
       caps g (layer_node g)
   with
  | Ok _ -> Fmt.pr "bf16: selected@."
  | Error r -> Fmt.pr "bf16: %a@." Layer_select.Reason.pp r);
  (match Layer_select.choose ~grant:None caps g (layer_node g) with
  | Ok _ -> Fmt.pr "ungranted: selected@."
  | Error r -> Fmt.pr "ungranted: %a@." Layer_select.Reason.pp r);
  List.iter
    (fun (name, precision, pipeline) ->
      run ~precision ~pipeline ~oracle:layer_oracle ~same:same_portable name
        LF.small)
    [
      ("exact pipeline", A.Precision.F32_ordered, Ssa_backends.Pipeline.Exact);
      ( "fused on a plan that does not contract",
        A.Precision.F32_fused,
        fp32_ordered );
      ( "ordered on a plan that contracts",
        A.Precision.F32_ordered,
        fp32_contracted );
    ];
  [%expect
    {|
    bf16: the core has no matrix engine
    ungranted: bf16_attention is not granted
    exact pipeline: refused: invocation 0 (n0): a binary32 attention layer needs a planned binary32 pipeline
    fused on a plan that does not contract: refused: invocation 0 (n0): a fused binary32 attention layer needs a plan that contracts dots only
    ordered on a plan that contracts: refused: invocation 0 (n0): an ordered binary32 attention layer needs a plan that does not contract |}]
