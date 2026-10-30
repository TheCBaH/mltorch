(* The fused attention's binding: a granted [Sdpa] node resolved by role to
   the buffers of its invocation, run through generic Machine IR against the
   granted streaming reference, and the reasons a node keeps the generated
   path. *)

open Graph_ir
open Model_test
module M = Machine_model.Mir_model
module Attention_site = Machine_model.Attention_site
module Sdpa_select = Machine_model.Sdpa_select
module F = Native_test.Graph_fixtures

(* ---- the fused attention's binding ---------------------------------------- *)

(* [Sdpa] over bfloat16 operands created in the order K, V, Q (so the ids are
   not in role order), optionally with the query as the key too and an
   explicit or omitted mask. *)
let sdpa_ordered_fixture ~tq ~tk ~d ?(query_is_key = false) ?(masked = false) ()
    =
  let bf16 = Payload.Fmt Payload.BF16 in
  F.build "sdpa_ordered"
    Graph_builder.(
      let* k = input ~shape:(F.s 1 1 1 1 tk d) ~fmt:bf16 () in
      let* v = input ~shape:(F.s 1 1 1 1 tk d) ~fmt:bf16 () in
      let* q = input ~shape:(F.s 1 1 1 1 tq d) ~fmt:bf16 () in
      let* m =
        if masked then
          let* m = input ~shape:(F.s 1 1 1 1 tq tk) () in
          return (Some m)
        else return None
      in
      let params = { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default } in
      let key = if query_is_key then q else k in
      match m with
      | None -> sdpa params ~query:q ~key ~value:v ()
      | Some m -> sdpa params ~query:q ~key ~value:v ~mask:m ())

let engine = { Contraction.Select.Capabilities.matrix = true; vector = true }

let sdpa_node (g : graph) =
  (List.find
     (fun (n : node) -> match n.Node.op with Sdpa _ -> true | _ -> false)
     (nodes g))
    .Node.id

(* The granted reference over the graph's own inputs, by the roles of [site]. *)
let attention_oracle (site : Attention_site.t) =
  Custom
    (fun (g : graph) _constants inputs ->
      let flat id =
        let (Tensor.Tensor t as p) = List.assoc id inputs in
        let a = Array.make (Vec6.numel t.Tensor.shape :> int) 0. in
        Vec6.iter t.Tensor.shape (fun c ->
            a.((Vec6.offset t.Tensor.shape c :> int)) <- Tensor.read p c);
        a
      in
      let q = flat site.Attention_site.query
      and k = flat site.Attention_site.key
      and v = flat site.Attention_site.value in
      let mask = Option.map flat site.Attention_site.mask in
      let d = site.Attention_site.block in
      let out =
        Array.make
          (Int64.to_int (Int64.mul d.Attention_block.tq d.Attention_block.d))
          0.
      in
      Attention_block.eval d
        ~q:(fun o -> q.(Int64.to_int o))
        ~k:(fun o -> k.(Int64.to_int o))
        ~v:(fun o -> v.(Int64.to_int o))
        ~mask:(fun o -> (Option.get mask).(Int64.to_int o))
        ~write:(fun o x -> out.(Int64.to_int o) <- x);
      let id = site.Attention_site.output in
      let shape = (sig_of g id).Tensor_sig.shape in
      Tensor_id.Map.singleton id
        (Tensor.materialize shape (fun c -> out.((Vec6.offset shape c :> int)))))

let%expect_test "an attention selected by role, not by position" =
  let grant = Some Sdpa_select.Grant.default in
  List.iter
    (fun (name, g) ->
      let graph = g () in
      match Sdpa_select.choose ~grant engine graph (sdpa_node graph) with
      | Error r -> Fmt.pr "%s: not selected: %a@." name Sdpa_select.Reason.pp r
      | Ok site ->
          let attention id =
            if Node_id.equal id (sdpa_node graph) then Some site else None
          in
          (* the roles' tensors, and the buffer each resolved to *)
          let b = bundle graph in
          List.iter
            (fun (inv : Loop_ir.Loop_bundle.invocation) ->
              match Attention_site.buffers site inv with
              | Ok bs -> Fmt.pr "%s: %a@." name Attention_site.Buffers.pp bs
              | Error u -> Fmt.pr "%s: %a@." name Attention_site.Unbound.pp u)
            b.Loop_ir.Loop_bundle.invocations;
          check ~attention ~oracle:(attention_oracle site) ~same:same_portable
            ~calls:1
            ~input:(fun ~salt i sg ->
              in_format sg
                (if i = 3 then
                   Tensor.materialize sg.Tensor_sig.shape (fun c ->
                       let n = (Vec6.offset sg.Tensor_sig.shape c :> int) in
                       if (n + salt) mod 7 = 0 then neg_infinity
                       else float_of_int (n mod 5) /. 4.)
                 else values ~salt:(salt + (3 * i)) sg))
            name g)
    [
      ( "K, V, Q created in that order",
        fun () -> sdpa_ordered_fixture ~tq:9 ~tk:29 ~d:8 () );
      ( "query is the key",
        fun () -> sdpa_ordered_fixture ~tq:16 ~tk:16 ~d:8 ~query_is_key:true ()
      );
      ( "explicit mask",
        fun () -> sdpa_ordered_fixture ~tq:9 ~tk:29 ~d:8 ~masked:true () );
    ];
  [%expect
    {|
    K, V, Q created in that order: q=b2 k=b0 v=b1 mask=none o=b8
    K, V, Q created in that order (1 invocations): bitwise
    query is the key: q=b2 k=b2 v=b1 mask=none o=b8
    query is the key (1 invocations): bitwise
    explicit mask: q=b2 k=b0 v=b1 mask=b3 o=b4
    explicit mask (1 invocations): bitwise |}]

(* Binding by position would take the first input for the query: the
   reordered case must turn red under it. *)
let%expect_test "binding by buffer position is caught" =
  let g = sdpa_ordered_fixture ~tq:16 ~tk:16 ~d:8 () in
  let node = sdpa_node g in
  let site =
    Result.get_ok
      (Sdpa_select.choose ~grant:(Some Sdpa_select.Grant.default) engine g node)
  in
  let first, second, third =
    match g.Graph.inputs with [ a; b; c ] -> (a, b, c) | _ -> assert false
  in
  let wrong =
    { site with Attention_site.query = first; key = second; value = third }
  in
  let attention id = if Node_id.equal id node then Some wrong else None in
  check ~attention ~oracle:(attention_oracle site) ~same:same_portable ~calls:1
    ~input:(fun ~salt i sg -> in_format sg (values ~salt:(salt + (3 * i)) sg))
    "by position"
    (fun () -> g);
  [%expect {| by position (1 invocations): DIFFERS |}]

let%expect_test "every reason a node keeps the generated path" =
  let engine_off =
    { engine with Contraction.Select.Capabilities.matrix = false }
  in
  let grant = Some Sdpa_select.Grant.default in
  let show name ?(grant = grant) ?(caps = engine) g =
    let graph = g () in
    let node =
      try sdpa_node graph with Not_found -> (List.hd (nodes graph)).Node.id
    in
    match Sdpa_select.choose ~grant caps graph node with
    | Ok _ -> Fmt.pr "%s: selected@." name
    | Error r -> Fmt.pr "%s: %a@." name Sdpa_select.Reason.pp r
  in
  let bf16 ?mask ~tq ~tk ~d () = sdpa_bf16_fixture ~tq ~tk ~d ?mask () in
  show "granted" (bf16 ~tq:32 ~tk:32 ~d:32);
  show "ungranted" ~grant:None (bf16 ~tq:32 ~tk:32 ~d:32);
  show "no engine" ~caps:engine_off (bf16 ~tq:32 ~tk:32 ~d:32);
  show "odd head dimension" (bf16 ~tq:32 ~tk:32 ~d:31);
  show "head dimension 128" (bf16 ~tq:32 ~tk:32 ~d:128);
  show "block of 24"
    ~grant:(Some { Sdpa_select.Grant.block_keys = 24 })
    (bf16 ~tq:32 ~tk:32 ~d:32);
  show "1x1" (bf16 ~tq:1 ~tk:1 ~d:32);
  show "9x29 masked" (bf16 ~mask:(9, 29) ~tq:9 ~tk:29 ~d:64);
  show "f32 operands" (fun () ->
      F.build "sdpa_f32"
        Graph_builder.(
          let* q = input ~shape:(F.s 1 1 1 1 32 32) () in
          let* k = input ~shape:(F.s 1 1 1 1 32 32) () in
          let* v = input ~shape:(F.s 1 1 1 1 32 32) () in
          sdpa
            { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
            ~query:q ~key:k ~value:v ()));
  show "outer axis" (fun () ->
      let bf16 = Payload.Fmt Payload.BF16 in
      F.build "sdpa_outer"
        Graph_builder.(
          let* q = input ~shape:(F.s 2 1 1 1 32 32) ~fmt:bf16 () in
          let* k = input ~shape:(F.s 2 1 1 1 32 32) ~fmt:bf16 () in
          let* v = input ~shape:(F.s 2 1 1 1 32 32) ~fmt:bf16 () in
          sdpa
            { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
            ~query:q ~key:k ~value:v ()));
  show "not an attention" (fun () ->
      F.build "softmax"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 1 1 4 5) () in
          softmax { Reduce.Softmax.axis = Axis.C } x));
  [%expect
    {|
    granted: selected
    ungranted: bf16_attention is not granted
    no engine: the core has no matrix engine
    odd head dimension: a head dimension of 31, odd or past 64
    head dimension 128: a head dimension of 128, odd or past 64
    block of 24: a block of 24 keys, neither at most 16 nor 32
    1x1: 512 products for 1, more than twice the work
    9x29 masked: selected
    f32 operands: the query is not bfloat16
    outer axis: an extent past 1 on the N or T axis
    not an attention: not an attention |}]
