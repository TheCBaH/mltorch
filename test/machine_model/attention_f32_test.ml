(* A binary32 attention: an [Sdpa] node granted as the streaming recurrence at
   binary32, expanded into ordinary operations before the plan, and run through
   every Machine IR route against [Attention_block.eval]. Its own file so the
   partition stays under the inline-test time ceiling. *)

open Graph_ir
open Model_test
module M = Machine_model.Mir_model
module Sdpa_select = Machine_model.Sdpa_select
module F = Native_test.Graph_fixtures
module A = Attention_block
module B = Attention_binding_test

let sdpa_f32 ~tq ~tk ~d ~masked () =
  F.build "sdpa_f32"
    Graph_builder.(
      let* q = input ~shape:(F.s 1 1 1 1 tq d) () in
      let* k = input ~shape:(F.s 1 1 1 1 tk d) () in
      let* v = input ~shape:(F.s 1 1 1 1 tk d) () in
      let params = { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default } in
      if masked then
        let* m = input ~shape:(F.s 1 1 1 1 tq tk) () in
        sdpa params ~query:q ~key:k ~value:v ~mask:m ()
      else sdpa params ~query:q ~key:k ~value:v ())

let grant precision = Some { Sdpa_select.Grant.block_keys = 8; precision }

let%expect_test "a binary32 attention through every route is the reference" =
  List.iter
    (fun precision ->
      List.iter
        (fun (name, g) ->
          let graph = g () in
          match
            Sdpa_select.choose ~grant:(grant precision)
              { Contraction.Select.Capabilities.matrix = false; vector = false }
              graph (B.sdpa_node graph)
          with
          | Error r ->
              Fmt.pr "%s: not selected: %a@." name Sdpa_select.Reason.pp r
          | Ok site ->
              let attention id =
                if Node_id.equal id (B.sdpa_node graph) then Some site else None
              in
              List.iter
                (fun (rname, route) ->
                  check ?route ~attention ~pipeline:fp32_ordered
                    ~oracle:(B.attention_oracle site) ~same:same_portable
                    ~calls:1
                    ~input:(fun ~salt i sg ->
                      if i = 3 then
                        Tensor.materialize sg.Tensor_sig.shape (fun c ->
                            let n =
                              (Vec6.offset sg.Tensor_sig.shape c :> int)
                            in
                            if (n + salt) mod 7 = 0 then neg_infinity
                            else float_of_int (n mod 5) /. 4.)
                      else values ~salt:(salt + (3 * i)) sg)
                    (Fmt.str "%s, %s, %s"
                       (A.Precision.name precision)
                       name rname)
                    g)
                [
                  ("generic", None);
                  ("aarch64", Some (M.Route.Aarch64 M.Stage.Scanned));
                  ("x86_64", Some (M.Route.X86_64 M.Stage.Scanned));
                ])
        [
          ("9x29 d8", fun () -> sdpa_f32 ~tq:9 ~tk:29 ~d:8 ~masked:false ());
          ( "5x20 d6 masked",
            fun () -> sdpa_f32 ~tq:5 ~tk:20 ~d:6 ~masked:true () );
        ])
    [ A.Precision.F32_ordered ];
  [%expect
    {|
    f32_ordered, 9x29 d8, generic (1 invocations): bitwise
    f32_ordered, 9x29 d8, aarch64 (1 invocations): bitwise
    f32_ordered, 9x29 d8, x86_64 (1 invocations): bitwise
    f32_ordered, 5x20 d6 masked, generic (1 invocations): bitwise
    f32_ordered, 5x20 d6 masked, aarch64 (1 invocations): bitwise
    f32_ordered, 5x20 d6 masked, x86_64 (1 invocations): bitwise |}]

(* A binary32 grant over bfloat16 operands is not selected; a selected one is
   refused on a pipeline that computes in binary64, where its expansion would be
   another arithmetic. *)
let%expect_test "what keeps a binary32 attention off the expansion" =
  let g = B.sdpa_ordered_fixture ~tq:9 ~tk:29 ~d:8 () in
  (match
     Sdpa_select.choose
       ~grant:(grant A.Precision.F32_ordered)
       B.engine g (B.sdpa_node g)
   with
  | Ok _ -> Fmt.pr "selected@."
  | Error r -> Fmt.pr "bf16 operands: %a@." Sdpa_select.Reason.pp r);
  let graph = sdpa_f32 ~tq:9 ~tk:29 ~d:8 ~masked:false () in
  let site =
    Result.get_ok
      (Sdpa_select.choose
         ~grant:(grant A.Precision.F32_ordered)
         B.engine graph (B.sdpa_node graph))
  in
  let attention id =
    if Node_id.equal id (B.sdpa_node graph) then Some site else None
  in
  check ~attention ~oracle:(B.attention_oracle site) ~same:same_portable
    ~calls:1 "exact pipeline" (fun () -> graph);
  [%expect
    {|
    bf16 operands: the query is not binary32
    exact pipeline: refused: invocation 0 (n0): a binary32 attention needs a planned binary32 pipeline |}]
