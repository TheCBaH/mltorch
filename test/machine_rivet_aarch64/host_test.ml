(* Whole bundles through the table-bound host: the tensors are the context's
   own storage, passed to the code by address, and every call is compared with
   the reference evaluator. *)

module F = Native_test.Graph_fixtures
module T = Machine_model_test.Model_test
module H = Machine_rivet_aarch64.Rivet_a64_host
module Mm = Machine_model.Mir_model
module Rt = Machine_rivet_aarch64.Rivet_a64_route
module Rn = Machine_rivet_aarch64.Rivet_a64_runtime
open Graph_ir

let check ?(allocation = Rt.Allocation.Reference)
    ?(runtime = Rn.Dependency_free) ?(pipeline = Ssa_backends.Pipeline.Exact)
    ?(calls = 2) ?constant ?(input = fun ~salt _ -> T.values ~salt) ?poison name
    g =
  let g = g () in
  let b = T.bundle g in
  let constant = Option.value constant ~default:(T.values ~salt:3) in
  let constants =
    List.map
      (fun id -> (id, constant (T.sig_of g id)))
      b.Loop_ir.Loop_bundle.constants
  in
  match H.prepare ~allocation ~runtime ~pipeline b with
  | Error rs ->
      Fmt.pr "%s: refused: %a@." name
        Fmt.(list ~sep:(any "; ") Mm.Refusal.pp)
        rs
  | Ok host -> (
      match
        H.Context.create host ~constants:(fun id -> List.assoc_opt id constants)
      with
      | Error s -> Fmt.pr "%s: %a@." name Mm.Stop.pp s
      | Ok cx ->
          let verdicts =
            List.init calls (fun call ->
                let inputs =
                  List.mapi
                    (fun i id -> (id, input ~salt:(call + i) i (T.sig_of g id)))
                    b.Loop_ir.Loop_bundle.inputs
                in
                let reference =
                  Err.or_raise ~pp_error:Eval_direct.pp_error
                    (Eval_direct.run g ~constants ~inputs)
                in
                match
                  H.Context.run ?poison cx ~inputs:(fun id ->
                      List.assoc_opt id inputs)
                with
                | Error s -> Fmt.str "%a" Mm.Stop.pp s
                | Ok outs ->
                    if
                      List.for_all2
                        (fun id t ->
                          T.bits t = T.bits (Tensor_id.Map.find id reference))
                        g.Graph.outputs outs
                    then "bitwise"
                    else "DIFFERS")
          in
          Fmt.pr "%s (%d invocations): %s@." name (H.invocations host)
            (String.concat ", " verdicts))

let%expect_test "conv, batch norm, relu on the table-bound host" =
  check ~constant:T.positive ~calls:3 "chain" F.chain;
  check ~constant:T.positive "wide chain" T.wide_chain;
  [%expect
    {|
    chain (3 invocations): bitwise, bitwise, bitwise
    wide chain (3 invocations): bitwise, bitwise |}]

let libm = Rn.System_libm

let%expect_test "Region nodes on the host: locals, scans and the meter" =
  check ~runtime:libm "softmax over C" (fun () ->
      F.build "softmax"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
          softmax { Reduce.Softmax.axis = Axis.C } x));
  check "layer_norm over W, C" (fun () ->
      F.build "layer_norm"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 1 2 4 5) () in
          layer_norm
            { Norm.LayerNorm.dims = [ Axis.W; Axis.C ]; eps = 1e-5 }
            ~x ()));
  check "rms_norm over C" (fun () ->
      F.build "rms_norm"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 1 3 4 5) () in
          rms_norm { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 } ~x ()));
  check ~runtime:libm "sdpa, masked rows"
    (fun () ->
      F.build "sdpa_mask"
        Graph_builder.(
          let* q = input ~shape:(F.s 1 1 2 3 4 5) () in
          let* k = input ~shape:(F.s 1 1 2 3 6 5) () in
          let* v = input ~shape:(F.s 1 1 2 3 6 5) () in
          let* m = input ~shape:(F.s 1 1 2 3 4 6) () in
          sdpa
            { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
            ~query:q ~key:k ~value:v ~mask:m ()))
    ~input:(fun ~salt i sg ->
      if i = 3 then
        Tensor.materialize sg.Tensor_sig.shape (fun c ->
            if
              (Vec6.get c Axis.W :> int) = 1
              || (salt mod 2 = 1 && (Vec6.get c Axis.C :> int) = 0)
            then neg_infinity
            else 0.)
      else T.values ~salt sg);
  [%expect
    {|
    softmax over C (1 invocations): bitwise, bitwise
    layer_norm over W, C (1 invocations): bitwise, bitwise
    rms_norm over C (1 invocations): bitwise, bitwise
    sdpa, masked rows (1 invocations): bitwise, bitwise |}]

(* Scratch is the context's, and a kernel must not read what it has not
   written: a pattern in every byte of it before each call changes nothing. *)
let%expect_test "poisoned scratch and the scanned allocation" =
  check ~poison:true ~constant:T.positive ~calls:3 "chain poisoned" F.chain;
  check ~allocation:Rt.Allocation.Scanned ~poison:true ~runtime:libm
    ~constant:T.positive ~calls:3 "chain scanned" F.chain;
  check ~allocation:Rt.Allocation.Scanned ~poison:true ~runtime:libm
    "softmax scanned" (fun () ->
      F.build "softmax"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
          softmax { Reduce.Softmax.axis = Axis.C } x));
  [%expect
    {|
    chain poisoned (3 invocations): bitwise, bitwise, bitwise
    chain scanned (3 invocations): bitwise, bitwise, bitwise
    softmax scanned (1 invocations): bitwise, bitwise |}]

let%expect_test "dependency-free runs exp on the owned code" =
  check "softmax over C" (fun () ->
      F.build "softmax"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
          softmax { Reduce.Softmax.axis = Axis.C } x));
  [%expect {| softmax over C (1 invocations): bitwise, bitwise |}]

(* Two contexts of one host share the loaded code and nothing else. *)
let%expect_test "independent contexts, and a comparison that can fail" =
  let g = F.chain () in
  let b = T.bundle g in
  let host =
    Result.get_ok (H.prepare ~pipeline:Ssa_backends.Pipeline.Exact b)
  in
  let constants salt =
    List.map
      (fun id -> (id, T.values ~salt (T.sig_of g id)))
      b.Loop_ir.Loop_bundle.constants
  in
  let inputs salt =
    List.map
      (fun id -> (id, T.values ~salt (T.sig_of g id)))
      b.Loop_ir.Loop_bundle.inputs
  in
  let context salt =
    Result.get_ok
      (H.Context.create host ~constants:(fun id ->
           List.assoc_opt id (constants salt)))
  in
  let a = context 3 and c = context 5 in
  let run cx ~cs ~is =
    let reference =
      Err.or_raise ~pp_error:Eval_direct.pp_error
        (Eval_direct.run g ~constants:(constants cs) ~inputs:(inputs is))
    in
    match
      H.Context.run cx ~inputs:(fun id -> List.assoc_opt id (inputs is))
    with
    | Ok [ t ] ->
        if
          T.bits t
          = T.bits (Tensor_id.Map.find (List.hd g.Graph.outputs) reference)
        then "bitwise"
        else "DIFFERS"
    | Ok _ -> "outputs?"
    | Error s -> Fmt.str "%a" Mm.Stop.pp s
  in
  Fmt.pr "a: %s@." (run a ~cs:3 ~is:1);
  Fmt.pr "c: %s@." (run c ~cs:5 ~is:2);
  Fmt.pr "a again, other inputs: %s@." (run a ~cs:3 ~is:7);
  Fmt.pr "c against a's constants: %s@." (run c ~cs:3 ~is:2);
  [%expect
    {|
    a: bitwise
    c: bitwise
    a again, other inputs: bitwise
    c against a's constants: DIFFERS |}]

(* The first failing invocation, and the context goes on. *)
let%expect_test "a failing invocation names its record, then the next call runs"
    =
  let g = T.gather () in
  let b = T.bundle g in
  let host =
    Result.get_ok (H.prepare ~pipeline:Ssa_backends.Pipeline.Exact b)
  in
  let cx = Result.get_ok (H.Context.create host ~constants:(fun _ -> None)) in
  let self_id, index_id =
    match b.Loop_ir.Loop_bundle.inputs with
    | [ s; i ] -> (s, i)
    | _ -> assert false
  in
  let call idx =
    let inputs =
      [
        (self_id, T.values ~salt:1 (T.sig_of g self_id));
        ( index_id,
          Tensor.materialize_i64 (F.s 1 1 1 1 1 2) (fun c ->
              List.nth idx (Vec6.get c Axis.C :> int)) );
      ]
    in
    Fmt.pr "index [%s]: %s@."
      (String.concat "; " (List.map Int64.to_string idx))
      (match H.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs) with
      | Ok _ -> "succeeds"
      | Error s -> Fmt.str "%a" Mm.Stop.pp s)
  in
  call [ 2L; -3L ];
  call [ 0L; 3L ];
  call [ -4L; 1L ];
  call [ 1L; 1L ];
  [%expect
    {|
    index [2; -3]: succeeds
    index [0; 3]: invocation 1 (n1): failure gather_index_out_of_range(3:i64, 3:i64)
    index [-4; 1]: invocation 1 (n1): failure gather_index_out_of_range(-4:i64, 3:i64)
    index [1; 1]: succeeds |}]
