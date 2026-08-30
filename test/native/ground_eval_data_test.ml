(* [Ground_eval]'s own resolver for an [Index.Data] source (round 10): it
   succeeds ONLY for a directly-bound constant, with no stage-walking
   fallback, and falls through to [`Data_index_unresolved] for anything else
   -- conservative rather than computing an answer that could disagree with
   real (F32-rounding) execution. Exercised through [Ground_eval.at], the
   only public entry point that reaches [resolve_data_source]. *)

open Graph_ir

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n
let zero = Expr.Coord.of_fn (fun _ -> Expr.Index.zero)
let origin = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0

let pp_result =
  Core.Pretty.err_result ~ok:Ground_expr.pp ~error:Ground_eval.pp_error

let index_id = Tensor_id.of_int 0
let self_id = Tensor_id.of_int 1
let out_id = Tensor_id.of_int 2

let index_sig =
  Tensor_sig.create ~id:index_id ~name:"" ~shape:(s1c 3)
    ~fmt:(Payload.Fmt Payload.I64) ()

let self_sig =
  Tensor_sig.create ~id:self_id ~name:"" ~shape:(s1c 3)
    ~fmt:(Payload.Fmt Payload.F32) ()

let out_sig =
  Tensor_sig.create ~id:out_id ~name:"" ~shape:(s1c 3)
    ~fmt:(Payload.Fmt Payload.F32) ()

(* Reads [self] at [C = index[0]] -- the same coordinate shape
   [Index_tensor.Compute.pixel] will build in Gate 3, without needing that
   op's Graph_ir node to exist yet: a stage's body is an ordinary
   [Expr.Value.t], independent of what op produced it. *)
let body =
  Expr.Value.load
    (Expr_bridge.source_of_id self_id)
    (Expr.Coord.set zero Axis.C
       (Expr.Index.data (Expr_bridge.source_of_id index_id) zero 3))

let out_stage = { Stage_program.Stage.id = out_id; sg = out_sig; body }

let program ~index_kind ~extra_inputs ~extra_stages =
  {
    Stage_program.inputs =
      (index_id, index_sig) :: (self_id, self_sig) :: extra_inputs;
    input_kinds =
      List.fold_left
        (fun m (id, kind) -> Tensor_id.Map.add id kind m)
        Tensor_id.Map.empty
        ((index_id, index_kind) :: (self_id, Input.Constant)
        :: List.map (fun (id, _) -> (id, Input.Constant)) extra_inputs);
    consts = [];
    stages = out_stage :: extra_stages;
    outputs = [ out_id ];
  }

let index_const values =
  Tensor.materialize_i64
    (s1c (List.length values))
    (fun c -> List.nth values (Dim.to_int (Vec6.get c Axis.C)))

let self_const values =
  Tensor.materialize
    (s1c (List.length values))
    (fun c -> List.nth values (Dim.to_int (Vec6.get c Axis.C)))

let%expect_test
    "resolve_data_source: a directly-bound I64 constant resolves exactly" =
  let p =
    program ~index_kind:Input.Constant ~extra_inputs:[] ~extra_stages:[]
  in
  let constants =
    Tensor_id.Map.empty
    |> Tensor_id.Map.add index_id (index_const [ 2L; 0L; 1L ])
    |> Tensor_id.Map.add self_id (self_const [ 10.; 20.; 30. ])
  in
  let env = Ground_eval.Env.of_program ~constants p ~side:`Src in
  (* self[index[0]] = self[2] = 30. *)
  Fmt.pr "%a@." pp_result (Ground_eval.at env out_id origin);
  [%expect {| f32(0x1.ep+4) |}]

let%expect_test
    "resolve_data_source: negative-index normalization matches \
     [resolve_gather_index]" =
  let p =
    program ~index_kind:Input.Constant ~extra_inputs:[] ~extra_stages:[]
  in
  let constants =
    Tensor_id.Map.empty
    |> Tensor_id.Map.add index_id (index_const [ -1L; 0L; 1L ])
    |> Tensor_id.Map.add self_id (self_const [ 10.; 20.; 30. ])
  in
  let env = Ground_eval.Env.of_program ~constants p ~side:`Src in
  (* self[index[0]] = self[-1] = self[2] = 30. *)
  Fmt.pr "%a@." pp_result (Ground_eval.at env out_id origin);
  [%expect {| f32(0x1.ep+4) |}]

let%expect_test
    "resolve_data_source: a Data source behind ANY intervening stage falls \
     through to Data_index_unresolved, never stage-walked" =
  (* [index_id] is now a STAGE -- an identity load from a further constant --
     rather than a directly-bound constant. Round 10: no stage-walking
     fallback, even for a value-preserving identity stage, because
     [Stage_program.ground] would round it through F32 on real execution. *)
  let raw_id = Tensor_id.of_int 3 in
  let raw_sig =
    Tensor_sig.create ~id:raw_id ~name:"" ~shape:(s1c 3)
      ~fmt:(Payload.Fmt Payload.I64) ()
  in
  let index_stage =
    {
      Stage_program.Stage.id = index_id;
      sg = index_sig;
      body = Expr.Value.load (Expr_bridge.source_of_id raw_id) zero;
    }
  in
  let p =
    program ~index_kind:Input.Constant
      ~extra_inputs:[ (raw_id, raw_sig) ]
      ~extra_stages:[ index_stage ]
  in
  let constants =
    Tensor_id.Map.empty
    |> Tensor_id.Map.add raw_id (index_const [ 2L; 0L; 1L ])
    |> Tensor_id.Map.add self_id (self_const [ 10.; 20.; 30. ])
  in
  let env = Ground_eval.Env.of_program ~constants p ~side:`Src in
  Fmt.pr "%a@." pp_result (Ground_eval.at env out_id origin);
  [%expect
    {| Data index source could not be resolved to a directly-bound I64 constant |}]

let%expect_test
    "resolve_data_source: an unbound Data source also falls through to \
     Data_index_unresolved" =
  let p = program ~index_kind:Input.Input ~extra_inputs:[] ~extra_stages:[] in
  let constants =
    Tensor_id.Map.empty
    |> Tensor_id.Map.add self_id (self_const [ 10.; 20.; 30. ])
  in
  let env = Ground_eval.Env.of_program ~constants p ~side:`Src in
  Fmt.pr "%a@." pp_result (Ground_eval.at env out_id origin);
  [%expect
    {| Data index source could not be resolved to a directly-bound I64 constant |}]
