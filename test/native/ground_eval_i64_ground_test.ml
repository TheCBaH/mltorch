(* [Ground_eval.at] has no grounded/fused representation for [I64_to_float]
   yet (only the plain evaluator, [Eval.value], supports it); it must reject
   explicitly rather than silently miscompiling
   or treating the cast as a no-op leaf. Mirrors [ground_eval_data_test.ml]'s
   minimal single-stage program shape. *)

open Graph_ir

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n
let origin = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0

let pp_result =
  Core.Pretty.err_result ~ok:Ground_expr.pp ~error:Ground_eval.pp_error

let at env id coord =
  Err.map Ground_eval.Term.expression
    (Ground_eval.at
       ~meter:(Ground_eval.Meter.create Ground_eval.default_budget)
       env id coord)

let out_id = Tensor_id.of_int 0

let out_sig =
  Tensor_sig.create ~id:out_id ~name:"" ~shape:(s1c 1)
    ~fmt:(Payload.Fmt Payload.F32) ()

let body = Expr.Value.i64_to_float (Expr.Value.i64_const 5L)

let out_stage =
  {
    Stage_program.Stage.id = out_id;
    sg = out_sig;
    computation = Region_group.Ref.Solo (Region_program.pixel body);
  }

let program =
  {
    Stage_program.inputs = [];
    input_kinds = Tensor_id.Map.empty;
    consts = [];
    stages = [ out_stage ];
    outputs = [ out_id ];
  }

let%expect_test "Ground_eval.at rejects I64_to_float outright" =
  let env = Ground_eval.Env.of_program program ~side:`Src in
  Fmt.pr "%a@." pp_result (at env out_id origin);
  [%expect
    {| I64_to_float has no grounded/fused representation yet (only the plain evaluator supports it) |}]
