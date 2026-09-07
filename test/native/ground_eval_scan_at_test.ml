(* An inline [Value.Scan_at] -- a raw scan descriptor appearing directly as a
   Pixel stage's own body, never wrapped through a Region local -- is the
   OTHER scan shape grounding must support besides a Region-authored trace
   local; see [Ground_eval.ground_scan_at].
   [test/native/fusion_test.ml] shows this shape is real: a bare
   [Expr.Builder.scan] result loaded as a Pixel body, with no Region local at
   all. *)

open Expr

let limits =
  Err.or_raise ~pp_error:Scan_limits.pp_error
    (Scan_limits.create ~max_state:100 ~max_updates:1000L)

let pos n = Index.clamp_low (Index.const n)

let counter ~width ~steps =
  Err.or_raise ~pp_error:Scan.pp_error
    (Builder.run
       (Builder.scan ~limits ~width ~steps
          ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
          ~update:(fun ~step:_ ~lane ~previous_at ->
            Builder.return (Value.add (previous_at lane) (Value.const 1.)))))

let out_sig =
  Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:""
    ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
    ~fmt:(Payload.Fmt Payload.F32) ()

let ground_value pixel_expr =
  let program = Region_program.pixel pixel_expr in
  let stage_program : Stage_program.t =
    {
      Stage_program.inputs = [];
      input_kinds = Tensor_id.Map.empty;
      consts = [];
      stages =
        [
          {
            Stage_program.Stage.id = Tensor_id.of_int 0;
            sg = out_sig;
            computation = program;
          };
        ];
      outputs = [ Tensor_id.of_int 0 ];
    }
  in
  let ground_env = Ground_eval.Env.of_program stage_program ~side:`Src in
  let meter = Ground_eval.Meter.create Ground_eval.default_budget in
  let term =
    Err.or_raise ~pp_error:Ground_eval.pp_error
      (Ground_eval.at ~meter ground_env (Tensor_id.of_int 0) Vec6.origin)
  in
  Ground_expr.eval
    (Ground_eval.Term.expression term)
    Ground_expr.Valuation.empty

let reference_value pixel_expr =
  let env =
    {
      Eval.Env.load = (fun _ _ -> assert false);
      load_index = (fun _ _ -> assert false);
    }
  in
  let scan_meter = Scan_meter.create ~limits in
  Err.or_raise ~pp_error:Eval.pp_error
    (Eval.value env ~scan_meter ~output:(Coord.of_fn (fun _ -> 0)) pixel_expr)

let%expect_test
    "grounding an inline Scan_at agrees with Expr.Eval, at row 0, mid-row and \
     the final row" =
  let s = counter ~width:2 ~steps:5 in
  List.iter
    (fun row ->
      let pixel_expr = Value.scan_at s ~row:(pos row) ~lane:(pos 1) in
      let g = ground_value pixel_expr and r = reference_value pixel_expr in
      Fmt.pr "row=%d grounded=%g reference=%g agree=%b@." row g r
        (Float.equal g r))
    [ 0; 1; 3; 5 ];
  [%expect
    {|
    row=0 grounded=0 reference=0 agree=true
    row=1 grounded=1 reference=1 agree=true
    row=3 grounded=3 reference=3 agree=true
    row=5 grounded=5 reference=5 agree=true |}]

(* Zero steps is meaningful: no update ever runs, so [row] must be 0 and the
   value is exactly [init]'s. *)
let%expect_test "grounding an inline Scan_at with zero steps" =
  let s = counter ~width:1 ~steps:0 in
  let pixel_expr = Value.scan_at s ~row:Index.zero ~lane:Index.zero in
  let g = ground_value pixel_expr and r = reference_value pixel_expr in
  Fmt.pr "grounded=%g reference=%g agree=%b@." g r (Float.equal g r);
  [%expect {| grounded=0 reference=0 agree=true |}]
