(* [Ground_eval]'s own [Budget]/[Meter]/[Term] contract, independent of the
   whole [Map_verify] pipeline -- see .ai/native_scan_design.md's "Grounding
   meter and verdict mapping". [max_ground_nodes] is cumulative construction
   fuel, charged as [ground]/[leaf]/[max_pool] build each node; [max_nodes] is
   the CURRENT total logical size of every root registered against one
   [Meter.t], shared by every root sharing it. *)

open Graph_ir

let f32 = Payload.Fmt Payload.F32
let s1c = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
let sg id = Tensor_sig.create ~id ~name:"" ~shape:s1c ~fmt:f32 ()
let origin = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0
let zero = Expr.Coord.of_fn (fun _ -> Expr.Index.zero)

let stage id body : Stage_program.Stage.t =
  {
    Stage_program.Stage.id;
    sg = sg id;
    computation = Region_program.pixel body;
  }

let program stages : Stage_program.t =
  {
    Stage_program.inputs = [];
    input_kinds = Tensor_id.Map.empty;
    consts = [];
    stages;
    outputs =
      List.map
        (fun (s : Stage_program.Stage.t) -> s.Stage_program.Stage.id)
        stages;
  }

let pp_result =
  Core.Pretty.err_result ~ok:Ground_expr.pp ~error:Ground_eval.pp_error

let load id = Expr.Value.load (Expr_bridge.source_of_id id) zero

let budget ~max_ground_nodes ~max_nodes =
  { Ground_eval.Budget.max_ground_nodes; max_nodes }

(* ---- construction fuel over a single root --------------------------------- *)

let%expect_test
    "at charges construction fuel node by node and rejects an oversized single \
     root" =
  (* An unrolled [Reduce] of extent 6: one [Const] and one combining [Binary]
     charged per iteration, plus the seed -- 13 nodes by hand, small enough to
     check exactly, and independent of any real op's actual cost. *)
  let a_id = Tensor_id.of_int 0 in
  let body =
    Expr.Builder.run
      (Expr.Builder.reduction ~kind:Expr.Reduction.Sum ~lo:Expr.Index.zero
         ~hi:(Expr.Index.const 6) (fun _pos ->
           Expr.Builder.return (Expr.Value.const 1.)))
  in
  let p = program [ stage a_id body ] in
  let env = Ground_eval.Env.of_program p ~side:`Src in
  let at ~max_ground_nodes =
    let meter =
      Ground_eval.Meter.create (budget ~max_ground_nodes ~max_nodes:1_000)
    in
    Result.map Ground_eval.Term.expression
      (Ground_eval.at ~meter env a_id origin)
  in
  Fmt.pr "generous: %a@.tight: %a@." pp_result
    (at ~max_ground_nodes:1_000L)
    pp_result (at ~max_ground_nodes:5L);
  [%expect
    {|
    generous: f32(((((((0x0p+0 + 0x1p+0) + 0x1p+0) + 0x1p+0) + 0x1p+0) + 0x1p+0) + 0x1p+0))
    tight: grounding exceeds max_ground_nodes (5) |}]

(* ---- pair total shared across every root registered against one meter ----- *)

let%expect_test "at charges every registered root against the SAME pair total" =
  (* Each stage's own grounded root is [Round (Const _)], size 2 -- [at]
     always wraps a stage's body in [Round]. The first root alone exactly
     fits [max_nodes]; a second one of the same size cannot join it. *)
  let a_id = Tensor_id.of_int 0 and b_id = Tensor_id.of_int 1 in
  let p =
    program
      [ stage a_id (Expr.Value.const 1.); stage b_id (Expr.Value.const 2.) ]
  in
  let env = Ground_eval.Env.of_program p ~side:`Src in
  let meter =
    Ground_eval.Meter.create (budget ~max_ground_nodes:1_000L ~max_nodes:2)
  in
  let first =
    Result.map Ground_eval.Term.expression
      (Ground_eval.at ~meter env a_id origin)
  in
  let second =
    Result.map Ground_eval.Term.expression
      (Ground_eval.at ~meter env b_id origin)
  in
  Fmt.pr "first: %a@.second: %a@." pp_result first pp_result second;
  [%expect
    {|
    first: f32(0x1p+0)
    second: grounding exceeds max_nodes (2) |}]

(* ---- expand: replacement deltas against the pair account ------------------ *)

let%expect_test
    "expand skips a replacement that would cross max_nodes, and performs it \
     once the cap allows exactly that" =
  let a_id = Tensor_id.of_int 0 and b_id = Tensor_id.of_int 1 in
  let p =
    program [ stage a_id (Expr.Value.const 1.); stage b_id (load a_id) ]
  in
  let env = Ground_eval.Env.of_program p ~side:`Src in
  let no_boundary (_ : Ground_expr.Origin.t) = None in
  let run ~max_nodes =
    let meter =
      Ground_eval.Meter.create (budget ~max_ground_nodes:1_000L ~max_nodes)
    in
    let term =
      Err.or_raise ~pp_error:Ground_eval.pp_error
        (Ground_eval.at ~meter env b_id origin)
    in
    let expanded =
      Err.or_raise ~pp_error:Ground_eval.pp_error
        (Ground_eval.expand ~meter ~boundary:no_boundary env term)
    in
    Format.asprintf "%a (size %Ld)" Ground_expr.pp
      (Ground_eval.Term.expression expanded)
      (Ground_eval.Term.size expanded)
  in
  Fmt.pr "too tight: %s@.exactly enough: %s@." (run ~max_nodes:2)
    (run ~max_nodes:3);
  [%expect
    {|
    too tight: f32(src.t0(0)) (size 2)
    exactly enough: f32(f32(0x1p+0)) (size 3) |}]
