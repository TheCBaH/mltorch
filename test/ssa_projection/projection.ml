open Ssa_bridge
open Loop_ir
open Loop_ir_test
open Ssa_bridge_test

(* SSA programs run by the existing emitters, checked against the reference.

   An SSA program has no emitter of its own yet, so it is projected to the Loop
   IR ({!Loop_of_ssa}) and handed to a backend's [Loop_check.Executor]. Shared,
   unmodified, by the C, Wasm and JavaScript suites: each supplies only the
   executor, so one table of cases describes what every backend is asked.

   Two routes, both against [Kernel_eval]: the source-direct SSA program, and
   the bridge program (Loop IR to SSA and back), so agreement between an adapter
   and its own output is never the evidence. *)

type case = {
  name : string;
  plan : Fusion_plan.t;
  bind : Tensor_id.t -> Tensor.packed option;
}

let pointwise name data kernel =
  {
    name;
    plan = Fusion_plan.default kernel;
    bind = Ssa_fixtures.bind_data ~shape:(Loop_fixtures.shape_w 4) data;
  }

let matmul (m, k, n) =
  {
    name = Fmt.str "matmul %dx%dx%d" m k n;
    plan = Fusion_plan.default (Ssa_fixtures.matmul_kernel ~m ~k ~n);
    bind =
      Ssa_fixtures.matmul_bind ~m ~k ~n
        ~a:(Ssa_fixtures.operand 3 (m * k))
        ~b:(Ssa_fixtures.operand 5 (k * n));
  }

let empty_sum ~lo ~hi =
  {
    name = Fmt.str "sum [%d,%d)" lo hi;
    plan =
      Fusion_plan.default
        (Loop_programs.reduction_kernel Expr.Reduction.Sum
           ~lo:(Expr.Index.assume_position (Expr.Index.const lo))
           ~hi:(Expr.Index.const hi));
    bind =
      (fun id ->
        if Tensor_id.equal id (Loop_fixtures.tid 0) then
          Some (Loop_fixtures.f32_tensor (Loop_programs.s1c 3) (fun _ -> -0.))
        else None);
  }

let chain =
  let fused, _ =
    Fusion_plan.plan (Loop_programs.chain ~outputs:[ Loop_fixtures.tid 2 ])
  in
  {
    name = "chain, inlined producer";
    plan = fused;
    bind =
      Ssa_fixtures.bind_data ~shape:(Loop_fixtures.shape_w 2)
        [| Loop_programs.two24; 1. |];
  }

let non_commutative =
  Loop_fixtures.pixel_kernel
    (Expr.Value.div
       (Expr.Value.sub Loop_fixtures.load_t0 (Expr.Value.const 1.5))
       (Expr.Value.const 0.7))

let cases =
  [
    pointwise "pointwise specials" [| -0.; 1.5; nan; 3. |] Loop_programs.kernel;
    pointwise "pointwise f32 boundary"
      [| 1e30; -1e30; 0.1; 16777217. |]
      Loop_programs.kernel;
    pointwise "pointwise sub and div" [| 0.; 1.5; -2.25; 7. |] non_commutative;
    matmul (1, 1, 1);
    matmul (1, 3, 2);
    matmul (5, 7, 3);
    matmul (4, 4, 4);
    empty_sum ~lo:0 ~hi:0;
    empty_sum ~lo:2 ~hi:1;
    empty_sum ~lo:0 ~hi:3;
    chain;
  ]

let verdict ~(exec : Loop_check.Executor.t) (c : case) program =
  let reference = Kernel_eval.run_plan c.plan ~bind:c.bind in
  match Err.payload (exec (Loop_of_ssa.convert program) ~bind:c.bind) with
  | Error (`Js_compile m | `Js_exception m) -> Fmt.str "host failure: %s" m
  | Error (#Loop_interp.error as e) ->
      Fmt.str "%a" Loop_check.pp_verdict
        (Loop_check.compare ~reference ~loop:(Err.fail e))
  | Ok outputs ->
      Fmt.str "%a" Loop_check.pp_verdict
        (Loop_check.compare ~reference ~loop:(Err.return outputs))

let direct (c : case) =
  match Err.payload (Ssa_lower.Ssa_lower_plan.lower c.plan) with
  | Ok p -> Some p
  | Error (`Unsupported _) -> None

let bridged (c : case) =
  match Err.payload (Loop_lower.lower c.plan) with
  | Error _ -> None
  | Ok loop -> (
      match Err.payload (Ssa_of_loop.convert loop) with
      | Ok p -> Some p
      | Error (`Unsupported _) -> None)

(* One line per case and route. *)
let run ~exec =
  List.iter
    (fun (c : case) ->
      List.iter
        (fun (route, program) ->
          match program c with
          | None -> Fmt.pr "%s, %s: not lowered@." c.name route
          | Some p -> Fmt.pr "%s, %s: %s@." c.name route (verdict ~exec c p))
        [ ("direct", direct); ("bridge", bridged) ])
    cases
