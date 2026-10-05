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

let reduction kind name data =
  {
    name;
    plan = Fusion_plan.default (Ssa_fixtures.four_cell_reduction kind);
    bind =
      (fun id ->
        if Tensor_id.equal id (Loop_fixtures.tid 0) then
          Some
            (Loop_fixtures.f32_tensor (Loop_programs.s1c 4) (fun c ->
                 data.((Vec6.offset (Loop_programs.s1c 4) c :> int))))
        else None);
  }

let format name ~fmt ?quant cells =
  let n = Array.length cells in
  {
    name;
    plan = Fusion_plan.default (Loop_programs.format_kernel ~fmt ?quant n);
    bind =
      (fun id ->
        if Tensor_id.equal id (Loop_fixtures.tid 0) then
          Some (Loop_programs.raw_tensor fmt ?quant n cells)
        else None);
  }

let int64_body name ~cells ?(floats = [| 2.9; -2.9; 0.; 0. |]) body =
  {
    name;
    plan = Fusion_plan.default (Loop_programs.i64_body_kernel body);
    bind = Loop_programs.i64_bind ~floats ~cells;
  }

let pool name ?(result = Expr.Intrinsic.Max_pool.Value) ~kernel ~pad data =
  {
    name;
    plan =
      Fusion_plan.default
        (Loop_programs.pool_kernel ~input:4 ~out:2 ~kernel ~stride:2 ~pad
           ~result);
    bind = Ssa_fixtures.bind_data ~shape:(Loop_programs.hw 4 4) data;
  }

let gather name cells =
  {
    name;
    plan = Fusion_plan.default Loop_programs.gather_kernel;
    bind = Loop_programs.i64_bind ~floats:[| 10.; 20.; 30.; 40. |] ~cells;
  }

let cases =
  [
    pointwise "pointwise specials" [| -0.; 1.5; nan; 3. |] Loop_programs.kernel;
    pointwise "pointwise f32 boundary"
      [| 1e30; -1e30; 0.1; 16777217. |]
      Loop_programs.kernel;
    pointwise "pointwise sub and div" [| 0.; 1.5; -2.25; 7. |] non_commutative;
    pointwise "pointwise exp" [| nan; -0.; 3.; -50. |]
      (Loop_programs.unary_kernel Expr.Value.Exp);
    pointwise "pointwise sqrt" [| nan; -0.; 2.; -1. |]
      (Loop_programs.unary_kernel Expr.Value.Sqrt);
    pointwise "lazy select" [| 5.; 6.; 7.; 8. |] Ssa_fixtures.lazy_select_kernel;
    format "f16 decode" ~fmt:(Payload.Fmt Payload.F16)
      [| 0x3c00L; 0x0001L; 0x7c00L; 0xfe00L |];
    format "i8 per channel" ~fmt:(Payload.Fmt Payload.I8)
      ~quant:
        (Err.or_raise ~pp_error:Quant.pp_error
           (Quant.per_channel ~scale:[| 0.5; 0.25; 2.; 0.1 |]
              ~zero_point:[| 0; 1; -2; 5 |]))
      [| -128L; 127L; 3L; -4L |];
    format "i64 read as a float" ~fmt:(Payload.Fmt Payload.I64)
      [|
        Int64.add (Int64.shift_left 1L 53) 1L; Int64.min_int; Int64.max_int; -1L;
      |];
    int64_body "i64 division" ~cells:[| -7L; 7L; -7L; 7L |]
      (Expr.Value.i64_div Loop_programs.i64_here (Expr.Value.i64_const 2L));
    int64_body "i64 division by zero" ~cells:[| -7L; 7L; -7L; 7L |]
      (Expr.Value.i64_div Loop_programs.i64_here (Expr.Value.i64_const 0L));
    int64_body "i64 modular multiply"
      ~cells:[| Int64.max_int; Int64.min_int; 3L; -5L |]
      (Expr.Value.i64_mul Loop_programs.i64_here Loop_programs.i64_here);
    int64_body "float to i64" ~floats:[| 2.9; -2.9; nan; 1e30 |]
      ~cells:[| 0L; 0L; 0L; 0L |]
      (Expr.Value.float_to_i64
         (Expr.Value.load
            (Expr_bridge.source_of_id (Loop_fixtures.tid 0))
            Loop_programs.out_coord));
    pool "max pool padded window" ~kernel:3 ~pad:1 (Array.init 16 float_of_int);
    pool "max pool index with NaN" ~result:Expr.Intrinsic.Max_pool.Index
      ~kernel:2 ~pad:0
      (Array.init 16 (fun i -> if i < 2 then nan else float_of_int i));
    gather "gather in range and negative" [| 0L; -1L; -4L; 3L |];
    gather "gather out of range" [| 4L; 0L; 0L; 0L |];
    reduction Expr.Reduction.Max "max with NaN" [| nan; 1.; nan; 0. |];
    reduction Expr.Reduction.Argmax_index "argmax index ties"
      [| 1.; 3.; 3.; 2. |];
    reduction Expr.Reduction.Argmax_value "argmax value NaN"
      [| 1.; nan; 3.; nan |];
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
