open Ssa_bridge
open Ssa_fixtures
open Loop_ir_test

let show ?(shape = Loop_fixtures.shape_w 4) ~data kernel =
  let plan = Fusion_plan.default kernel in
  Fmt.pr "%a@." Ssa_check.pp_verdict
    (Ssa_check.run plan ~bind:(bind_data ~shape data))

let%expect_test "pointwise: signed zero, NaN and the f32 boundary agree" =
  show ~data:[| -0.; 1.5; nan; 3. |] Loop_programs.kernel;
  show ~data:[| 1e30; -1e30; 0.1; 16777217. |] Loop_programs.kernel;
  [%expect {|
    agree
    agree
    |}]

let%expect_test "a failure agrees in kind and payload" =
  show ~data:[| 0.; 0.; 0.; 0. |] Loop_programs.shifted_kernel;
  [%expect {| agree on failure: coord_out_of_range |}]

(* 2^30 * w leaves the index domain at w = 2. The reference's checked domain is
   its host int: 32 bits under js_of_ocaml, where it reports the same first
   operation, and wider natively, where it computes 2^31 without complaint. *)
let%expect_test "an index that leaves the domain is a failure, never a wrap" =
  let plan = Fusion_plan.default Loop_programs.overflow_kernel in
  let bind = bind_data ~shape:(Loop_fixtures.shape_w 4) [| 0.; 0.; 0.; 0. |] in
  (match Err.payload (Ssa_lower.Ssa_lower_plan.lower plan) with
  | Error _ -> Fmt.pr "refused@."
  | Ok program -> (
      match Err.payload (Ssa_lower.Ssa_exec.run plan program ~bind) with
      | Error e -> Fmt.pr "%a@." Ssa_lower.Ssa_exec.pp_error e
      | Ok _ -> Fmt.pr "no failure@."));
  Fmt.pr "agrees where the reference's int is 32 bits: %b@."
    (Sys.int_size > 32
    ||
    match Ssa_check.run plan ~bind with
    | Ssa_check.Agree_on_failure _ -> true
    | _ -> false);
  [%expect
    {|
    index mul overflows on 1073741824 and 2
    agrees where the reference's int is 32 bits: true
    |}]

let%expect_test "binding validation precedes evaluation, in input order" =
  let plan = Fusion_plan.default Loop_programs.kernel in
  let verdict bind =
    Fmt.pr "%a@." Ssa_check.pp_verdict (Ssa_check.run plan ~bind)
  in
  verdict (fun _ -> None);
  verdict (fun id ->
      if Tensor_id.equal id (Loop_fixtures.tid 0) then
        Some (Loop_fixtures.f32_tensor (Loop_fixtures.shape_w 2) (fun _ -> 0.))
      else None);
  [%expect
    {|
    agree on failure: unbound_input
    agree on failure: binding_mismatch
    |}]

let%expect_test "unsupported constructs are refusals, not failures" =
  let refused kernel =
    match Ssa_check.run (Fusion_plan.default kernel) ~bind:(fun _ -> None) with
    | Ssa_check.Refused u ->
        Fmt.pr "%s@." (Ssa_lower.Ssa_unsupported.construct_name u.construct)
    | v -> Fmt.pr "%a@." Ssa_check.pp_verdict v
  in
  (* an int64 read of an f32 input is refused by format; the reference fails at
     run time, so the two are never compared *)
  refused Loop_programs.i64_load_of_f32_kernel;
  (* a region program is no longer a refusal *)
  refused Loop_fixtures.region_kernel;
  [%expect {|
    load of format f32
    agree |}]
