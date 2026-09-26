open Loop_ir
open Loop_fixtures
open Loop_programs

(* The foundation refuses everything, with the construct named: a partially
   lowered program is a defect however plausible its output. One kernel of each
   form the kernel IR admits. *)

let refusal name plan =
  match Err.payload (Loop_lower.lower plan) with
  | Ok _ -> Fmt.pr "%s: lowered@." name
  | Error (`Unsupported u) -> Fmt.pr "%s: %a@." name Loop_unsupported.pp u

let%expect_test "every form of value is refused, named" =
  (* A Region program: one scalar local read by the pixel. *)
  refusal "int64 load of an f32 input"
    (Fusion_plan.default i64_load_of_f32_kernel);
  refusal "int64" (Fusion_plan.default i64_kernel);
  [%expect
    {|
    int64 load of an f32 input: t1: load of format f32 is not lowered
    int64: lowered |}]

let%expect_test "a kernel with nothing to execute lowers to the empty program" =
  let empty =
    Err.or_raise ~pp_error:Kernel.pp_error
      (Kernel.create ~inputs:[] ~values:[] ~outputs:[] ())
  in
  refusal "empty" (Fusion_plan.default empty);
  [%expect {| empty: lowered |}]
