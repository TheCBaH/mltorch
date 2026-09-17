(* [Eval_direct]/[Eval_symbolic]'s extension of [Mul_scalar]'s own
   Bool-rejection fix (`mul_scalar_i64_test.ml`) to the rest of the
   `*_scalar` family: [Add_scalar]/[Div_scalar]/[Floor_div_scalar]/[Pow]/
   [Rpow_scalar]/[Rsub_scalar]/[Addcmul] had no per-format admission point
   of their own before this fix -- every operand format reached the generic
   default float path unchecked, so a genuine [Payload.Bool] operand would
   silently read as 0./1. and combine with the compile-time scalar. This
   does not add I64 support to any of these seven: an I64 operand still
   flows through the same silently-lossy float path it always has. See the
   implementation tracker's P6.3 note. *)

open Graph_ir
open Graph_direct_fixtures

let bool_x () = Tensor.materialize_bool (s1c 3) (fun _ -> true)

let run_direct op_of =
  let open Err.Syntax in
  let* g =
    lift_build
      Graph_builder.(
        build ~name:"scalar_bool" ~outputs:(fun r -> [ r ])
        @@
        let* x = input ~shape:(s1c 3) ~name:"x" ~fmt:Payload.(Fmt Bool) () in
        op_of x)
  in
  lift_eval
    (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ bool_x () ]))

let pp_ok ppf (_ : Tensor.packed Tensor_id.Map.t) = Fmt.string ppf "ok"

let%expect_test
    "Direct graph: arithmetic on a Bool operand is rejected for the rest of \
     the *_scalar family" =
  Graph_builder.(
    Format.printf "%a@." (pp_result pp_ok) (run_direct (add_scalar 2.5));
    Format.printf "%a@." (pp_result pp_ok) (run_direct (div_scalar 2.5));
    Format.printf "%a@." (pp_result pp_ok) (run_direct (floor_div_scalar 2.5));
    Format.printf "%a@." (pp_result pp_ok) (run_direct (pow 2.5));
    Format.printf "%a@." (pp_result pp_ok) (run_direct (rpow_scalar 2.5));
    Format.printf "%a@." (pp_result pp_ok)
      (run_direct
         (rsub_scalar { Pointwise.Rsub_scalar.other = 1.0; alpha = 1.0 }));
    Format.printf "%a@." (pp_result pp_ok)
      (run_direct (fun x -> addcmul 1.0 x x x)));
  [%expect
    {|
    add_scalar: arithmetic on a Bool operand is not supported, x=bool
    div_scalar: arithmetic on a Bool operand is not supported, x=bool
    floor_div_scalar: arithmetic on a Bool operand is not supported, x=bool
    pow: arithmetic on a Bool operand is not supported, x=bool
    rpow_scalar: arithmetic on a Bool operand is not supported, x=bool
    rsub_scalar: arithmetic on a Bool operand is not supported, x=bool
    addcmul: arithmetic on a Bool operand is not supported, x=bool
    |}]

(* The Symbolic twin of the fixture above: [Eval_symbolic.run] raises rather
   than returning [Err.t], the same convention `eval_symbolic_mixed_dtype_
   test.ml`'s own [catch] already uses. *)
let catch f =
  try
    ignore (f ());
    "no exception"
  with Err.Exn.E e -> Format.asprintf "raised: %a" Err.Exn.pp_kind e

let run_symbolic op_of () =
  let g =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"scalar_bool" ~outputs:(fun r -> [ r ])
        @@
        let* x = input ~shape:(s1c 3) ~name:"x" ~fmt:Payload.(Fmt Bool) () in
        op_of x)
  in
  Eval_symbolic.run g

let%expect_test
    "Symbolic graph: arithmetic on a Bool operand is rejected for the rest of \
     the *_scalar family" =
  Graph_builder.(
    Fmt.pr "%s@." (catch (run_symbolic (add_scalar 2.5)));
    Fmt.pr "%s@." (catch (run_symbolic (div_scalar 2.5)));
    Fmt.pr "%s@." (catch (run_symbolic (floor_div_scalar 2.5)));
    Fmt.pr "%s@." (catch (run_symbolic (pow 2.5)));
    Fmt.pr "%s@." (catch (run_symbolic (rpow_scalar 2.5)));
    Fmt.pr "%s@."
      (catch
         (run_symbolic
            (rsub_scalar { Pointwise.Rsub_scalar.other = 1.0; alpha = 1.0 })));
    Fmt.pr "%s@." (catch (run_symbolic (fun x -> addcmul 1.0 x x x))));
  [%expect
    {|
    raised: add_scalar: arithmetic on a Bool operand is not supported, x=bool
    raised: div_scalar: arithmetic on a Bool operand is not supported, x=bool
    raised: floor_div_scalar: arithmetic on a Bool operand is not supported, x=bool
    raised: pow: arithmetic on a Bool operand is not supported, x=bool
    raised: rpow_scalar: arithmetic on a Bool operand is not supported, x=bool
    raised: rsub_scalar: arithmetic on a Bool operand is not supported, x=bool
    raised: addcmul: arithmetic on a Bool operand is not supported, x=bool
    |}]
