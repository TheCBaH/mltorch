(* The Native4D twin of `test/native/scalar_family_bool_test.ml`: none of
   Add_scalar/Div_scalar/Floor_div_scalar/Pow/Rpow_scalar/Rsub_scalar/
   Addcmul has ANY Native4D dispatch at all (confirmed by `grep -n`), so
   every operand format reached the generic default float path unchecked --
   a genuine [Payload.Bool] operand would silently read as 0./1. and
   combine with the compile-time scalar. See the implementation tracker's
   P6.3 note. *)

open Native4d

let shape3 = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:3
let bool_x () = Tensor.materialize_bool (Shape4.to_vec6 shape3) (fun _ -> true)

let run_direct op_of =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape3 ~fmt:Payload.(Fmt Bool) () in
       op_of x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ bool_x () ])

let pp_ok fmt = function
  | Ok (_ : Tensor.packed Tensor_id.Map.t) -> Fmt.string fmt "ok"
  | Error e -> Fmt.pf fmt "%a" Eval_direct4.pp_error (Err.Error.kind e)

let%expect_test
    "direct4: arithmetic on a Bool operand is rejected for the rest of the \
     *_scalar family" =
  Builder.(
    Fmt.pr "%a@." pp_ok (run_direct (add_scalar 2.5));
    Fmt.pr "%a@." pp_ok (run_direct (div_scalar 2.5));
    Fmt.pr "%a@." pp_ok (run_direct (floor_div_scalar 2.5));
    Fmt.pr "%a@." pp_ok (run_direct (pow 2.5));
    Fmt.pr "%a@." pp_ok (run_direct (rpow_scalar 2.5));
    Fmt.pr "%a@." pp_ok
      (run_direct
         (rsub_scalar { Pointwise.Rsub_scalar.other = 1.0; alpha = 1.0 }));
    Fmt.pr "%a@." pp_ok (run_direct (fun x -> addcmul 1.0 x x x)));
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

let catch f =
  try
    ignore (f ());
    "no exception"
  with Err.Exn.E e -> Format.asprintf "raised: %a" Err.Exn.pp_kind e

let run_symbolic op_of () =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape3 ~fmt:Payload.(Fmt Bool) () in
       op_of x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  Eval_symbolic4.run g

let%expect_test
    "Symbolic4 graph: arithmetic on a Bool operand is rejected for the rest of \
     the *_scalar family" =
  Builder.(
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
