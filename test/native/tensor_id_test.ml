(* The id-space guard both multi-output builders share ([Graph_builder.opN] and
   Native4D's), so their overflow behaviour cannot drift.

   An INVARIANT, not a reachable resource ceiling — per-node output count is
   already bounded at [Kernel.Limits.Hard.outputs], so reaching the id space
   needs on the order of 2^31 live edges each holding a [Tensor_sig.t] in a map,
   and memory is exhausted first by orders of magnitude. That is why it raises
   rather than returning a row the way [Shape_error.Output_count] does. *)

(* Written [count > max_int - next], never [next + count > max_int]: under
   js_of_ocaml [Sys.int_size] is 32, so the naive spelling wraps to a negative
   and sails past the comparison. Both boundary directions are exercised, so the
   check is known to be neither vacuous nor off by one.

   The cases are LABELLED rather than printed with their operands, because
   [max_int] differs between the native and js_of_ocaml backends and this file
   runs under both against this one golden. Only the verdict is backend-
   independent, so only the verdict is asserted. *)
let%expect_test "Tensor_id.check_room: bounded before the addition, not after" =
  let show label ~next ~count =
    Printf.printf "%-22s -> %s\n" label
      (match Tensor_id.check_room ~next ~count with
      | () -> "ok"
      | exception Invalid_argument _ -> "refused")
  in
  show "nothing to allocate" ~next:0 ~count:0;
  show "an ordinary node" ~next:10 ~count:4096;
  show "exactly fills" ~next:1 ~count:(max_int - 1);
  show "one past the end" ~next:1 ~count:max_int;
  show "no room at the top" ~next:max_int ~count:1;
  show "negative count" ~next:0 ~count:(-1);
  [%expect
    {|
    nothing to allocate    -> ok
    an ordinary node       -> ok
    exactly fills          -> ok
    one past the end       -> refused
    no room at the top     -> refused
    negative count         -> refused |}]
