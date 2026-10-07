(* A simultaneous transfer's cycle broken through the scratch: the saved value
   is read as its readers read it, whatever view of the register the move
   that overwrites it writes. *)

open Machine_ir
module Pc = Machine_alloc.Mir_parallel_copy
module Loc = Mir_phys.Loc
module R = Machine_target_aarch64.A64_reg

let value id ty = { Mir_value.id = Mir_id.Value.of_int id; ty }

let%expect_test "a cycle across a W and an X view" =
  let p = value 1 Mir_type.Ptr and i = value 2 Mir_type.i32 in
  let moves =
    Pc.resolve
      ~scratch:(fun (v : Mir_value.t) ->
        Loc.Reg
          (if Mir_type.equal v.Mir_value.ty Mir_type.Ptr then R.x 14 else R.w 14))
      [
        { Pc.dst = Loc.Reg (R.x 1); src = Loc.Reg (R.x 0); value = p };
        { Pc.dst = Loc.Reg (R.w 0); src = Loc.Reg (R.w 1); value = i };
      ]
  in
  List.iter
    (fun (m : Pc.move) ->
      Fmt.pr "%a: %a <- %a@." Mir_value.pp m.Pc.value Loc.pp m.Pc.dst Loc.pp
        m.Pc.src)
    moves;
  [%expect {|
    %2: w14 <- w1
    %1: x1 <- x0
    %2: w0 <- w14 |}]
