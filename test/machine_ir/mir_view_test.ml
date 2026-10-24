(* Register views that span several units (RVV register groups), checked on
   hand-built views of an RV32-like vector bank (vK is unit 65 + K, 128 bits):
   their members, which views they overlap, and which are the same register.
   The same checks run against two deliberately wrong implementations, which
   must fail them. *)

open Machine_ir
module V = Mir_target.View

let v ?(span = Mir_target.Span.One) k =
  {
    V.name =
      (match span with
      | Mir_target.Span.One -> Printf.sprintf "v%d" k
      | s -> Printf.sprintf "v%d%s" k (Mir_target.Span.name s));
    bank = Mir_target.Bank.Vector;
    unit = Mir_id.Unit.of_int (65 + k);
    span;
    lo = 0;
    bits = 128 * Mir_target.Span.count span;
  }

let m4 = v ~span:Mir_target.Span.Four
let m8 = v ~span:Mir_target.Span.Eight

(* The checks, against an [overlap] and a [same_register]; the failures they
   find, as text. *)
let check ~overlap ~same_register =
  let fails = ref [] in
  let expect what got want = if got <> want then fails := what :: !fails in
  let ov a b want =
    expect (Fmt.str "overlap %s %s" a.V.name b.V.name) (overlap a b) want;
    expect (Fmt.str "overlap %s %s" b.V.name a.V.name) (overlap b a) want
  in
  ov (m4 4) (v 4) true;
  ov (m4 4) (v 5) true;
  ov (m4 4) (v 7) true;
  ov (m4 4) (v 8) false;
  ov (m4 4) (v 3) false;
  ov (m4 4) (m8 0) true;
  ov (m4 4) (m4 8) false;
  ov (m8 8) (m4 12) true;
  ov (v 4) (v 4) true;
  ov (v 4) (v 5) false;
  expect "v4 same as v4m4" (same_register (v 4) (m4 4)) false;
  expect "v4m4 same as v4m4" (same_register (m4 4) (m4 4)) true;
  expect "v4m4 same as v4m8" (same_register (m4 4) (m8 4)) false;
  List.rev !fails

let show name fails =
  Fmt.pr "%s: %s@." name
    (match fails with [] -> "passes" | l -> "fails " ^ String.concat "; " l)

let%expect_test "members of a group" =
  List.iter
    (fun g ->
      Fmt.pr "%s: %s@." g.V.name
        (String.concat " "
           (List.map
              (fun u -> Printf.sprintf "u%d" (Mir_id.Unit.to_int u))
              (V.units g))))
    [ v 4; m4 4; m8 8 ];
  [%expect
    {|
    v4: u69
    v4m4: u69 u70 u71 u72
    v8m8: u73 u74 u75 u76 u77 u78 u79 u80 |}]

let%expect_test "overlap and register identity, and two wrong versions" =
  show "the views" (check ~overlap:V.overlap ~same_register:V.same_register);
  (* overlap by the base unit alone: v4m4 then misses v5 *)
  show "base-unit overlap"
    (check
       ~overlap:(fun a b -> Mir_id.Unit.equal a.V.unit b.V.unit)
       ~same_register:V.same_register);
  (* the unit-only tied check: v4 then counts as the v4m4 group *)
  show "unit-only identity"
    (check ~overlap:V.overlap ~same_register:(fun a b ->
         Mir_id.Unit.equal a.V.unit b.V.unit));
  [%expect
    {|
    the views: passes
    base-unit overlap: fails overlap v4m4 v5; overlap v5 v4m4; overlap v4m4 v7; overlap v7 v4m4; overlap v4m4 v0m8; overlap v0m8 v4m4; overlap v8m8 v12m4; overlap v12m4 v8m8
    unit-only identity: fails v4 same as v4m4; v4m4 same as v4m8 |}]
