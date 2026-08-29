(* Split out of what was verify_test.ml see verify_fixtures.ml. *)

open Graph_ir
open Verify_fixtures

(* ---- the rounding boundary ------------------------------------------------

   Every node output is materialized as float32 (Schedule.evaluate ->
   Tensor.materialize), so a stage boundary rounds. Inlining that away would
   turn f32(f32(a+b)*c) into f32((a+b)*c) and let a future fusion be "proved"
   identical while changing bits. [Round] keeps the boundary in the term; these
   pin the three rules that may remove one. *)

(* [Src] arbitrarily: these are unit tests of normalisation, where the side an
   id belongs to is not what is under test — only that both cells carry the same
   one, since normalisation never compares across graphs. *)
let cell n =
  {
    Ground_expr.Cell.origin = Ground_expr.Origin.Src (Tensor_id.of_int n);
    coord = Vec6.origin;
  }

let show_norm ~stored_f32 e =
  let n = Ground_expr.normalise ~stored_f32 e in
  Format.printf "%a  blocked=[%a]@." Ground_expr.pp n.Ground_expr.expr
    (Fmt.list ~sep:Fmt.comma Ground_expr.Cell.pp)
    (Ground_expr.Cell.Set.elements n.Ground_expr.blocked)

let%expect_test "normalise: a cell is already stored, so its Round collapses" =
  let all_f32 _ = true in
  show_norm ~stored_f32:all_f32 (Ground_expr.Round (Ground_expr.Cell (cell 0)));
  (* idempotent *)
  show_norm ~stored_f32:all_f32
    (Ground_expr.Round (Ground_expr.Round (Ground_expr.Cell (cell 0))));
  (* a constant is folded to its f32 image, so a fold can be compared bitwise
     against a payload the pass computed through the same materialization *)
  show_norm ~stored_f32:all_f32 (Ground_expr.Round (Ground_expr.Const 0.1));
  [%expect
    {|
    src.t0(0)  blocked=[]
    src.t0(0)  blocked=[]
    0x1.99999ap-4  blocked=[] |}]

let%expect_test "normalise: a computed Round is NOT removed" =
  (* The whole point: only a stored value or a constant may lose its boundary.
     An arithmetic node keeps it, so two graphs that differ only in where they
     materialize do not compare equal. *)
  show_norm
    ~stored_f32:(fun _ -> true)
    (Ground_expr.Round
       (Ground_expr.Binary
          (Expr.Value.Add, Ground_expr.Cell (cell 0), Ground_expr.Cell (cell 1))));
  [%expect {| f32((src.t0(0) + src.t1(0)))  blocked=[] |}]

let%expect_test "normalise: a non-f32 cell blocks the collapse" =
  (* [Payload.get_float] decodes I32/I64 via Int32/Int64.to_float, which leaves
     f32's exact range above 2^24, and I8/I16 through a dequantizing multiply.
     For those the materialization is observable, so the Round has to stay. *)
  show_norm
    ~stored_f32:(fun _ -> false)
    (Ground_expr.Round (Ground_expr.Cell (cell 0)));
  [%expect {| f32(src.t0(0))  blocked=[src.t0(0)] |}]

(* End to end: trimming an identity permute off a non-F32 input is not merely
   unproven, it is FALSE for a large enough value — the permute's f32
   materialization is what the source computes and the destination skips.

   [Trim_permute] therefore DECLINES the match: a permute's output is
   materialized as f32, so tying it to an i32 input would claim
   [{t0(i32), t1(f32)} -> {t0}] identical, which is the contradiction step 9 of
   native_transform_design.md §7 names. The graph is left alone and the three
   untouched clusters verify, which is the right answer for "this rewrite does
   not apply here".

   Declining rather than applying-and-being-rejected matters because
   [Graph_map.create]'s rejection is an ERROR: it stops [Pass.run_all] and every
   later pass with it, so one unsupported match anywhere would take down a
   pipeline that has nothing else wrong with it.

   The blocked-collapse machinery itself is still covered, by "normalise: a
   non-f32 cell blocks the collapse" above. *)
let%expect_test "verify: trimming a permute off an i32 input is declined" =
  let g =
    build "i32_permute"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" ~fmt:(Payload.Fmt Payload.I32) () in
        let* t1 = permute Graph_fixtures.identity_perm a in
        relu t1)
  in
  detail "i32 input [trim]" g [ Trim_permute.pass ];
  [%expect
    {|
    i32 input [trim]:
      {t0} -> {t0} identical: proved (structural) [exhaustive]
      {t1} -> {t1} identical: proved (structural) [exhaustive]
      {t2} -> {t2} identical: proved (structural) [exhaustive] |}]
