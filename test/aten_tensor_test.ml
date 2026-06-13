(* Tensor lifecycle / RAII tests: every handle that crosses into OCaml is
   passed through [Tensor.manage], which attaches a Gc finaliser that atc_free's
   the C++ at::Tensor when the OCaml handle is collected. The C++ shim keeps an
   exact live count (atc_wrap allocations not yet atc_free'd), exposed as
   [Tensor.live_count], so we can assert the round-trip: live rises with
   allocation and returns to baseline once handles are unreachable and the GC
   has run their finalisers. *)

open Ctypes
module F = Aten.C.Functions
module O = Aten.C.Operations
module Stype = Aten.Scalar_type
module T = Aten.Tensor

(* A managed, uninitialised tensor of [shape]. No data view is taken (unlike the
   op tests), so the only finaliser is the free — no anchoring view to keep the
   handle alive across a GC. Values are irrelevant here; we only count handles. *)
let alloc shape =
  let sizes = CArray.of_list int64_t (List.map Int64.of_int shape) in
  T.manage
    (F.new_ (CArray.start sizes)
       (Unsigned.Size_t.of_int (List.length shape))
       (Stype.to_int Stype.Float))

(* Two full majors: the first collects the now-unreachable handles, and running
   finalisers can unreference further handles (e.g. op results aliasing inputs),
   so a second pass drains those too. *)
let collect () =
  Gc.full_major ();
  Gc.full_major ()

let%expect_test "live count rises on alloc, returns to baseline after GC" =
  collect ();
  let base = T.live_count () in
  (* Allocate in an inner scope so the handles are unreachable once it returns;
     measure the live delta while they are still referenced. *)
  let while_referenced =
    let ts = List.init 10 (fun _ -> alloc [ 3 ]) in
    let delta = T.live_count () - base in
    ignore (Sys.opaque_identity ts);
    delta
  in
  Printf.printf "while referenced: +%d\n" while_referenced;
  collect ();
  Printf.printf "after gc: +%d\n" (T.live_count () - base);
  [%expect {|
    while referenced: +10
    after gc: +0 |}]

let%expect_test "long op sequence leaves no live tensors" =
  collect ();
  let base = T.live_count () in
  (* A sequence in the spirit of aten_ops_test: inputs + several op results per
     iteration, all dropped each turn. Values are uninitialised — we only care
     that every atc_wrap'd handle (inputs and results) is freed. *)
  for _ = 1 to 100 do
    let a = alloc [ 2; 3 ] and b = alloc [ 2; 3 ] in
    let c = T.manage (O.add_Tensor a b 1.0) in
    let d = T.manage (O.mul_Tensor a b) in
    let _ = T.manage (O.relu c) in
    let _ = T.manage (O.relu_ d) in
    ()
  done;
  collect ();
  Printf.printf "live delta after 100 iterations: %d\n" (T.live_count () - base);
  [%expect {| live delta after 100 iterations: 0 |}]
