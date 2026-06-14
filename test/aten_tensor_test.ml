(* Tensor object-API tests: RAII / live count, metadata accessors, and the
   error boundary.

   RAII: every handle that crosses into OCaml is passed through [Tensor.manage]
   (done by [Tensor.create]), which attaches a Gc finaliser that atc_free's the
   C++ at::Tensor when the OCaml handle is collected. The shim keeps an exact
   live count (atc_wrap allocations not yet freed), so we can assert the
   round-trip: live rises with allocation and returns to baseline once handles
   are unreachable and the GC has run their finalisers. *)

open Ctypes
module F = Aten.C.Functions
module O = Aten.C.Operations
module Stype = Aten.Scalar_type
module Dtype = Aten.Dtype
module T = Aten.Tensor

(* A managed, uninitialised tensor of [shape]. No data view is taken, so the
   only finaliser is the free — no anchoring view to keep the handle alive
   across a GC. Values are irrelevant here; we only count handles. *)
let alloc shape = T.create shape

let ints a =
  "[" ^ String.concat ";" (Array.to_list (Array.map string_of_int a)) ^ "]"

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

let%expect_test "metadata accessors" =
  let t = T.create [ 2; 3 ] in
  Printf.printf
    "dim=%d numel=%d esize=%d dtype=%d contig=%b cpu=%b defined=%b\n" (T.dim t)
    (T.numel t) (T.element_size t)
    (Stype.to_int (T.scalar_type t))
    (T.is_contiguous t) (T.is_cpu t) (T.defined t);
  Printf.printf "shape=%s strides=%s\n" (ints (T.shape t)) (ints (T.strides t));
  [%expect
    {|
    dim=2 numel=6 esize=4 dtype=6 contig=true cpu=true defined=true
    shape=[2;3] strides=[3;1] |}]

let%expect_test "error boundary: bad dtype raises Tensor.Error" =
  (* atc_new with an unknown dtype code throws a c10::Error in C++; the boundary
     catches it, returns the null sentinel, and Tensor.check raises. *)
  let bad =
    let sizes = CArray.of_list int64_t [ 2L; 3L ] in
    F.new_ (CArray.start sizes) (Unsigned.Size_t.of_int 2) 99
  in
  (match T.check bad with
  | exception T.Error _ -> print_string "raised Tensor.Error"
  | _ -> print_string "no error");
  [%expect "raised Tensor.Error"]

let%expect_test "data: typed view round-trips; wrong dtype is None" =
  let t = T.create [ 2; 2 ] in
  let v = T.data Dtype.float32 t |> Option.get in
  v.{0} <- 1.5;
  v.{1} <- 2.5;
  v.{2} <- 3.5;
  v.{3} <- 4.5;
  (* a fresh view sees the same backing store *)
  let v2 = T.data Dtype.float32 t |> Option.get in
  Printf.printf "%g %g %g %g\n" v2.{0} v2.{1} v2.{2} v2.{3};
  Printf.printf "int64 view of a float tensor: %b\n"
    (Option.is_none (T.data Dtype.int64 t));
  [%expect {|
    1.5 2.5 3.5 4.5
    int64 view of a float tensor: true |}]

let%expect_test "data: int64 tensor" =
  let t = T.create ~dtype:Stype.Long [ 3 ] in
  let v = T.data Dtype.int64 t |> Option.get in
  v.{0} <- 10L;
  v.{1} <- 20L;
  v.{2} <- 30L;
  Printf.printf "%Ld %Ld %Ld\n" v.{0} v.{1} v.{2};
  [%expect "10 20 30"]

let%expect_test "of_bigarray copies the source" =
  let src =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout
      [| 1.; 2.; 3.; 4.; 5.; 6. |]
  in
  let t = T.of_bigarray Dtype.float32 src [ 2; 3 ] in
  let v = T.data Dtype.float32 t |> Option.get in
  Printf.printf "shape=%s data=%g,%g,%g,%g,%g,%g\n"
    (ints (T.shape t))
    v.{0} v.{1} v.{2} v.{3} v.{4} v.{5};
  src.{0} <- 99.;
  (* it's a copy: mutating src does not change the tensor *)
  Printf.printf "after src.{0}<-99: %g\n"
    (T.data Dtype.float32 t |> Option.get).{0};
  [%expect {|
    shape=[2;3] data=1,2,3,4,5,6
    after src.{0}<-99: 1 |}]

let%expect_test "item_float / item_int; numel<>1 raises" =
  let tf = T.create [ 1 ] in
  (T.data Dtype.float32 tf |> Option.get).{0} <- 42.5;
  Printf.printf "item_float=%g\n" (T.item_float tf);
  let ti = T.create ~dtype:Stype.Long [ 1 ] in
  (T.data Dtype.int64 ti |> Option.get).{0} <- 7L;
  Printf.printf "item_int=%d\n" (T.item_int ti);
  (match T.item_float (T.create [ 2; 3 ]) with
  | exception T.Error _ -> print_string "raised on numel<>1\n"
  | _ -> print_string "no error\n");
  [%expect {|
    item_float=42.5
    item_int=7
    raised on numel<>1 |}]

(* Phase 3: kernel-backed conversions. *)

let float_tensor shape vals =
  let t = T.create shape in
  let v = T.data Dtype.float32 t |> Option.get in
  List.iteri (fun i x -> v.{i} <- x) vals;
  t

let floats t =
  let v = T.data Dtype.float32 t |> Option.get in
  String.concat ","
    (List.init (Bigarray.Array1.dim v) (fun i -> Printf.sprintf "%g" v.{i}))

let%expect_test "clone makes an independent copy" =
  let a = float_tensor [ 3 ] [ 1.; 2.; 3. ] in
  let b = T.manage (O.clone a) in
  (T.data Dtype.float32 a |> Option.get).{0} <- 99.;
  Printf.printf "a=%s clone=%s\n" (floats a) (floats b);
  [%expect "a=99,2,3 clone=1,2,3"]

let%expect_test "contiguous / cpu preserve values" =
  let a = float_tensor [ 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  Printf.printf "contiguous=%s cpu=%s\n"
    (floats (T.manage (O.contiguous a)))
    (floats (T.manage (O.cpu a)));
  [%expect "contiguous=1,2,3,4 cpu=1,2,3,4"]

let%expect_test "to.dtype casts float -> int64 (truncating)" =
  let a = float_tensor [ 3 ] [ 1.5; 2.7; 3.9 ] in
  let i = T.manage (O.to_dtype a (Stype.to_int Stype.Long) false false) in
  let v = T.data Dtype.int64 i |> Option.get in
  Printf.printf "dtype=%d vals=%Ld,%Ld,%Ld\n"
    (Stype.to_int (T.scalar_type i))
    v.{0} v.{1} v.{2};
  [%expect "dtype=4 vals=1,2,3"]

(* Phase 4: pretty-printing and comparison helpers. *)

let%expect_test "pp renders via ATen's printer" =
  let t = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  Format.printf "%a@." T.pp t;
  [%expect {|
     1  2  3
     4  5  6
    [ CPUFloatType{2,3} ] |}]

let%expect_test "allclose / equal" =
  let a = float_tensor [ 3 ] [ 1.; 2.; 3. ] in
  let b = float_tensor [ 3 ] [ 1.; 2.; 3. ] in
  let c = float_tensor [ 3 ] [ 1.; 2.; 3.5 ] in
  Printf.printf "equal a b=%b  equal a c=%b\n" (T.equal a b) (T.equal a c);
  Printf.printf "allclose a b=%b  allclose a c=%b  allclose ~atol:1.0 a c=%b\n"
    (T.allclose a b) (T.allclose a c) (T.allclose ~atol:1.0 a c);
  [%expect
    {|
    equal a b=true  equal a c=false
    allclose a b=true  allclose a c=false  allclose ~atol:1.0 a c=true |}]
