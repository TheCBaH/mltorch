open Ctypes
open Bigarray
module F = C.Functions

type float32_array = (float, float32_elt, c_layout) Array1.t

(* RAII: attach a finaliser so the C++ at::Tensor behind [t] is atc_free'd when
   the OCaml handle becomes unreachable. Returns [t] for chaining. Every handle
   that crosses into OCaml (atc_new and op results) should be passed through
   this exactly once; double-managing would double-free. *)
let manage t =
  Gc.finalise F.free t;
  t

(* Number of live (atc_wrap'd, not yet freed) C++ tensors. *)
let live_count () = Int64.to_int (F.live_count ())
let numel t = Int64.to_int (F.numel t)

(* The tensor's shape as an int array (length = rank). *)
let shape t =
  let n = Unsigned.Size_t.to_int (F.dim t) in
  let out = CArray.make int64_t n in
  F.sizes t (CArray.start out);
  Array.init n (fun i -> Int64.to_int (CArray.get out i))

(* Return a Bigarray view over the tensor's contiguous buffer if its dtype
   matches [dtype], or None on mismatch.  The view shares memory with the
   tensor — the caller must not free the tensor while the view is live.
   A Gc finaliser on the returned array keeps the OCaml handle reachable,
   guarding against the GC collecting the handle box while the array exists;
   explicit [F.free t] while a view is live remains the caller's responsibility. *)
let as_float32 t : float32_array option =
  let n = numel t in
  let vp = F.data_ptr t (Scalar_type.to_int Scalar_type.Float) in
  if is_null vp then None
  else
    let fp = from_voidp float vp in
    let ba = bigarray_of_ptr array1 n float32 fp in
    Gc.finalise (fun _ -> ignore t) ba;
    Some ba

let pp_float32 fmt (ba : float32_array) =
  let seq = Seq.init (Array1.dim ba) (fun i -> ba.{i}) in
  Format.fprintf fmt "[%a]"
    (Format.pp_print_seq
       ~pp_sep:(fun fmt () -> Format.pp_print_string fmt "; ")
       (fun fmt v -> Format.fprintf fmt "%g" v))
    seq
