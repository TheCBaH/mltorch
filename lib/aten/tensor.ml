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

(* Raised when a shim call fails (ATen threw a c10::Error at the boundary). *)
exception Error of string

(* Raise [Error] if [t] is the failure sentinel (a null handle); otherwise
   return it. The message comes from the thread-local atc_last_error. *)
let check t =
  if is_null t then
    raise (Error (Option.value (F.last_error ()) ~default:"aten error"))
  else t

(* A new uninitialised contiguous CPU tensor, owned (freed by the GC via
   [manage]). Raises [Error] on a bad shape/dtype. *)
let create ?(dtype = Scalar_type.Float) shape =
  let sizes = CArray.of_list int64_t (List.map Int64.of_int shape) in
  F.new_ (CArray.start sizes)
    (Unsigned.Size_t.of_int (List.length shape))
    (Scalar_type.to_int dtype)
  |> check |> manage

let numel t = Int64.to_int (F.numel t)
let dim t = Unsigned.Size_t.to_int (F.dim t)
let element_size t = Int64.to_int (F.element_size t)
let is_contiguous t = F.is_contiguous t <> 0
let defined t = F.defined t <> 0
let is_cpu t = F.is_cpu t <> 0

(* The tensor's dtype. Total over the supported (CPU) dtype set. *)
let scalar_type t =
  let code = F.scalar_type t in
  match Scalar_type.of_int code with
  | Some s -> s
  | None ->
      raise (Error (Printf.sprintf "unsupported scalar type code %d" code))

(* Read the [atc_dim] int64 entries written by [fill] (sizes or strides). *)
let read_dims fill t =
  let n = Unsigned.Size_t.to_int (F.dim t) in
  let out = CArray.make int64_t n in
  fill t (CArray.start out);
  Array.init n (fun i -> Int64.to_int (CArray.get out i))

(* The tensor's shape / strides as int arrays (length = rank). *)
let shape t = read_dims F.sizes t
let strides t = read_dims F.strides t

(* A Bigarray view over the tensor's contiguous buffer at dtype [dt], or None if
   the tensor's dtype differs. The view shares memory with the tensor (writes go
   through), so the caller must not free the tensor while the view is live; a Gc
   finaliser anchors the handle for as long as the view exists. The [Dtype.t] tag
   ties the element/kind, so the returned array's type is checked at compile time. *)
let data : type a b. (a, b) Dtype.t -> _ -> (a, b, c_layout) Array1.t option =
 fun dt t ->
  let vp = F.data_ptr t (Dtype.to_int dt) in
  if is_null vp then None
  else
    let p = from_voidp (Dtype.typ dt) vp in
    let ba = bigarray_of_ptr array1 (numel t) (Dtype.kind dt) p in
    Gc.finalise (fun _ -> ignore t) ba;
    Some ba

let as_float32 t : float32_array option = data Dtype.float32 t

(* A managed tensor of [shape] and dtype [dt], copied from the 1-D [src] (whose
   element count must equal the shape's). The tensor owns its own buffer, so
   later mutating [src] does not affect it. *)
let of_bigarray : type a b.
    (a, b) Dtype.t -> (a, b, c_layout) Array1.t -> int list -> _ =
 fun dt src shape ->
  let t = create ~dtype:(Dtype.scalar_type dt) shape in
  (match data dt t with
  | Some dst ->
      if Array1.dim dst <> Array1.dim src then
        raise (Error "of_bigarray: element-count mismatch");
      Array1.blit src dst
  | None -> raise (Error "of_bigarray: dtype mismatch"));
  t

(* Extract the single element of a one-element tensor. Raises [Error] otherwise. *)
let item_float t =
  let out = allocate double 0.0 in
  if F.item_double t out = 0 then
    raise (Error (Option.value (F.last_error ()) ~default:"item failed"));
  !@out

let item_int t =
  let out = allocate int64_t 0L in
  if F.item_int64 t out = 0 then
    raise (Error (Option.value (F.last_error ()) ~default:"item failed"));
  Int64.to_int !@out

let pp_float32 fmt (ba : float32_array) =
  let seq = Seq.init (Array1.dim ba) (fun i -> ba.{i}) in
  Format.fprintf fmt "[%a]"
    (Format.pp_print_seq
       ~pp_sep:(fun fmt () -> Format.pp_print_string fmt "; ")
       (fun fmt v -> Format.fprintf fmt "%g" v))
    seq

(* Canonical rendering via ATen's tensor printer (multi-line, with a dtype/shape
   footer). [pp] is the Format pretty-printer over it. *)
let to_string t =
  match F.to_string t with
  | Some s -> s
  | None -> raise (Error (Option.value (F.last_error ()) ~default:"to_string"))

let pp fmt t = Format.pp_print_string fmt (to_string t)

(* [r] is the shim's bool result: 1/0, or -1 (error). *)
let to_bool name r =
  if r < 0 then raise (Error (Option.value (F.last_error ()) ~default:name))
  else r <> 0

(* Elementwise closeness (ATen at::allclose): |a-b| <= atol + rtol*|b|. *)
let allclose ?(rtol = 1e-5) ?(atol = 1e-8) ?(equal_nan = false) a b =
  to_bool "allclose" (F.allclose a b rtol atol (if equal_nan then 1 else 0))

(* Exact elementwise equality (ATen at::equal); false on shape/dtype mismatch. *)
let equal a b = to_bool "equal" (F.equal a b)
