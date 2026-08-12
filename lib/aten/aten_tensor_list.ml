(* The OCaml side of the owned Tensor[] result container (atc_tensor_list in
   atg_shim.h), produced by the generated wrapper of any op whose schema return
   is [Tensor[]] -- at::unbind is the first.

   Two ownership levels, deliberately distinct: the CONTAINER owns its
   std::vector<at::Tensor>, while each extracted handle owns itself. at::Tensor
   is intrusive-reference-counted, so freeing the container drops only its own
   handles; the tensors handed to OCaml stay valid, and they keep ATen's storage
   sharing, offset and strides (they are views, not copies -- see
   .ai/aten_core_build.md on the flat-data boundary). *)

module F = Aten_c.Aten_functions

(* Raise [Aten_tensor.Error] if [l] is the failure sentinel (a null handle).
   Reuses Aten_tensor's exception rather than introducing a second one: a caller
   catching a boundary failure should not care which shim call produced it. *)
let check l =
  if Ctypes.is_null l then
    raise
      (Aten_tensor.Error
         (Option.value (F.last_error ()) ~default:"aten tensor list error"))
  else l

(* Convert an owned list into managed OCaml handles, freeing the container.

   Every element is [Aten_tensor.check]ed and [Aten_tensor.manage]d exactly
   once, so the caller receives tensors it already owns and the container is
   gone by the time this returns. [Fun.protect] means that holds on the failure
   paths too: an element that fails mid-way still frees the container, the
   handles extracted before it are managed (the GC will free them), and the ones
   after it die with the container. Nothing leaks on any path. *)
let to_list l =
  let l = check l in
  Fun.protect
    ~finally:(fun () -> F.tensor_list_free l)
    (fun () ->
      let n = F.tensor_list_len l in
      (* Keep the two failures apart. The -1 sentinel carries a real diagnostic
         from the shim; replacing it with generic text would lose the only
         explanation of what went wrong. The range check is separate because
         [List.init] would otherwise raise [Invalid_argument] out of a module
         whose documented failure is [Aten_tensor.Error]. *)
      if n < 0L then
        raise
          (Aten_tensor.Error
             (Option.value (F.last_error ()) ~default:"tensor list: bad length"));
      if n > Int64.of_int max_int then
        raise (Aten_tensor.Error "tensor list: length exceeds OCaml int");
      List.init (Int64.to_int n) (fun i ->
          F.tensor_list_get l (Int64.of_int i)
          |> Aten_tensor.check |> Aten_tensor.manage))

(* Number of live list containers (allocated, not yet freed). The tensor
   [Aten_tensor.live_count] does not cover these, so the leak tests assert both:
   this one returns to baseline synchronously (Fun.protect frees it), the tensor
   count only after the GC has run the handles' finalisers. *)
let live_count () = Int64.to_int (F.tensor_list_live_count ())
