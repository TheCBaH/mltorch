(* The LOADER of Loop_c_embed on a Coral NPU simulator. The unit is compiled and
   lowered once, by the F-only RV32 loader (CompCert's units and the binary32
   C library, under the binary32 profile); a call builds the guest image around
   the caller's region, one [In_out] slot passed to [entry] as its address, and
   runs it: the region goes in through the input frame and comes back in the
   result frame. The image is kept for the next call of the same size. *)

module Q = Loop_c_embed_compcert_rv32.Compcert_rv32_loader.Ilp32f

(* Where and on what the unit runs: the extracted emulators archive, one of its
   configurations, and the wall-clock limit of one call, in seconds. *)
module type CONFIG = sig
  val dir : string
  val configuration : string
  val deadline : float
end

module Make (C : CONFIG) = struct
  type loaded = {
    lowered : Riscv32_encode.fixup_kind Asm_core.Lowered_ast.module_ list;
    profile : Coral_profile.t;
    mutable image : (int * Coral_image.t) option;  (** by region size *)
  }

  type error = string

  let dialect = Q.dialect
  let pp_error = Fmt.string
  let ( let* ) = Result.bind
  let render pp r = Result.map_error (Fmt.str "%a" pp) (Err.payload r)

  let load ~host_symbols source =
    let* q = Q.load ~host_symbols source in
    let* profile =
      render Coral_profile.pp_error
        (Coral_profile.load ~dir:C.dir C.configuration)
    in
    Ok { lowered = q.Q.lowered; profile; image = None }

  let image l size =
    match l.image with
    | Some (n, image) when n = size -> Ok image
    | _ ->
        let slot =
          {
            Coral_layout.Slot.align = 64L;
            placement = In_out;
            size = Int64.of_int size;
          }
        in
        let* image =
          render Coral_image.pp_error
            (Coral_image.build ~abi:Slot0 ~lowered:l.lowered l.profile
               ~entry:"entry" ~kernel:[] [ slot ])
        in
        l.image <- Some (size, image);
        Ok image

  let call l ~io =
    let size = Bigarray.Array1.dim io in
    let* image = image l size in
    let region = String.init size (fun i -> Bigarray.Array1.get io i) in
    let* outcome =
      render Coral_run.Run_error.pp
        (Coral_run.run ~deadline:C.deadline ~dir:C.dir image ~bound:(fun k ->
             if k = 0 then Some region else None))
    in
    match List.assoc_opt 0 outcome.outputs with
    | None -> Error "the guest returned no region"
    | Some bytes ->
        String.iteri (fun i c -> Bigarray.Array1.set io i c) bytes;
        Ok (Int64.of_int32 outcome.status)

  let close (_ : loaded) = ()
end
