(* The LOADER of Loop_c_embed over an embedded CompCert: the unit is compiled in
   process by this target's CompCert, assembled by rivet, bound to the host's
   libc and libm through trampolines and mapped executable by [Native_exec].
   [Compcert_embed_target] is rivet-compcert's embedding for the host's ISA. *)

type loaded = Native_exec.loaded
type error = string

let pp_error = Fmt.string

let load ~host_symbols source =
  match
    Compcert_embed_target.load ~entry:"entry" ~host_symbols ~name:"model.c"
      source
  with
  | Ok loaded -> Ok loaded
  | Error e -> Error (Format.asprintf "%a" Compcert_embed.pp_error e)

let call loaded ~io =
  Native_exec.call loaded ~io
  |> Result.map_error (Format.asprintf "%a" Native_exec.pp_error)

let close = Native_exec.close
