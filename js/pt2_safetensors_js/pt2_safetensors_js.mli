(** Fetch the checkpoint a [safetensors.json] pins, through hf-hub's JavaScript
    driver, under js_of_ocaml.

    The page or node must have loaded a host first: [globalThis.hfHubHost]
    performing the driver's operations, plus a [bytes] method returning a
    downloaded file's contents as a [Uint8Array]. Node's host is [node-host.cjs]
    and a browser's [browser-host.js], both from the hf-hub submodule's
    [javascript/] directory. *)

module Host : sig
  type t = {
    cache_dir : string;  (** Where the host keeps its cache. *)
    endpoint : string;  (** The Hub, or a metadata proxy for a browser. *)
  }
end

type error = [ Pt2_safetensors.error | `Hub of string ]

val pp_error : Format.formatter -> error -> unit

val fetch :
  Host.t ->
  Pt2_safetensors.Checkpoint_map.Source.t ->
  ((string, [> error ]) Err.t -> unit) ->
  unit
(** The pinned checkpoint's bytes, once downloaded and checked against the pin
    ({!Pt2_safetensors.check_source}). The continuation runs on a later turn of
    the event loop: the host is asynchronous. *)
