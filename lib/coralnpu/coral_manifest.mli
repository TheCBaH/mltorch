(** The subset of a Coral release's [hardware-configurations.json] that running
    a guest needs, decoded with every configuration's ID recomputed.

    An ID is [coralnpu:<name>:<hex>], where [<hex>] is the first twenty
    hexadecimal digits of the SHA-256 of the configuration record without its
    [id], in the producer's canonical JSON ({!Coral_json}). A record whose
    recomputed ID differs from the declared one is refused, so a decoded
    configuration is the one its ID names. *)

module Access : sig
  type t = {
    dma : bool option;
    guest_load : bool option;
    guest_store : bool option;
    host_load : bool option;
    instruction_fetch : bool option;
  }
  (** What may touch a region. [None] is the manifest's [null]: unknown. *)
end

module Region : sig
  type t = {
    access : Access.t;
    base_address : int64;
    kind : string;  (** The manifest's [type]: [itcm], [dtcm], [ram], ... *)
    name : string;
    size_bytes : int64;
  }
end

module Configuration : sig
  type t = {
    backend : string;  (** [simulation.backend]: [mpact], [verilator]. *)
    extensions : string list;  (** [isa.extensions], lower case. *)
    id : string;
    interface : string;
        (** [simulation.interface]: [standalone-cli], [subsystem-cli]. *)
    name : string;
    regions : Region.t list;
  }
end

module Artifact : sig
  type t = {
    command : string list;
        (** [launch.command]: the path first, [<elf>] where the guest goes. *)
    configuration_id : string option;
    path : string;  (** Relative to the archive root. *)
    sha256 : string;
  }
end

type t = {
  archive : string;  (** [archive.name]. *)
  artifacts : Artifact.t list;
  configurations : Configuration.t list;
}

module Id_mismatch : sig
  type t = { computed : string; declared : string }
end

type error =
  [ Coral_json.error
  | `Id_mismatch of Id_mismatch.t
  | `Manifest_decode of string  (** Jsont's message. *)
  | `Schema_version of string ]

val pp_error : error Fmt.t

val configuration_id : Jsont.json -> (string, [> error ]) Err.t
(** The ID of a configuration record, from its [name] and every member but [id].
*)

val of_string : string -> (t, [> error ]) Err.t
(** Decodes a manifest of schema version 1.0 and checks every configuration's
    ID. *)
