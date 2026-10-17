(** A runnable Coral simulator: one manifest configuration, the artifact that
    simulates it, and what a guest image may do with each memory region.

    [loadable] (an ELF PT_LOAD may lie there) is the manifest's [host_load] of a
    memory (TCM or RAM); unknown counts as no, and a CSR or MMIO block the host
    may write is never loadable. [htif_reachable] (an HTIF [read]/[write] buffer
    may lie there) is mltorch's, because the manifest does not record it and
    running guests settled it: DTCM and the external RAM on both simulators;
    every other region is treated as unreachable until a run shows otherwise. *)

module Role : sig
  (** The manifest's region [type]. *)
  type t = Csr | Dtcm | Itcm | Mmio | Ram

  val to_string : t -> string
  (** The manifest's spelling. *)
end

module Simulator : sig
  type t = Mpact | Rtl  (** Manifest backends [mpact] and [verilator]. *)
end

module Region : sig
  type t = {
    base : int64;
    htif_reachable : bool;
    loadable : bool;
    name : string;
    role : Role.t;
    size : int64;
    writable : bool;  (** The manifest says guest stores succeed. *)
  }
end

type t = {
  artifact : string;  (** Path relative to the archive root. *)
  artifact_sha256 : string;
  command : string list;  (** The manifest's launch command, [<elf>] a hole. *)
  configuration_id : string;
  extensions : string list;
  name : string;
  regions : Region.t list;
  simulator : Simulator.t;
}

module Artifact_digest : sig
  type t = { actual : string; expected : string; path : string }
end

module Artifact_read : sig
  type t = { message : string; path : string }
end

type error =
  [ Coral_manifest.error
  | `Artifact_digest of Artifact_digest.t
  | `Artifact_read of Artifact_read.t  (** The system's message. *)
  | `No_artifact of string  (** A configuration ID no artifact simulates. *)
  | `Unknown_backend of string
  | `Unknown_configuration of string
  | `Unknown_region_kind of string ]

val pp_error : error Fmt.t

val baseline : string list
(** The configurations the Coral route runs: [core-mini-axi],
    [core-mini-highmem-axi], [mpact-v2]. *)

val manifest_file : string
(** [hardware-configurations.json], at the archive root. *)

val of_manifest : Coral_manifest.t -> string -> (t, [> error ]) Err.t
(** The profile of the named configuration. *)

val verify : dir:string -> t -> (unit, [> error ]) Err.t
(** Checks the artifact under the archive root [dir] against its SHA-256. *)

val load : dir:string -> string -> (t, [> error ]) Err.t
(** Reads [dir]'s manifest, builds the named profile and verifies its artifact.
*)

val argv : dir:string -> elf:string -> t -> string list
(** The command for one run: the artifact under [dir], [elf] in the hole. *)

val pp : t Fmt.t
(** Name, simulator, ID, extensions and one line per region. *)
