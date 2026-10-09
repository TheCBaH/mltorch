(** Local files as the byte sources {!Pt2_checkpoint_map.Prepare} verifies. *)

type error = [ `Source_io of string * string ]
(** The path and the operating system's message: third-party text. *)

val pp_error : error Fmt.t

val map_file : string -> (Safetensors.Bigstring.t, [> error ]) Err.t
(** Map a whole file read-only and copy-on-write. The mapping lives as long as
    the bigstring or any view of it; the file must not change while it is
    mapped. An empty file is an empty bigstring. *)

val sources_in_dir :
  Pt2_checkpoint_map.Document.t ->
  dir:string ->
  (Pt2_checkpoint_map.Prepare.source list, [> error ]) Err.t
(** Map [dir/NAME] for every file the document declares, checkpoint files and
    the graph-owned file. A missing file is {!error}; nothing else in [dir] is
    touched. *)
